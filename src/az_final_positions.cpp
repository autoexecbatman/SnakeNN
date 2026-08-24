// AlphaZeroFinalPositions: what does the board look like at the moment the snake dies?
//
// A diagnostic, not part of training or evaluation. Four arms - more simulations, ten
// iterations of bootstrapped death labels, and the tail-seal guard - all ended with the
// snake dead between 38 and 43 percent of the board filled, however differently they
// played. The trap guard is the sharpest case: it bought 48 percent longer games and not
// one extra apple. So the question stopped being "can the search see further" and became
// "what is on the board at 40 percent that ends the game", and no summary statistic
// answers that. This prints the boards.
//
// What it reports, per game, from the position the fatal move was played out of:
//
//   - the board as text, so it can be read rather than inferred
//   - empty cells, and how many of them the head can still reach
//   - the reachable share, which is the fragmentation measure: 1.00 means the free
//     space is one region and the snake merely mis-stepped, while 0.20 means it was
//     already sealed into a fifth of the space and the death was decided earlier
//   - whether the food was inside that reachable region at all
//   - steps since the last apple, which separates starving from being trapped
//
// Then the same numbers aggregated over every game, which is what decides whether the
// death is a local blunder or a structural end.
//
// It plays under evaluation conditions - greedy, no root noise, reserved seeds - so the
// positions are the ones the measured runs actually reach. The search settings are taken
// from search_defaults so they cannot drift from what az_evaluate uses.
//
// Usage:
//
//     AlphaZeroFinalPositions.exe --checkpoint az20_steps340.pt --board 20 --games 100
//        --simulations 200 --step-limit 9600 --search-seed 777 --show 8
//
//     # --games 100      games played, all of them aggregated
//     # --show 8         how many boards are printed in full; 0 prints none
//     # --search-seed    fixes the search stream, so a rerun reproduces the boards
//     # --trap-guard on  play with the seal veto, to dump that arm's deaths instead
//
// Exit codes: 2 for a bad flag, 1 for a checkpoint that cannot be read, 0 otherwise. A
// game that wins or times out is counted and reported separately rather than dumped -
// this tool is about deaths.

#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <format>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "az_network.h"
#include "az_parameters.h"
#include "flag_parser.h"
#include "mcts.h"
#include "network_evaluator.h"
#include "network_setup.h"
#include "search_defaults.h"
#include "seed_policy.h"
#include "snake_env.h"

namespace
{

// What one run was asked to do. Every field has a flag; nothing here is defaulted from
// a constant that a reader would have to go and find.
struct Settings
{
    // The checkpoint to play with. No default - a run without one has nothing to say.
    std::string checkpoint;
    // Board side. Square, like every other program here.
    int board{ 20 };
    // Games played, all aggregated.
    int games{ 100 };
    // Search budget per move, matched to evaluation.
    int simulations{ 200 };
    // Longest a game may run, in moves. Zero means derive it from the board.
    int step_limit{ 0 };
    // Offset into the reserved evaluation band.
    unsigned int seed_offset{ 0 };
    // The search's random stream, fixed so a rerun reproduces the boards.
    unsigned int search_seed{ 777 };
    // How many boards are printed in full. The rest contribute to the aggregate only.
    int show{ 8 };
    // Trunk width and depth, which must match the checkpoint that is being loaded.
    int channels{ 64 };
    int blocks{ 4 };
    // Whether the seal veto is on, so the guard arm's deaths can be dumped too.
    bool trap_guard{ false };
};

// What one dead game looked like at its last decision.
struct Death
{
    // Apples eaten, and moves taken, when the fatal move was chosen.
    int score{ 0 };
    int steps{ 0 };
    // Cells not occupied by the snake.
    int empty{ 0 };
    // The largest region any legal move opened onto, on the last move that had one.
    // Zero here would mean the game began already lost, so it is a sanity value.
    int space_before_death{ 0 };
    // Empty cells in the whole board on that same move, for comparison with the above:
    // space far below empty is a head enclosed while the board is still open.
    int empty_before_death{ 0 };
    // The move at which the head's available space last fell below the snake's own
    // length and never recovered. A region smaller than the body cannot hold it, so
    // from here the death is arithmetic rather than a blunder. -1 if it never happened.
    int doomed_at{ -1 };
    // Moves played between that point and the death.
    int doom_to_death{ 0 };
    // Moves since the last apple, which separates starving from being trapped.
    int since_food{ 0 };
};

// What one position offers the head, and whether it is a position a game can end at.
struct Room
{
    // The largest region any surviving action opens onto. Zero means every move loses.
    int space{ 0 };
    // Whether some action dies this tick. False means no choice here can end the game,
    // so the position is not worth keeping.
    bool any_move_loses{ false };
};

// The largest region any legal move from `game` opens onto, and zero when every move
// loses.
//
//     const Room room = availableSpace(game);   // space 0 means dead whatever it plays
//
// Reachability is taken over actions rather than from the head's own cell, because the
// head sits on a body cell and a flood fill from there would report nothing.
Room availableSpace(const SnakeEnv& game)
{
    Room room;
    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
    {
        const SnakeEnv::Action move = static_cast<SnakeEnv::Action>(action);
        // A move that dies this tick opens onto nothing worth counting, and its
        // existence is what makes the position one a game can end at.
        if (game.wouldDie(move))
        {
            room.any_move_loses = true;
            continue;
        }
        room.space = std::max(room.space, game.reachableCells(move));
    }
    return room;
}

// Parses argv into Settings, throwing std::invalid_argument on anything unrecognised.
//
//     const Settings settings = parseSettings({ "--checkpoint", "az20.pt", "--games", "8" });
//
// Every flag takes a value; a flag without one throws rather than defaulting.
Settings parseSettings(std::span<const std::string> arguments)
{
    Settings settings;
    // Two at a time: a name and its value.
    for (std::size_t index = 0; index + 1 < arguments.size(); index += 2)
    {
        const std::string& name = arguments[index];
        const std::string& value = arguments[index + 1];
        if (name == "--checkpoint")
        {
            settings.checkpoint = value;
        }
        else if (name == "--board")
        {
            settings.board = flags::parseWholeInt(name, value);
        }
        else if (name == "--games")
        {
            settings.games = flags::parseWholeInt(name, value);
        }
        else if (name == "--simulations")
        {
            settings.simulations = flags::parseWholeInt(name, value);
        }
        else if (name == "--step-limit")
        {
            settings.step_limit = flags::parseWholeInt(name, value);
        }
        else if (name == "--seed-offset")
        {
            settings.seed_offset = flags::parseWholeUnsigned(name, value);
        }
        else if (name == "--search-seed")
        {
            settings.search_seed = flags::parseWholeUnsigned(name, value);
        }
        else if (name == "--show")
        {
            settings.show = flags::parseWholeInt(name, value);
        }
        else if (name == "--channels")
        {
            settings.channels = flags::parseWholeInt(name, value);
        }
        else if (name == "--blocks")
        {
            settings.blocks = flags::parseWholeInt(name, value);
        }
        else if (name == "--trap-guard")
        {
            settings.trap_guard = flags::parseOnOff(name, value);
        }
        else
        {
            throw std::invalid_argument(std::format("unknown flag {}", name));
        }
    }
    // An odd argument count means a flag arrived without its value.
    if (arguments.size() % 2 != 0)
    {
        throw std::invalid_argument("every flag needs a value");
    }
    if (settings.checkpoint.empty())
    {
        throw std::invalid_argument("--checkpoint is required");
    }
    return settings;
}

// The board as text: H the head, o the body, * the food, . an empty cell.
//
//     std::cout << render(game);
//
// One line per row, top row first, so it reads the way the grid is drawn.
std::string render(const SnakeEnv& game)
{
    // Start empty, then paint the snake and the food over it.
    std::string grid(static_cast<std::size_t>(game.width() * game.height()), '.');
    const std::vector<Position>& body = game.body();
    for (std::size_t segment = 0; segment < body.size(); segment++)
    {
        const std::size_t cell =
            static_cast<std::size_t>(body[segment].y * game.width() + body[segment].x);
        // body[0] is the head; everything behind it is trunk.
        grid[cell] = (segment == 0) ? 'H' : 'o';
    }
    const std::size_t food_cell =
        static_cast<std::size_t>(game.food().y * game.width() + game.food().x);
    grid[food_cell] = '*';

    // Cut the flat string into rows.
    std::string out;
    for (int row = 0; row < game.height(); row++)
    {
        out += "  ";
        out += grid.substr(static_cast<std::size_t>(row * game.width()),
                           static_cast<std::size_t>(game.width()));
        out += '\n';
    }
    return out;
}

// The free space the head can still reach, and whether the food is in it.
//
//     const Death measured = measure(game_before_the_fatal_move);
//
// Reachability is taken over the best action available: the largest region any legal
// move leads into. A snake with one good move left is not yet trapped, and scoring it
// as trapped would blame the position for what the next move does.
Death measure(const SnakeEnv& game)
{
    Death death;
    death.score = game.score();
    death.steps = game.steps();
    death.empty = game.cellCount() - static_cast<int>(game.body().size());
    death.since_food = game.stepsSinceFood();
    return death;
}

// What one game's trajectory recorded on the way to its end.
struct Trace
{
    // The last position that offered the head anywhere to go, and the board's free cells
    // at that same moment. Comparing the two is the measurement: room far below empty is a
    // head enclosed while the board is still open.
    int last_space{ 0 };
    int last_empty{ 0 };
    // The move at which the head's room last fell below the snake's own length without
    // recovering. -1 if that never happened.
    int doomed_at{ -1 };
};

// The search every arm of this tool uses: evaluation conditions, greedy, no root noise.
//
//     MonteCarloSearch search(evaluator, searchConfig(settings));
//
// Built from az::paperSearchDefaults so the eight shared constants cannot drift from what
// az_evaluate searches with; only the four fields this tool decides are set here.
MonteCarloSearch::Config searchConfig(const Settings& settings)
{
    MonteCarloSearch::Config config = az::paperSearchDefaults();
    config.simulations = settings.simulations;
    // Off, because a diagnostic reads the policy rather than exploring around it.
    config.root_noise_fraction = 0.0f;
    config.trap_guard = settings.trap_guard;
    config.seed = settings.search_seed;
    return config;
}

// One line of progress, rewritten in place, at most once a second.
//
//     drawProgress(pass, live_count, settings.games, last_drawn, last_width);
//
// Games finish at wildly different times, so the live count falling is the real signal
// rather than the pass number. `last_width` is carried so the next write wipes exactly the
// last line; a fixed-width wipe leaves the tail of anything longer on screen.
void drawProgress(int pass, std::size_t live, int total,
                  std::chrono::high_resolution_clock::time_point& last_drawn,
                  std::size_t& last_width)
{
    const auto now = std::chrono::high_resolution_clock::now();
    if (std::chrono::duration<double>(now - last_drawn).count() < 1.0)
    {
        return;
    }
    last_drawn = now;
    const std::string line = std::format("  pass {}  live {}/{}  finished {}", pass, live, total,
                                         total - static_cast<int>(live));
    std::cout << '\r' << line
              << std::string(last_width > line.size() ? last_width - line.size() : 0, ' ')
              << std::flush;
    last_width = line.size();
}

// Plays every game to its end in lockstep, filling `before` and `traces`.
//
//     playAll(search, settings, step_limit, games, before, traces);
//
// The batch is not a speed setting: mcts draws food placement across the whole batch from
// one generator, so a different number of games in flight searches different futures and
// reaches different deaths. All of them are played together for that reason, not for
// throughput.
//
// `before` ends holding, per game, the position its last losing choice was made from -
// saved only where some move loses, since a position no move can die from is never the one
// a death is read out of.
void playAll(MonteCarloSearch& search, const Settings& settings, int step_limit,
             std::vector<SnakeEnv>& games, std::vector<SnakeEnv>& before,
             std::vector<Trace>& traces)
{
    auto last_drawn = std::chrono::high_resolution_clock::now();
    std::size_t last_width = 0;

    // Bounded rather than open: every live game advances one move per pass, so the batch
    // cannot outlast step_limit passes. The break below is the normal exit.
    for (int pass = 0; pass < step_limit; pass++)
    {
        std::vector<int> live;
        std::vector<const SnakeEnv*> roots;
        for (int index = 0; index < settings.games; index++)
        {
            if (games[index].done() || games[index].steps() >= step_limit)
            {
                continue;
            }
            live.push_back(index);
            roots.push_back(&games[index]);
        }
        if (live.empty())
        {
            break;
        }

        drawProgress(pass, live.size(), settings.games, last_drawn, last_width);

        const std::vector<MonteCarloSearch::Result> results = search.search(roots);
        for (std::size_t position = 0; position < live.size(); position++)
        {
            const int index = live[position];
            // Sampled before the move, so it describes the position the search chose from
            // rather than the one it landed in.
            const Room room = availableSpace(games[index]);
            const int body = static_cast<int>(games[index].body().size());
            if (room.space > 0)
            {
                traces[index].last_space = room.space;
                traces[index].last_empty = games[index].cellCount() - body;
            }
            // A region smaller than the snake cannot hold it, so this is where the loss
            // becomes arithmetic. Reset when the head gets back into room, so what is
            // reported is the onset that stuck rather than the first scare.
            if (room.space < body)
            {
                if (traces[index].doomed_at < 0)
                {
                    traces[index].doomed_at = games[index].steps();
                }
            }
            else
            {
                traces[index].doomed_at = -1;
            }
            // Only where some action loses: copying an environment per game per move is
            // this tool's whole overhead.
            if (room.any_move_loses)
            {
                before[index] = games[index];
            }
            games[index].step(results[position].best_action);
        }
    }
    // Wiped before any board lands, or the first one is printed over the bar's tail.
    std::cout << '\r' << std::string(last_width, ' ') << '\r' << std::flush;
}

// One death, as a line of numbers and the board it happened on.
//
//     reportDeath(3, death, before[3]);
//
// The board is printed so it can be read rather than inferred; the numbers above it are
// what the aggregate is built from.
void reportDeath(int index, const Death& death, const SnakeEnv& position)
{
    std::cout << std::format(
        "\ngame {}  score {}  steps {}  empty {}  last room {} of {} free"
        "  ({:.2f})  doomed at {}  {} moves before death  since food {}\n",
        index, death.score, death.steps, death.empty, death.space_before_death,
        death.empty_before_death,
        death.empty_before_death > 0 ? static_cast<double>(death.space_before_death) /
                                           static_cast<double>(death.empty_before_death)
                                     : 0.0,
        death.doomed_at, death.doom_to_death, death.since_food);
    std::cout << render(position);
}

// The totals over every death, which is what decides between a local blunder and a
// structural end.
//
//     reportAggregate(deaths);
//
// Prints nothing when there were none, so a run of wins and timeouts does not divide by
// zero to say so.
void reportAggregate(const std::vector<Death>& deaths)
{
    if (deaths.empty())
    {
        return;
    }
    double score_total = 0.0;
    double empty_total = 0.0;
    double room_total = 0.0;
    double share_total = 0.0;
    double doom_total = 0.0;
    int enclosed = 0;
    int doomed_count = 0;
    for (const Death& death : deaths)
    {
        score_total += death.score;
        empty_total += death.empty;
        room_total += death.space_before_death;
        const double share = death.empty_before_death > 0
                                 ? static_cast<double>(death.space_before_death) /
                                       static_cast<double>(death.empty_before_death)
                                 : 0.0;
        share_total += share;
        // A head with under a fifth of the board's free space was enclosed rather than
        // surprised: the room ran out around it while the board stayed open.
        if (share < 0.2)
        {
            enclosed++;
        }
        if (death.doomed_at >= 0)
        {
            doomed_count++;
            doom_total += death.doom_to_death;
        }
    }
    const double count = static_cast<double>(deaths.size());
    std::cout << std::format("mean score        {:.2f}\n", score_total / count);
    std::cout << std::format("mean empty        {:.2f}\n", empty_total / count);
    std::cout << std::format("mean last room    {:.2f}\n", room_total / count);
    std::cout << std::format(
        "mean room share   {:.4f}  (1.00 = the head owned all the free space)\n",
        share_total / count);
    std::cout << std::format("enclosed under 0.2 share: {} of {}\n", enclosed, deaths.size());
    std::cout << std::format(
        "room fell below body length in {} of {} games, a mean of "
        "{:.1f} moves before death\n",
        doomed_count, deaths.size(), doomed_count > 0 ? doom_total / doomed_count : 0.0);
}

}  // namespace

int main(int argc, char** argv)
{
    // Parses argv into settings; a bad flag exits 2 before any game is played.
    Settings settings;
    try
    {
        settings = parseSettings(std::vector<std::string>(argv + 1, argv + argc));
    }
    catch (const std::exception& error)
    {
        std::cerr << "bad arguments: " << error.what() << std::endl;
        return 2;
    }

    // Zero means derive the budget from the board, the same rule every program uses.
    const int step_limit =
        settings.step_limit > 0 ? settings.step_limit : az::deriveStepLimit(settings.board);

    // CUDA when available, else the CPU.
    const Compute compute = chooseDevice();

    // The checkpoint under test, loaded onto the device in eval mode.
    AlphaZeroNet network(settings.board, settings.board, settings.channels, settings.blocks);
    try
    {
        network = loadForEvaluation(settings.board, settings.channels, settings.blocks,
                                    settings.checkpoint, "", compute.device);
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }

    std::cout << std::format(
        "final positions: {} on {}x{}, {} games, {} simulations, "
        "step limit {}, trap guard {}\n",
        settings.checkpoint, settings.board, settings.board, settings.games, settings.simulations,
        step_limit, settings.trap_guard ? "on" : "off");

    NetworkEvaluator evaluator(network, compute.device);
    MonteCarloSearch search(evaluator, searchConfig(settings));

    // The same reserved band every measured run uses, so these are those games.
    std::vector<SnakeEnv> games;
    games.reserve(static_cast<std::size_t>(settings.games));
    for (int index = 0; index < settings.games; index++)
    {
        games.emplace_back(settings.board, settings.board,
                           seeds::evaluationGameSeed(settings.seed_offset, index), step_limit);
    }
    std::vector<SnakeEnv> before = games;
    std::vector<Trace> traces(static_cast<std::size_t>(settings.games));

    const auto started = std::chrono::high_resolution_clock::now();
    playAll(search, settings, step_limit, games, before, traces);

    // Walked by game index rather than by finish order, so a rerun prints the same boards
    // in the same order.
    std::vector<Death> deaths;
    int won = 0;
    int timed_out = 0;
    for (int index = 0; index < settings.games; index++)
    {
        if (games[index].won())
        {
            won++;
            continue;
        }
        if (!games[index].done())
        {
            timed_out++;
            continue;
        }
        Death death = measure(before[index]);
        death.space_before_death = traces[index].last_space;
        death.empty_before_death = traces[index].last_empty;
        death.doomed_at = traces[index].doomed_at;
        death.doom_to_death =
            traces[index].doomed_at >= 0 ? death.steps - traces[index].doomed_at : 0;
        deaths.push_back(death);
        if (static_cast<int>(deaths.size()) <= settings.show)
        {
            reportDeath(index, death, before[index]);
        }
    }

    std::cout << std::format("\n{} deaths, {} wins, {} timeouts\n", deaths.size(), won, timed_out);
    reportAggregate(deaths);
    std::cout << std::format(
        "\nelapsed {:.2f}s\n",
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - started).count());
    return 0;
}
