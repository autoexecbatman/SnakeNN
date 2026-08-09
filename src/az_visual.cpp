#include <torch/torch.h>
#include <raylib.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "az_network.h"
#include "az_parameters.h"
#include "board_render.h"
#include "eval_options.h"
#include "mcts.h"
#include "network_evaluator.h"
#include "snake_env.h"

// Watch a trained network play.
//
// Same conditions as AlphaZeroEvaluate - root noise off, visit-count argmax - so
// what is on screen is what the win-rate number describes. A demo that plays
// under self-play settings would be showing a different agent from the one being
// reported, which is the sort of gap nobody notices until the number is quoted
// somewhere it matters.

namespace
{

// Height in pixels of the statistics panel below the board.
constexpr int PANEL_HEIGHT = 150;
// Narrowest the window is allowed to get, so the header text still fits.
constexpr int MIN_WINDOW_WIDTH = 440;

void drawStatTile(const ui::Fonts& fonts, const char* label, const char* value, float x, float y,
                  float width, Color value_color)
{
    ui::drawLabel(fonts.label, label, x, y, ui::TEXT_MUTED);
    float value_width = ui::textWidth(fonts.mono, value, 21.0f, 0.5f);
    ui::drawText(fonts.mono, value, x + (width - value_width) / 2.0f - 4.0f, y + 17.0f, 21.0f, 0.5f,
                 value_color);
}

}  // namespace

int main(int argc, char** argv)
{
    visual::Settings settings;
    try
    {
        settings = visual::parseArguments(std::vector<std::string>(argv + 1, argv + argc));
    }
    catch (const std::invalid_argument& error)
    {
        std::cerr << error.what() << std::endl;
        std::cerr << "usage: --checkpoint <file> [--board N] [--simulations N] [--seed N]"
                  << std::endl;
        return 2;
    }
    const int step_limit = settings.stepLimit();

    torch::Device device =
        torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    AlphaZeroNet network(settings.board, settings.board, settings.channels, settings.blocks);
    try
    {
        torch::load(network, settings.checkpoint);
    }
    catch (const std::exception& error)
    {
        std::cerr << "could not load " << settings.checkpoint << ": " << error.what() << std::endl;
        return 1;
    }
    network->to(device);
    network->eval();

    const int board_width = ui::boardPixelWidth(settings.board);
    const int window_width = std::max(MIN_WINDOW_WIDTH, board_width + ui::MARGIN * 2);
    const int window_height =
        ui::HEADER_HEIGHT + ui::boardPixelHeight(settings.board) + PANEL_HEIGHT;

    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(window_width, window_height, "Snake - Learned Agent");
    SetTargetFPS(60);
    ui::Fonts fonts = ui::loadFonts();

    NetworkEvaluator evaluator(network, device);
    MonteCarloSearch::Config search_config;
    search_config.simulations = settings.simulations;
    search_config.exploration = az::EXPLORATION;
    search_config.discount = az::DISCOUNT;
    // Off, as in the evaluator, so what is on screen is the agent the win rate
    // describes rather than the exploration policy.
    search_config.root_noise_fraction = 0.0f;
    search_config.root_noise_alpha = az::ROOT_NOISE_ALPHA;
    search_config.seed = settings.seed;
    MonteCarloSearch search(evaluator, search_config);

    unsigned int game_seed = settings.seed;
    SnakeEnv game(settings.board, settings.board, game_seed);
    int games_played = 0;
    int wins = 0;
    float last_value = 0.0f;
    bool finished = false;

    while (!WindowShouldClose())
    {
        if (finished && IsKeyPressed(KEY_SPACE))
        {
            game_seed++;
            game = SnakeEnv(settings.board, settings.board, game_seed);
            finished = false;
        }

        for (int move = 0; move < settings.moves_per_frame && !finished; move++)
        {
            if (game.done() || game.steps() >= step_limit)
            {
                games_played++;
                wins += game.won() ? 1 : 0;
                finished = true;
                break;
            }
            std::vector<const SnakeEnv*> roots{&game};
            std::vector<MonteCarloSearch::Result> results = search.search(roots);
            last_value = results[0].value;
            game.step(results[0].best_action);
        }

        BeginDrawing();
        ClearBackground(ui::BACKGROUND);

        ui::drawText(fonts.display, "LEARNED AGENT", static_cast<float>(ui::MARGIN), 20.0f, 27.0f,
                     0.5f, ui::TEXT_PRIMARY);
        ui::drawText(fonts.display,
                     TextFormat("policy + %d-simulation search", settings.simulations),
                     static_cast<float>(ui::MARGIN), 49.0f, 13.0f, 0.4f, ui::TEXT_MUTED);

        ui::drawBoard(settings.board, settings.board, game.body(), game.food(), game.heading(),
                      !game.won());

        float panel_y = static_cast<float>(ui::HEADER_HEIGHT +
                                           ui::boardPixelHeight(settings.board) + ui::GAP + 6);
        float completion = static_cast<float>(game.score()) / static_cast<float>(game.foodsToWin());

        ui::drawLabel(fonts.label, "GRID FILLED", static_cast<float>(ui::MARGIN), panel_y,
                      ui::TEXT_MUTED);
        const char* percent = TextFormat("%.1f%%", completion * 100.0f);
        float percent_width = ui::textWidth(fonts.mono, percent, 13.0f, 0.5f);
        ui::drawText(fonts.mono, percent, window_width - ui::MARGIN - percent_width, panel_y - 2.0f,
                     13.0f, 0.5f, completion >= 1.0f ? ui::MINT : ui::TEXT_PRIMARY);

        Rectangle track = {static_cast<float>(ui::MARGIN), panel_y + 18.0f,
                           static_cast<float>(window_width - ui::MARGIN * 2), 6.0f};
        DrawRectangleRounded(track, 1.0f, 6, ui::SURFACE);
        if (completion > 0.0f)
        {
            Rectangle fill = {track.x, track.y, track.width * completion, track.height};
            DrawRectangleRounded(fill, 1.0f, 6, ui::MINT);
        }

        float tile_y = panel_y + 44.0f;
        float tile_width = static_cast<float>(window_width - ui::MARGIN * 2) / 4.0f;
        drawStatTile(fonts, "SCORE", TextFormat("%d", game.score()), ui::MARGIN + tile_width * 0.0f,
                     tile_y, tile_width, ui::TEXT_PRIMARY);
        drawStatTile(fonts, "STEPS", TextFormat("%d", game.steps()), ui::MARGIN + tile_width * 1.0f,
                     tile_y, tile_width, ui::TEXT_PRIMARY);
        // What the network thinks of the position it is in, which is the one
        // number here that a hand-written agent could not have produced.
        drawStatTile(fonts, "VALUE", TextFormat("%+.2f", last_value),
                     ui::MARGIN + tile_width * 2.0f, tile_y, tile_width,
                     last_value >= 0.0f ? ui::MINT : ui::AMBER);
        drawStatTile(fonts, "WON", TextFormat("%d/%d", wins, games_played),
                     ui::MARGIN + tile_width * 3.0f, tile_y, tile_width, ui::MINT);

        if (finished)
        {
            const char* verdict = game.won() ? "PERFECT - ENTIRE GRID FILLED" : "GAME OVER";
            ui::drawText(fonts.display, verdict, static_cast<float>(ui::MARGIN), tile_y + 46.0f,
                         20.0f, 0.5f, game.won() ? ui::MINT : ui::AMBER);
            const char* prompt = "PRESS SPACE FOR NEXT GAME";
            float prompt_width = ui::textWidth(fonts.label, prompt, 13.0f, 2.0f);
            ui::drawText(fonts.label, prompt, (window_width - prompt_width) / 2.0f, tile_y + 74.0f,
                         13.0f, 2.0f, ui::AMBER);
        }

        EndDrawing();
    }

    ui::unloadFonts(fonts);
    CloseWindow();
    return 0;
}
