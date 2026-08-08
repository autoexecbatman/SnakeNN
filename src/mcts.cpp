#include "mcts.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <optional>
#include <stdexcept>

// The move the position leaves no choice about, if there is one.
//
// Empty means the search has something to decide - either because two or more
// moves survive, or because none does and the snake is lost whatever it picks.
// Both are handled the same way by the caller, which expands a node normally, so
// they share a return value; the distinction is spelled out here rather than
// hidden behind a -1 that means two different things.
//
// "Survives" is one step deep. It does not claim the move is good, only that it
// is not fatal this tick - which is precisely the condition under which there is
// nothing to choose.
//
// `wouldDie` is const and copies nothing. The first version of this stepped a
// clone of the environment for each move: three full copies of the body and
// occupancy vectors per level of descent, on the hottest path in the search. It
// showed up as one saturated CPU core with the GPU at 30 percent.
std::optional<SnakeEnv::Action> forcedAction(const SnakeEnv& state)
{
    // Asking a finished game which move is forced is meaningless - there are no
    // moves left. This is impossible when the search is wired correctly, so it
    // is an assertion at the site of the fault rather than a silent empty
    // return that would move the symptom somewhere else.
    assert(!state.done() && "forcedAction called on a finished episode");

    // The first surviving move found so far. Empty until one is found, which is
    // also how the "every move kills" case reports itself at the end.
    std::optional<SnakeEnv::Action> survivor;

    // Three moves to consider, always: go straight, turn left, turn right. They
    // are relative to the snake's heading, so reversing into itself is not an
    // action that exists rather than one that has to be filtered out.
    for (int index = 0; index < SnakeEnv::ACTION_COUNT; index++)
    {
        // The loop counts, so it needs an integer; the environment takes an
        // Action. This is the one place the two representations meet.
        const SnakeEnv::Action action = static_cast<SnakeEnv::Action>(index);

        // Ask whether this move ends the episode on this tick - a wall, the
        // body, or starvation. It answers without copying or mutating anything.
        if (state.wouldDie(action))
        {
            // Fatal, so it is not a candidate. Try the next one.
            continue;
        }

        // This move survives. If something already did, then at least two moves
        // survive and the position is a genuine decision - which is the opposite
        // of forced. Stop immediately: the answer cannot change, and the third
        // `wouldDie` call would be wasted on the hottest path in the search.
        if (survivor.has_value())
        {
            return std::nullopt;
        }

        // The first survivor. Remember it and keep looking, because it only
        // counts as forced if nothing else survives.
        survivor = action;
    }

    // Falling out of the loop means at most one move survived.
    //   - exactly one -> that move is forced, and `survivor` holds it
    //   - none        -> the snake is lost whatever it does, and `survivor` is
    //                    empty, which is the right answer too: there is nothing
    //                    to skip past, so the caller expands a node and lets the
    //                    value head price the position
    //
    // The caller steps whatever comes back without re-checking it, so returning
    // a fatal move would kill the snake inside the search and corrupt the
    // statistics rather than crashing. Debug builds only - this re-runs a
    // `wouldDie` on the hottest path in the system and must not cost anything in
    // the builds that train.
    assert((!survivor.has_value() || !state.wouldDie(survivor.value())) &&
           "forcedAction returned a move that kills");
    return survivor;
}

MonteCarloSearch::MonteCarloSearch(Evaluator& evaluator, const Config& config)
    : evaluator_(evaluator), config_(config), rng_(config.seed)
{
    if (config.simulations < 1)
    {
        throw std::invalid_argument("search needs at least one simulation");
    }
}

int MonteCarloSearch::selectChild(const Tree& tree, int node_index) const
{
    const Node& parent = tree.nodes[node_index];
    // The parent's visit count is the total weight behind this decision; the
    // square root of it is what lets a promising-but-unvisited action keep a
    // claim on attention as the subtree below it grows.
    const float parent_weight = std::sqrt(static_cast<float>(std::max(1, parent.visit_count)));

    int best_action = 0;
    float best_score = -1e30f;
    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
    {
        const Node& child = tree.nodes[parent.first_child + action];

        // Value of taking this action: the reward it collects now, plus the
        // discounted value of where it lands. Unvisited children contribute
        // nothing but their prior, which is what the exploration term carries.
        float action_value = 0.0f;
        if (child.visit_count > 0)
        {
            float elapsed =
                std::pow(config_.discount, static_cast<float>(std::max(1, child.edge_steps)));
            action_value =
                child.reward + elapsed * (child.value_sum / static_cast<float>(child.visit_count));
        }

        float exploration = config_.exploration * child.prior * parent_weight /
                            (1.0f + static_cast<float>(child.visit_count));

        float score = action_value + exploration;
        if (score > best_score)
        {
            best_score = score;
            best_action = action;
        }
    }
    return best_action;
}

void MonteCarloSearch::expand(Tree& tree, int node_index, const float* priors)
{
    const int first_child = static_cast<int>(tree.nodes.size());
    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
    {
        Node child;
        child.prior = priors[action];
        child.reward = 0.0f;
        child.value_sum = 0.0f;
        child.visit_count = 0;
        child.first_child = -1;
        child.edge_steps = 1;
        child.expanded = false;
        child.terminal = false;
        tree.nodes.push_back(child);
    }
    // Written after the pushes: growing the arena invalidates references, so
    // nothing may hold one across an expansion.
    tree.nodes[node_index].first_child = first_child;
    tree.nodes[node_index].expanded = true;
}

void MonteCarloSearch::backup(Tree& tree, float leaf_value)
{
    float carried = leaf_value;
    for (int position = static_cast<int>(tree.path.size()) - 1; position >= 0; position--)
    {
        Node& node = tree.nodes[tree.path[position]];
        node.visit_count++;
        node.value_sum += carried;
        // Step the return back across the edge that entered this node, over the
        // number of ticks that edge actually spans.
        carried =
            node.reward +
            std::pow(config_.discount, static_cast<float>(std::max(1, node.edge_steps))) * carried;
    }
}

void MonteCarloSearch::addRootNoise(Tree& tree)
{
    if (config_.root_noise_fraction <= 0.0f)
    {
        return;
    }

    std::gamma_distribution<float> gamma(config_.root_noise_alpha, 1.0f);
    float noise[SnakeEnv::ACTION_COUNT];
    float total = 0.0f;
    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
    {
        noise[action] = gamma(rng_);
        total += noise[action];
    }
    if (total <= 0.0f)
    {
        return;
    }

    const int first_child = tree.nodes[0].first_child;
    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
    {
        float dirichlet = noise[action] / total;
        Node& child = tree.nodes[first_child + action];
        child.prior = (1.0f - config_.root_noise_fraction) * child.prior +
                      config_.root_noise_fraction * dirichlet;
    }
}

std::vector<MonteCarloSearch::Result> MonteCarloSearch::search(
    const std::vector<const SnakeEnv*>& roots)
{
    for (const SnakeEnv* root : roots)
    {
        if (root == nullptr)
        {
            throw std::invalid_argument("null root handed to search");
        }
        if (root->done())
        {
            throw std::invalid_argument("search called on a finished game");
        }
    }

    const int tree_count = static_cast<int>(roots.size());
    trees_.clear();
    trees_.resize(tree_count);
    for (Tree& tree : trees_)
    {
        tree.nodes.clear();
        Node root;
        root.prior = 1.0f;
        root.reward = 0.0f;
        root.value_sum = 0.0f;
        root.visit_count = 0;
        root.first_child = -1;
        root.edge_steps = 0;
        root.expanded = false;
        root.terminal = false;
        tree.nodes.push_back(root);
        tree.path.clear();
        tree.replay.clear();
        tree.awaiting_evaluation = false;
    }

    std::vector<const SnakeEnv*> batch;
    std::vector<float> priors;
    std::vector<float> values;
    std::vector<float> terminal_leaf_value(tree_count, 0.0f);

    for (int simulation = 0; simulation < config_.simulations; simulation++)
    {
        batch.clear();

        // Selection: every tree walks down to a leaf, replaying the path from
        // its root rather than reading a stored snapshot.
        for (int index = 0; index < tree_count; index++)
        {
            Tree& tree = trees_[index];
            tree.path.clear();
            tree.replay.clear();
            tree.replay.push_back(*roots[index]);
            SnakeEnv& state = tree.replay[0];
            // A copy carries the root's generator, so every simulation would
            // otherwise draw the same apples and the tree would search a
            // deterministic problem - planning routes to cells it had no way of
            // knowing about. One stream per simulation makes the visit counts an
            // average over where the apple might land, which is what Du et al.
            // get by branching on every empty cell.
            state.reseed(rng_());

            int node_index = 0;
            tree.path.push_back(node_index);

            while (tree.nodes[node_index].expanded && !tree.nodes[node_index].terminal)
            {
                int action = selectChild(tree, node_index);
                int child_index = tree.nodes[node_index].first_child + action;

                SnakeEnv::StepResult outcome = state.step(static_cast<SnakeEnv::Action>(action));
                float edge_reward = outcome.reward;
                int edge_steps = 1;

                // Where only one move is survivable there is no decision to
                // make, so simulate through it rather than spending a ply of
                // tree on it. On a crowded board most of the game is like this,
                // and giving each forced move its own node buries the real
                // decisions below a search depth that never reaches them.
                while (!outcome.done)
                {
                    const std::optional<SnakeEnv::Action> forced = forcedAction(state);
                    if (!forced.has_value())
                    {
                        break;
                    }
                    outcome = state.step(forced.value());
                    edge_reward +=
                        std::pow(config_.discount, static_cast<float>(edge_steps)) * outcome.reward;
                    edge_steps++;
                }

                tree.nodes[child_index].reward = edge_reward;
                tree.nodes[child_index].edge_steps = edge_steps;
                tree.nodes[child_index].terminal = outcome.done;

                node_index = child_index;
                tree.path.push_back(node_index);
            }

            if (tree.nodes[node_index].terminal)
            {
                // Nothing follows a finished game, so no evaluation is owed and
                // the leaf contributes only the reward already on its edge.
                tree.awaiting_evaluation = false;
                terminal_leaf_value[index] = 0.0f;
            }
            else
            {
                tree.awaiting_evaluation = true;
                batch.push_back(&state);
            }
        }

        // One forward pass for every tree's leaf.
        if (!batch.empty())
        {
            priors.assign(batch.size() * SnakeEnv::ACTION_COUNT, 0.0f);
            values.assign(batch.size(), 0.0f);
            evaluator_.evaluate(batch, priors.data(), values.data());
        }

        // Expansion and backup.
        size_t batch_position = 0;
        for (int index = 0; index < tree_count; index++)
        {
            Tree& tree = trees_[index];
            float leaf_value = terminal_leaf_value[index];

            if (tree.awaiting_evaluation)
            {
                const float* leaf_priors = priors.data() + batch_position * SnakeEnv::ACTION_COUNT;
                leaf_value = values[batch_position];
                expand(tree, tree.path.back(), leaf_priors);
                if (tree.path.size() == 1)
                {
                    // The root has just acquired its priors, which is the only
                    // moment noise can be applied to them.
                    addRootNoise(tree);
                }
                batch_position++;
            }

            backup(tree, leaf_value);
        }
    }

    std::vector<Result> results;
    results.reserve(tree_count);
    for (int index = 0; index < tree_count; index++)
    {
        const Tree& tree = trees_[index];
        Result result;
        result.policy.assign(SnakeEnv::ACTION_COUNT, 0.0f);

        float total_visits = 0.0f;
        if (tree.nodes[0].first_child >= 0)
        {
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
            {
                total_visits +=
                    static_cast<float>(tree.nodes[tree.nodes[0].first_child + action].visit_count);
            }
        }

        int best_action = 0;
        int best_visits = -1;
        for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
        {
            int visits = 0;
            if (tree.nodes[0].first_child >= 0)
            {
                visits = tree.nodes[tree.nodes[0].first_child + action].visit_count;
            }
            if (total_visits > 0.0f)
            {
                result.policy[action] = static_cast<float>(visits) / total_visits;
            }
            else
            {
                result.policy[action] = 1.0f / SnakeEnv::ACTION_COUNT;
            }
            if (visits > best_visits)
            {
                best_visits = visits;
                best_action = action;
            }
        }

        result.value = tree.nodes[0].visit_count > 0
                           ? tree.nodes[0].value_sum / static_cast<float>(tree.nodes[0].visit_count)
                           : 0.0f;
        result.best_action = static_cast<SnakeEnv::Action>(best_action);
        results.push_back(result);
    }

    return results;
}
