#include "mcts.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

// The index of the only action that does not immediately kill, or -1 when the
// position offers a real choice (or none at all). "Survivable" here means one
// step deep: it does not promise the move is good, only that it is not fatal
// now, which is exactly the condition under which there is nothing to decide.
int onlySurvivableAction(const SnakeEnv& state) {
    int survivor = -1;
    int count = 0;
    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++) {
        SnakeEnv probe = state;
        SnakeEnv::StepResult outcome = probe.step(static_cast<SnakeEnv::Action>(action));
        if (!outcome.done || outcome.won) {
            survivor = action;
            count++;
            if (count > 1) {
                return -1;
            }
        }
    }
    return count == 1 ? survivor : -1;
}

}  // namespace

MonteCarloSearch::MonteCarloSearch(Evaluator& evaluator, const Config& config)
    : evaluator_(evaluator), config_(config), rng_(config.seed) {
    if (config.simulations < 1) {
        throw std::invalid_argument("search needs at least one simulation");
    }
}

int MonteCarloSearch::selectChild(const Tree& tree, int node_index) const {
    const Node& parent = tree.nodes[node_index];
    // The parent's visit count is the total weight behind this decision; the
    // square root of it is what lets a promising-but-unvisited action keep a
    // claim on attention as the subtree below it grows.
    const float parent_weight = std::sqrt((float)std::max(1, parent.visit_count));

    int best_action = 0;
    float best_score = -1e30f;
    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++) {
        const Node& child = tree.nodes[parent.first_child + action];

        // Value of taking this action: the reward it collects now, plus the
        // discounted value of where it lands. Unvisited children contribute
        // nothing but their prior, which is what the exploration term carries.
        float action_value = 0.0f;
        if (child.visit_count > 0) {
            float elapsed = std::pow(config_.discount, (float)std::max(1, child.edge_steps));
            action_value = child.reward + elapsed * (child.value_sum / (float)child.visit_count);
        }

        float exploration = config_.exploration * child.prior * parent_weight /
                            (1.0f + (float)child.visit_count);

        float score = action_value + exploration;
        if (score > best_score) {
            best_score = score;
            best_action = action;
        }
    }
    return best_action;
}

void MonteCarloSearch::expand(Tree& tree, int node_index, const float* priors) {
    const int first_child = (int)tree.nodes.size();
    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++) {
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

void MonteCarloSearch::backup(Tree& tree, float leaf_value) {
    float carried = leaf_value;
    for (int position = (int)tree.path.size() - 1; position >= 0; position--) {
        Node& node = tree.nodes[tree.path[position]];
        node.visit_count++;
        node.value_sum += carried;
        // Step the return back across the edge that entered this node, over the
        // number of ticks that edge actually spans.
        carried = node.reward +
                  std::pow(config_.discount, (float)std::max(1, node.edge_steps)) * carried;
    }
}

void MonteCarloSearch::addRootNoise(Tree& tree) {
    if (config_.root_noise_fraction <= 0.0f) {
        return;
    }

    std::gamma_distribution<float> gamma(config_.root_noise_alpha, 1.0f);
    float noise[SnakeEnv::ACTION_COUNT];
    float total = 0.0f;
    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++) {
        noise[action] = gamma(rng_);
        total += noise[action];
    }
    if (total <= 0.0f) {
        return;
    }

    const int first_child = tree.nodes[0].first_child;
    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++) {
        float dirichlet = noise[action] / total;
        Node& child = tree.nodes[first_child + action];
        child.prior = (1.0f - config_.root_noise_fraction) * child.prior +
                      config_.root_noise_fraction * dirichlet;
    }
}

std::vector<MonteCarloSearch::Result> MonteCarloSearch::search(
    const std::vector<const SnakeEnv*>& roots) {
    for (const SnakeEnv* root : roots) {
        if (root == nullptr) {
            throw std::invalid_argument("null root handed to search");
        }
        if (root->done()) {
            throw std::invalid_argument("search called on a finished game");
        }
    }

    const int tree_count = (int)roots.size();
    trees_.clear();
    trees_.resize(tree_count);
    for (Tree& tree : trees_) {
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
    std::vector<int> batch_owner;
    std::vector<float> priors;
    std::vector<float> values;
    std::vector<float> terminal_leaf_value(tree_count, 0.0f);

    for (int simulation = 0; simulation < config_.simulations; simulation++) {
        batch.clear();
        batch_owner.clear();

        // Selection: every tree walks down to a leaf, replaying the path from
        // its root rather than reading a stored snapshot.
        for (int index = 0; index < tree_count; index++) {
            Tree& tree = trees_[index];
            tree.path.clear();
            tree.replay.clear();
            tree.replay.push_back(*roots[index]);
            SnakeEnv& state = tree.replay[0];

            int node_index = 0;
            tree.path.push_back(node_index);

            while (tree.nodes[node_index].expanded && !tree.nodes[node_index].terminal) {
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
                while (!outcome.done) {
                    int forced = onlySurvivableAction(state);
                    if (forced < 0) {
                        break;
                    }
                    outcome = state.step(static_cast<SnakeEnv::Action>(forced));
                    edge_reward += std::pow(config_.discount, (float)edge_steps) * outcome.reward;
                    edge_steps++;
                }

                tree.nodes[child_index].reward = edge_reward;
                tree.nodes[child_index].edge_steps = edge_steps;
                tree.nodes[child_index].terminal = outcome.done;

                node_index = child_index;
                tree.path.push_back(node_index);
            }

            if (tree.nodes[node_index].terminal) {
                // Nothing follows a finished game, so no evaluation is owed and
                // the leaf contributes only the reward already on its edge.
                tree.awaiting_evaluation = false;
                terminal_leaf_value[index] = 0.0f;
            } else {
                tree.awaiting_evaluation = true;
                batch.push_back(&state);
                batch_owner.push_back(index);
            }
        }

        // One forward pass for every tree's leaf.
        if (!batch.empty()) {
            priors.assign(batch.size() * SnakeEnv::ACTION_COUNT, 0.0f);
            values.assign(batch.size(), 0.0f);
            evaluator_.evaluate(batch, priors.data(), values.data());
        }

        // Expansion and backup.
        size_t batch_position = 0;
        for (int index = 0; index < tree_count; index++) {
            Tree& tree = trees_[index];
            float leaf_value = terminal_leaf_value[index];

            if (tree.awaiting_evaluation) {
                const float* leaf_priors = priors.data() + batch_position * SnakeEnv::ACTION_COUNT;
                leaf_value = values[batch_position];
                expand(tree, tree.path.back(), leaf_priors);
                if (tree.path.size() == 1) {
                    // The root has just acquired its priors, which is the only
                    // moment noise can be applied to them.
                    addRootNoise(tree);
                }
                batch_position++;
            }

            backup(tree, leaf_value);
        }
        (void)batch_owner;
    }

    std::vector<Result> results;
    results.reserve(tree_count);
    for (int index = 0; index < tree_count; index++) {
        const Tree& tree = trees_[index];
        Result result;
        result.policy.assign(SnakeEnv::ACTION_COUNT, 0.0f);

        float total_visits = 0.0f;
        if (tree.nodes[0].first_child >= 0) {
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++) {
                total_visits += (float)tree.nodes[tree.nodes[0].first_child + action].visit_count;
            }
        }

        int best_action = 0;
        int best_visits = -1;
        for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++) {
            int visits = 0;
            if (tree.nodes[0].first_child >= 0) {
                visits = tree.nodes[tree.nodes[0].first_child + action].visit_count;
            }
            if (total_visits > 0.0f) {
                result.policy[action] = (float)visits / total_visits;
            } else {
                result.policy[action] = 1.0f / SnakeEnv::ACTION_COUNT;
            }
            if (visits > best_visits) {
                best_visits = visits;
                best_action = action;
            }
        }

        result.value = tree.nodes[0].visit_count > 0
                           ? tree.nodes[0].value_sum / (float)tree.nodes[0].visit_count
                           : 0.0f;
        result.best_action = static_cast<SnakeEnv::Action>(best_action);
        results.push_back(result);
    }

    return results;
}
