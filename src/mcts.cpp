// Implementation of MonteCarloSearch. The interface, and how to call it, are in mcts.h.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <optional>
#include <stdexcept>

#include "mcts.h"

namespace
{

// How far priors may drift from summing to one; slack for rounding, nothing else.
constexpr float PRIOR_SUM_TOLERANCE = 1e-3f;

// How far two computations of one edge's reward may differ and still be the same
// edge. The differences the probe looks for are orders of magnitude above this.
constexpr float REWARD_TOLERANCE = 1e-4f;

// How far two computations must differ before the gap could change a move. Half an
// apple separates a longer forced-move chain from one simulation having eaten.
constexpr float MATERIAL_REWARD_DIFFERENCE = SnakeEnv::FOOD_REWARD / 2.0f;

// Re-draws allowed when every Dirichlet weight underflows to zero, which a
// concentration below one makes ordinary. Running out of attempts throws.
constexpr int MAX_NOISE_DRAW_ATTEMPTS = 8;

}  // namespace

// The move the position leaves no choice about, if there is one. Empty means the
// caller decides: either several moves survive, or none does. Survival is one tick
// deep and says nothing about whether the move is good.
std::optional<SnakeEnv::Action> forcedAction(const SnakeEnv& state)
{
    // A finished game has no moves left, and a correct search never asks.
    assert(!state.done() && "forcedAction called on a finished episode");

    // Empty at the end also reports the case where every move kills.
    std::optional<SnakeEnv::Action> survivor;

    // Relative to the heading, so reversing into itself is not an action at all.
    for (int index = 0; index < SnakeEnv::ACTION_COUNT; index++)
    {
        const SnakeEnv::Action action = static_cast<SnakeEnv::Action>(index);

        // Wall, body or starvation on this tick; answered without copying.
        if (state.wouldDie(action))
        {
            continue;
        }

        // A second survivor means a genuine decision, and no later move can undo that.
        if (survivor.has_value())
        {
            return std::nullopt;
        }

        // Forced only if nothing else survives, so keep looking.
        survivor = action;
    }

    // The caller steps this without re-checking, so a fatal move would corrupt the
    // statistics rather than crash. Debug only; this is the hottest path here.
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

float MonteCarloSearch::rawActionValue(const Node& child) const
{
    assert(child.visit_count > 0 && "an unvisited child has no value to report");

    // Averaged over traversals when asked, since one traversal is one draw.
    float edge_reward = child.reward;
    float discount_over_edge = std::pow(config_.discount, static_cast<float>(child.edge_steps));
    if (config_.average_edges && child.edge_traversals > 0)
    {
        const float traversals = static_cast<float>(child.edge_traversals);
        edge_reward = child.reward_sum / traversals;
        discount_over_edge = child.discount_sum / traversals;
    }
    const float mean_return = child.value_sum / static_cast<float>(child.visit_count);
    return edge_reward + discount_over_edge * mean_return;
}

float MonteCarloSearch::actionScore(const Node& child, float parent_weight,
                                    const ValueRange& range) const
{
    assert(child.visit_count >= 0 && "a child cannot have been visited a negative number of times");
    // Every edge covers a tick, so a zero is a corrupt arena rather than a short edge.
    assert(child.edge_steps >= 1 && "an edge that spans no ticks cannot exist");
    // A negative prior would let a child bid negatively for attention.
    assert(child.prior >= 0.0f && std::isfinite(child.prior) &&
           "Evaluator supplied a prior that is not part of a distribution");

    // What the edge pays, plus the discounted mean return from where it lands. An
    // unvisited child offers no estimate, so its whole claim is the prior below.
    float exploitation = 0.0f;
    if (child.visit_count > 0)
    {
        exploitation = rawActionValue(child);
        // Against the range this tree has compared, so the sum below adds two numbers on
        // one scale. An unestablished range returns it untouched, which is what the
        // first comparisons in a tree get.
        if (config_.normalize_values)
        {
            exploitation = range.normalize(exploitation);
        }
    }

    // The prior, scaled by the weight behind the decision, decaying with visits.
    const float exploration = config_.exploration * child.prior * parent_weight /
                              (1.0f + static_cast<float>(child.visit_count));

    const float score = exploitation + exploration;
    // A NaN compares false against everything, so the argmax would keep action 0.
    assert(std::isfinite(score) && "action score is not a finite number");
    return score;
}

int MonteCarloSearch::selectChild(const Tree& tree, int node_index) const
{
    assert(node_index >= 0 && node_index < static_cast<int>(tree.nodes.size()) &&
           "selectChild given a node index outside the arena");

    const Node& parent = tree.nodes[node_index];

    // Both guaranteed by the descent loop, which tests them before calling here.
    assert(parent.first_child.has_value() && "selectChild called on a node with no children");
    assert(!parent.terminal && "selectChild called on a terminal node");

    const int first_child = parent.first_child.value();
    assert(first_child >= 0 &&
           first_child + SnakeEnv::ACTION_COUNT <= static_cast<int>(tree.nodes.size()) &&
           "the parent's children do not all lie inside the arena");

    // Children are acquired on a descent whose backup then visits every path node,
    // so having children implies having been visited.
    assert(parent.visit_count > 0 &&
           "a node with children that was never visited - backup was skipped");

    // Its square root keeps a promising unvisited action in contention.
    const float parent_weight = std::sqrt(static_cast<float>(parent.visit_count));

    // Seeded with a real score, so no sentinel stands in for "nothing chosen yet".
    int best_action = 0;
    float best_score = actionScore(tree.nodes[first_child], parent_weight, tree.value_range);

    for (int action = 1; action < SnakeEnv::ACTION_COUNT; action++)
    {
        const float score =
            actionScore(tree.nodes[first_child + action], parent_weight, tree.value_range);
        if (score > best_score)
        {
            best_score = score;
            best_action = action;
        }
    }

    // Ties go to the lowest action index, since the comparison is strict. That is
    // deliberate and it is what makes the search reproducible on a seed.
    assert(best_action >= 0 && best_action < SnakeEnv::ACTION_COUNT &&
           "selectChild returned something that is not an action");
    return best_action;
}

void MonteCarloSearch::expand(Tree& tree, int node_index, std::span<const float> priors,
                              std::span<const float> death_risks)
{
    assert(death_risks.size() == static_cast<size_t>(SnakeEnv::ACTION_COUNT) &&
           "expand needs exactly one death risk per action");
    assert(node_index >= 0 && node_index < static_cast<int>(tree.nodes.size()) &&
           "expand given a node index outside the arena");
    // The length used to travel separately from the pointer, which meant this
    // function read three floats on the caller's word alone.
    assert(priors.size() == static_cast<size_t>(SnakeEnv::ACTION_COUNT) &&
           "expand needs exactly one prior per action");

    // A second expansion orphans the first set of children and every statistic in
    // them, and the tree goes on working with counts nothing can reach.
    assert(!tree.nodes[node_index].first_child.has_value() &&
           "expanding a node that already has children");

    // The descent stops at a terminal node and never asks for an evaluation.
    assert(!tree.nodes[node_index].terminal && "expanding a terminal node");

    // A broken distribution otherwise surfaces later as a strangely shaped policy.
    float prior_total = 0.0f;
    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
    {
        assert(std::isfinite(priors[action]) && priors[action] >= 0.0f &&
               "Evaluator supplied a prior that is not part of a distribution");
        prior_total += priors[action];
    }
    assert(std::fabs(prior_total - 1.0f) < PRIOR_SUM_TOLERANCE &&
           "Evaluator priors do not sum to one");

    // One child per action, contiguous, so the set is reachable from the first index.
    const int first_child = static_cast<int>(tree.nodes.size());
    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
    {
        Node child;
        child.prior = priors[action];
        // The network's estimate, standing until the subtree below is expanded.
        child.death_risk = death_risks[action];
        tree.nodes.push_back(child);
    }

    // Written after the pushes - growing the arena invalidates any earlier reference.
    tree.nodes[node_index].first_child = first_child;

    assert(tree.nodes[node_index].first_child.value() + SnakeEnv::ACTION_COUNT ==
               static_cast<int>(tree.nodes.size()) &&
           "the children this expansion created are not the last block in the arena");
}

void MonteCarloSearch::refreshDeathRisk(Tree& tree, int node_index)
{
    assert(node_index >= 0 && node_index < static_cast<int>(tree.nodes.size()) &&
           "refreshDeathRisk given a node index outside the arena");

    Node& node = tree.nodes[node_index];
    if (node.terminal || !node.first_child.has_value())
    {
        // Neither has actions to minimise over, so both keep what they were given.
        return;
    }

    const int first_child = node.first_child.value();
    float lowest = tree.nodes[first_child].death_risk;
    for (int action = 1; action < SnakeEnv::ACTION_COUNT; action++)
    {
        lowest = std::min(lowest, tree.nodes[first_child + action].death_risk);
    }
    node.death_risk = lowest;
}

void MonteCarloSearch::backup(Tree& tree, float leaf_value, float leaf_steps)
{
    // NaN survives addition, so one bad leaf would poison the tree permanently while
    // the policies went on looking ordinary. Caught before it can spread.
    assert(std::isfinite(leaf_value) && "backup handed a leaf value that is not a finite number");

    // An empty path is a spent simulation that no statistic records.
    assert(!tree.path.empty() && "backup on an empty path - the simulation would vanish");

    float carried = leaf_value;
    // Undiscounted, unlike the return - a step costs a step however far away.
    float carried_steps = leaf_steps;
    for (int position = static_cast<int>(tree.path.size()) - 1; position >= 0; position--)
    {
        assert(tree.path[position] >= 0 &&
               tree.path[position] < static_cast<int>(tree.nodes.size()) &&
               "a path entry points outside the arena");

        Node& node = tree.nodes[tree.path[position]];
        node.visit_count++;
        node.value_sum += carried;
        node.steps_sum += carried_steps;

        // Every node this simulation touched, not just the siblings of one decision.
        // A range built from one node's children is empty until two of them have been
        // visited, which on a sharp policy never happens - so it would normalise
        // nothing in exactly the position it exists for. Skips the root, which is
        // nobody's child and so is never a term in a comparison.
        if (config_.normalize_values && position > 0)
        {
            tree.value_range.observe(rawActionValue(node));
        }

        // Refreshed, not accumulated: unavoidability is a property of the position.
        refreshDeathRisk(tree, tree.path[position]);

        // Children start at one tick and the descent only raises them.
        assert(node.edge_steps >= 1 && "backing up across an edge that spans no ticks");

        // Back across the entering edge, over the ticks it spans - the paper's
        // gamma^(t(s\') - t(s)), above one where forced moves were simulated through.
        carried =
            node.reward + std::pow(config_.discount, static_cast<float>(node.edge_steps)) * carried;
        // Added undiscounted, over the whole game's budget, so it stays a fraction.
        carried_steps += static_cast<float>(node.edge_steps) / steps_budget_;

        assert(std::isfinite(carried) && "the carried return stopped being a finite number");
    }
}

void MonteCarloSearch::addRootNoise(Tree& tree)
{
    if (config_.root_noise_fraction <= 0.0f)
    {
        return;
    }

    // Above one the weight on the network's prior goes negative, and expand's check
    // has already run by then.
    assert(config_.root_noise_fraction <= 1.0f &&
           "root noise fraction above one inverts the prior");
    // std::gamma_distribution is undefined for a non-positive shape.
    assert(config_.root_noise_alpha > 0.0f && "Dirichlet concentration must be positive");

    // Called straight after the root is expanded, so the priors exist to mix into.
    assert(tree.nodes[0].first_child.has_value() &&
           "root noise applied before the root had children");

    // Draws are non-negative, so a non-positive total means all three underflowed,
    // which leaves the distribution undefined rather than skewed.
    std::gamma_distribution<float> gamma(config_.root_noise_alpha, 1.0f);
    float noise[SnakeEnv::ACTION_COUNT];
    float total = 0.0f;
    for (int attempt = 0; attempt < MAX_NOISE_DRAW_ATTEMPTS && total <= 0.0f; attempt++)
    {
        total = 0.0f;
        for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
        {
            noise[action] = gamma(rng_);
            assert(noise[action] >= 0.0f && std::isfinite(noise[action]) &&
                   "a gamma draw must be a finite non-negative number");
            total += noise[action];
        }
    }
    if (total <= 0.0f)
    {
        throw std::runtime_error(
            "root noise: every Dirichlet draw underflowed to zero repeatedly - the concentration "
            "is too small for single precision");
    }

    const int first_child = tree.nodes[0].first_child.value();
    float mixed_total = 0.0f;
    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
    {
        const float dirichlet = noise[action] / total;
        Node& child = tree.nodes[first_child + action];
        child.prior = (1.0f - config_.root_noise_fraction) * child.prior +
                      config_.root_noise_fraction * dirichlet;
        assert(child.prior >= 0.0f && std::isfinite(child.prior) &&
               "mixing noise produced a prior that is not part of a distribution");
        mixed_total += child.prior;
    }

    // The only check that runs after mixing; expand's identical one runs before.
    assert(std::fabs(mixed_total - 1.0f) < PRIOR_SUM_TOLERANCE &&
           "root priors stopped summing to one after noise was mixed in");
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

    assert(config_.simulations >= 1 && "the constructor rejects anything less");

    const int tree_count = static_cast<int>(roots.size());

    // Grows but never shrinks: the capacity of every arena, path and replay slot is
    // what is being kept, and only the contents are reset below.
    if (static_cast<int>(trees_.size()) < tree_count)
    {
        trees_.resize(tree_count);
    }
    assert(static_cast<int>(trees_.size()) >= tree_count &&
           "fewer trees than games - some root would go unsearched");

    // Read off the roots, which is the only place the limit reaches the search. At
    // least one so the division holds, and guarded because no roots is legal.
    if (!roots.empty())
    {
        steps_budget_ = static_cast<float>(std::max(1, roots.front()->stepLimit()));
    }

    // By index, not by range: the tail belongs to a larger earlier batch and holds
    // the capacity this is keeping.
    for (int index = 0; index < tree_count; index++)
    {
        Tree& tree = trees_[index];
        tree.nodes.clear();
        // Certain rather than predicted, so its prior is one; its absent edge keeps
        // the default, since backup discards what it computes across the root.
        Node root;
        root.prior = 1.0f;
        tree.nodes.push_back(root);
        // Overwritten every simulation before anything reads them; they stay because
        // this loop is the one place that says what a fresh tree is.
        tree.path.clear();
        tree.known_leaf_value.reset();
    }

    for (int simulation = 0; simulation < config_.simulations; simulation++)
    {
        batch_.clear();

        // Every tree walks to a leaf, replaying from its root rather than a snapshot.
        for (int index = 0; index < tree_count; index++)
        {
            Tree& tree = trees_[index];
            tree.path.clear();

            // Assigned into rather than refilled, so both buffers are reused - this
            // runs once per simulation per tree.
            if (tree.replay.empty())
            {
                tree.replay.push_back(*roots[index]);
            }
            else
            {
                tree.replay[0] = *roots[index];
            }
            assert(tree.replay.size() == 1 && "the replay slot holds exactly one state");
            SnakeEnv& state = tree.replay[0];
            // Without this every simulation draws the same apples and plans routes to
            // cells it cannot know. One stream each averages over where food lands.
            state.reseed(rng_());

            int node_index = 0;
            tree.path.push_back(node_index);

            while (tree.nodes[node_index].first_child.has_value() &&
                   !tree.nodes[node_index].terminal)
            {
                int action = selectChild(tree, node_index);
                int child_index = tree.nodes[node_index].first_child.value() + action;

                SnakeEnv::StepResult outcome = state.step(static_cast<SnakeEnv::Action>(action));
                // Paid on every tick the edge covers, as the training target does.
                float edge_reward = outcome.reward + config_.step_reward;
                int edge_steps = 1;

                // One survivable move is no decision, so simulate through it - on a
                // crowded board, nodes for these would bury the real decisions.
                while (!outcome.done)
                {
                    const std::optional<SnakeEnv::Action> forced = forcedAction(state);
                    if (!forced.has_value())
                    {
                        break;
                    }
                    outcome = state.step(forced.value());
                    edge_reward += std::pow(config_.discount, static_cast<float>(edge_steps)) *
                                   (outcome.reward + config_.step_reward);
                    edge_steps++;
                }

                if (config_.alias_report)
                {
                    Node& child = tree.nodes[child_index];
                    if (child.edge_recorded)
                    {
                        revisited_edges_++;
                        if (!child.revisit_counted)
                        {
                            revisited_nodes_++;
                            child.revisit_counted = true;
                        }
                        // The same actions walked into different games.
                        const float reward_gap = std::abs(child.first_reward - edge_reward);
                        const bool differs =
                            reward_gap > REWARD_TOLERANCE || child.first_edge_steps != edge_steps;
                        if (differs)
                        {
                            aliased_edges_++;
                            if (reward_gap > MATERIAL_REWARD_DIFFERENCE)
                            {
                                materially_aliased_edges_++;
                            }
                            if (!child.alias_counted)
                            {
                                aliased_nodes_++;
                                child.alias_counted = true;
                            }
                        }
                    }
                    else
                    {
                        child.first_reward = edge_reward;
                        child.first_edge_steps = edge_steps;
                        child.edge_recorded = true;
                    }
                }

                if (config_.average_edges)
                {
                    Node& child = tree.nodes[child_index];
                    child.reward_sum += edge_reward;
                    child.discount_sum +=
                        std::pow(config_.discount, static_cast<float>(edge_steps));
                    child.edge_traversals++;
                }

                tree.nodes[child_index].reward = edge_reward;
                tree.nodes[child_index].edge_steps = edge_steps;
                tree.nodes[child_index].terminal = outcome.done;
                if (outcome.done)
                {
                    // Not won counts as death; an exhausted budget lands here too, rarely.
                    tree.nodes[child_index].death_risk = state.won() ? 0.0f : 1.0f;
                }

                node_index = child_index;
                tree.path.push_back(node_index);
            }

            if (tree.nodes[node_index].terminal)
            {
                // No evaluation is owed; the edge reward is the whole contribution.
                tree.known_leaf_value = 0.0f;
            }
            else
            {
                tree.known_leaf_value.reset();
                batch_.push_back(&state);
            }
        }

        assert(batch_.size() <= static_cast<size_t>(tree_count) &&
               "more leaves queued than there are trees");

        // One forward pass for every leaf; assign keeps the largest batch's capacity.
        if (!batch_.empty())
        {
            priors_.assign(batch_.size() * SnakeEnv::ACTION_COUNT, 0.0f);
            values_.assign(batch_.size(), 0.0f);
            steps_.assign(batch_.size(), 1.0f);
            death_risks_.assign(batch_.size() * SnakeEnv::ACTION_COUNT, 0.0f);
            evaluator_.evaluate(batch_, priors_.data(), values_.data(), steps_.data(),
                                death_risks_.data());
        }

        // Expansion and backup.
        size_t batch_position = 0;
        for (int index = 0; index < tree_count; index++)
        {
            Tree& tree = trees_[index];

            // The optional alone decides between a known terminal and a batched leaf.
            float leaf_value = 0.0f;
            // A filled board needs none; a death needs more than the budget, so it
            // takes the whole budget, as a lost game does in self-play.
            float leaf_steps = 0.0f;
            if (tree.known_leaf_value.has_value())
            {
                leaf_value = tree.known_leaf_value.value();
                leaf_steps = tree.replay.front().won() ? 0.0f : 1.0f;
            }
            else
            {
                assert(batch_position < values_.size() &&
                       "a tree is owed an evaluation the batch does not contain");
                leaf_value = values_[batch_position];
                leaf_steps = steps_[batch_position];
                tree.nodes[tree.path.back()].steps_to_go = steps_[batch_position];

                const std::span<const float> leaf_priors = std::span<const float>(priors_).subspan(
                    batch_position * SnakeEnv::ACTION_COUNT, SnakeEnv::ACTION_COUNT);
                const std::span<const float> leaf_risks =
                    std::span<const float>(death_risks_)
                        .subspan(batch_position * SnakeEnv::ACTION_COUNT, SnakeEnv::ACTION_COUNT);
                expand(tree, tree.path.back(), leaf_priors, leaf_risks);

                if (tree.path.size() == 1)
                {
                    // The only moment the root's priors exist and are still untouched.
                    addRootNoise(tree);
                }
                batch_position++;
            }

            backup(tree, leaf_value, leaf_steps);
        }

        // A mismatch means values were read against the wrong tree, which would train
        // the network on positions it never saw and leave no other trace.
        assert(batch_position == batch_.size() &&
               "the evaluated batch was not consumed exactly once");
    }

    std::vector<Result> results;
    results.reserve(tree_count);
    for (int index = 0; index < tree_count; index++)
    {
        const Tree& tree = trees_[index];
        Result result;
        result.policy.assign(SnakeEnv::ACTION_COUNT, 0.0f);

        // A childless root is a legal case - no search was asked for - and the only
        // reason these visits can all be zero, in which case the policy is uniform.
        int visits[SnakeEnv::ACTION_COUNT] = { 0, 0, 0 };
        int total_visits = 0;
        result.death_risk.assign(SnakeEnv::ACTION_COUNT, 0.0f);
        if (tree.nodes[0].first_child.has_value())
        {
            const int first_child = tree.nodes[0].first_child.value();
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
            {
                visits[action] = tree.nodes[first_child + action].visit_count;
                result.death_risk[action] = tree.nodes[first_child + action].death_risk;
            }

            // Applied to the visits before anything reads them, so a refusal is absent
            // from policy and argmax alike. Refuses only when something survives.
            if (config_.death_cap)
            {
                bool anything_survives = false;
                for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
                {
                    if (result.death_risk[action] <= config_.death_cap_threshold)
                    {
                        anything_survives = true;
                    }
                }
                if (anything_survives)
                {
                    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
                    {
                        if (result.death_risk[action] > config_.death_cap_threshold)
                        {
                            visits[action] = 0;
                            death_cap_fires_++;
                        }
                    }
                }
            }

            result.all_actions_visited = true;
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
            {
                total_visits += visits[action];
                if (visits[action] == 0)
                {
                    result.all_actions_visited = false;
                }
            }
        }

        // Seeded with a real count; ties go to the lowest index, so a seed reproduces.
        int best_action = 0;
        for (int action = 1; action < SnakeEnv::ACTION_COUNT; action++)
        {
            if (visits[action] > visits[best_action])
            {
                best_action = action;
            }
        }

        // Among near-equals, prefer the one finishing sooner; the margin keeps this
        // from overruling a clear preference. The policy target goes out unchanged.
        if (config_.steps_tiebreak_margin > 0.0f && tree.nodes[0].first_child.has_value())
        {
            const int first_child = tree.nodes[0].first_child.value();
            const float floor_visits =
                static_cast<float>(visits[best_action]) * (1.0f - config_.steps_tiebreak_margin);
            const auto meanSteps = [&tree, first_child](int action)
            {
                const Node& child = tree.nodes[first_child + action];
                // Search-derived where it went, the network's guess where it did not.
                return child.visit_count > 0
                           ? child.steps_sum / static_cast<float>(child.visit_count)
                           : child.steps_to_go;
            };
            float best_steps = meanSteps(best_action);
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
            {
                if (action == best_action || visits[action] == 0)
                {
                    continue;
                }
                if (static_cast<float>(visits[action]) >= floor_visits &&
                    meanSteps(action) < best_steps)
                {
                    best_steps = meanSteps(action);
                    best_action = action;
                }
            }
        }

        // Last, so it overrules everything above: a seal kills far past what the search
        // sees. Tail reachability, not a cell count, which would veto every endgame.
        if (config_.trap_guard || config_.trap_report)
        {
            const SnakeEnv& root_state = *roots[index];
            const bool sealed =
                !root_state.tailReachable(static_cast<SnakeEnv::Action>(best_action));
            if (sealed)
            {
                sealed_choices_++;
            }
            if (sealed && config_.trap_guard)
            {
                // The guard says which moves are available, never which is good.
                int rescue = best_action;
                int rescue_visits = -1;
                for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
                {
                    if (root_state.tailReachable(static_cast<SnakeEnv::Action>(action)) &&
                        visits[action] > rescue_visits)
                    {
                        rescue_visits = visits[action];
                        rescue = action;
                    }
                }
                // All sealed means nothing to veto, so the judged move stands.
                if (rescue != best_action)
                {
                    best_action = rescue;
                    trap_guard_fires_++;
                }
            }
        }

        for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
        {
            result.policy[action] =
                (total_visits > 0)
                    ? static_cast<float>(visits[action]) / static_cast<float>(total_visits)
                    : 1.0f / static_cast<float>(SnakeEnv::ACTION_COUNT);
        }

        result.value = tree.nodes[0].visit_count > 0
                           ? tree.nodes[0].value_sum / static_cast<float>(tree.nodes[0].visit_count)
                           : 0.0f;
        result.best_action = static_cast<SnakeEnv::Action>(best_action);

        // Checked once here rather than at the trainer, evaluator and demo alike.
        float policy_total = 0.0f;
        for (float weight : result.policy)
        {
            assert(std::isfinite(weight) && weight >= 0.0f && "a visit weight is not a proportion");
            policy_total += weight;
        }
        assert(std::fabs(policy_total - 1.0f) < PRIOR_SUM_TOLERANCE &&
               "the visit policy does not sum to one");
        assert(std::isfinite(result.value) && "the root value is not a finite number");
        assert(static_cast<int>(result.best_action) >= 0 &&
               static_cast<int>(result.best_action) < SnakeEnv::ACTION_COUNT &&
               "best_action is not an action");

        results.push_back(result);
    }

    // Callers index results against roots, so a short return mispairs them.
    assert(results.size() == roots.size() && "search returned a different number of results");
    return results;
}
