#include "mcts.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace
{

// How far the priors an Evaluator hands over may drift from summing to one before
// the distribution is called broken. Float accumulation over three terms, so the
// slack is for rounding and nothing else.
constexpr float PRIOR_SUM_TOLERANCE = 1e-3f;

// How many times to re-draw the Dirichlet weights if every one of them comes back
// zero. With a concentration below one the gamma density diverges at the origin,
// so very small draws are ordinary and an underflow to exactly zero is not a
// theoretical branch. A degenerate sample is a sampling accident and re-drawing is
// the honest response; running out of attempts is not, and throws.
constexpr int MAX_NOISE_DRAW_ATTEMPTS = 8;

}  // namespace

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

float MonteCarloSearch::actionScore(const Node& child, float parent_weight) const
{
    assert(child.visit_count >= 0 && "a child cannot have been visited a negative number of times");
    // Every edge covers at least one tick. expand() creates children at one and
    // the descent only ever raises it, so a zero here means the arena was
    // corrupted rather than that an edge is short. This used to be clamped with
    // max(1, ...), which turned a corrupt tree into a plausible score.
    assert(child.edge_steps >= 1 && "an edge that spans no ticks cannot exist");
    // Evaluator promises priors that form a distribution over the actions, so a
    // negative one is a broken evaluator and would let a child bid negatively for
    // attention - the opposite of what a prior is for.
    assert(child.prior >= 0.0f && std::isfinite(child.prior) &&
           "Evaluator supplied a prior that is not part of a distribution");

    // Exploitation: what the edge pays on the way, plus the discounted mean
    // return from where it lands, over the ticks the edge actually spans.
    // An unvisited child has no estimate to offer, so its entire claim on
    // attention is the prior, carried by the exploration term below.
    float exploitation = 0.0f;
    if (child.visit_count > 0)
    {
        const float discount_over_edge =
            std::pow(config_.discount, static_cast<float>(child.edge_steps));
        const float mean_return = child.value_sum / static_cast<float>(child.visit_count);
        exploitation = child.reward + discount_over_edge * mean_return;
    }

    // Exploration: the prior, scaled by how much weight stands behind this
    // decision and decaying as this particular child gets visited.
    const float exploration = config_.exploration * child.prior * parent_weight /
                              (1.0f + static_cast<float>(child.visit_count));

    const float score = exploitation + exploration;
    // A NaN would compare false against everything, so the argmax below would
    // silently keep whichever action it started with and the search would look
    // like a policy that always goes straight. Caught here, at the arithmetic
    // that produced it.
    assert(std::isfinite(score) && "action score is not a finite number");
    return score;
}

int MonteCarloSearch::selectChild(const Tree& tree, int node_index) const
{
    assert(node_index >= 0 && node_index < static_cast<int>(tree.nodes.size()) &&
           "selectChild given a node index outside the arena");

    const Node& parent = tree.nodes[node_index];

    // Both are guaranteed by the descent loop, which tests them before it calls
    // here. A node without children has nothing to choose between, and a terminal
    // one has nothing to choose.
    assert(parent.first_child.has_value() && "selectChild called on a node with no children");
    assert(!parent.terminal && "selectChild called on a terminal node");

    const int first_child = parent.first_child.value();
    assert(first_child >= 0 &&
           first_child + SnakeEnv::ACTION_COUNT <= static_cast<int>(tree.nodes.size()) &&
           "the parent's children do not all lie inside the arena");

    // A node with children has always been visited at least once. It acquires
    // them only at the end of a descent that had it on the path, and that same
    // simulation's backup increments every node on the path - so having children
    // implies having been visited, and the two cannot come apart without the
    // descent loop changing.
    //
    // This replaced a max(1, visit_count) clamp. The clamp read as a guard
    // against a first descent reaching an unvisited root, which cannot happen for
    // the reason above; a probe that asserted the condition across the whole
    // search suite never fired. Left as a clamp it would have silently supplied a
    // weight of one for a tree that had lost its statistics.
    assert(parent.visit_count > 0 &&
           "a node with children that was never visited - backup was skipped");

    // The visit count is the weight behind this decision, and its square root is
    // what keeps a promising-but-unvisited action in contention as the subtree
    // below it grows.
    const float parent_weight = std::sqrt(static_cast<float>(parent.visit_count));

    // Argmax over the three actions. Seeded with the first action's real score
    // rather than with a very negative number standing in for "nothing chosen
    // yet" - that sentinel was indistinguishable from a legitimately terrible
    // score, and it made action 0 the answer whenever every score was NaN.
    int best_action = 0;
    float best_score = actionScore(tree.nodes[first_child], parent_weight);

    for (int action = 1; action < SnakeEnv::ACTION_COUNT; action++)
    {
        const float score = actionScore(tree.nodes[first_child + action], parent_weight);
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

void MonteCarloSearch::expand(Tree& tree, int node_index, std::span<const float> priors)
{
    assert(node_index >= 0 && node_index < static_cast<int>(tree.nodes.size()) &&
           "expand given a node index outside the arena");
    // The length used to travel separately from the pointer, which meant this
    // function read three floats on the caller's word alone.
    assert(priors.size() == static_cast<size_t>(SnakeEnv::ACTION_COUNT) &&
           "expand needs exactly one prior per action");

    // Expanding twice would push a second set of children and then point the
    // parent at them, orphaning the first set together with every visit and
    // return already accumulated in it. The tree would keep working and the
    // statistics would quietly be for a subtree nothing can reach any more.
    assert(!tree.nodes[node_index].first_child.has_value() &&
           "expanding a node that already has children");

    // Nothing follows a finished game. The descent stops at a terminal node and
    // never asks for an evaluation, so reaching here means the caller lost track
    // of which leaves were owed one.
    assert(!tree.nodes[node_index].terminal && "expanding a terminal node");

    // Evaluator promises a distribution over the actions. A prior that is
    // negative, infinite or does not sum to one is a broken evaluator, and every
    // symptom of it appears later as a strangely shaped policy rather than here.
    float prior_total = 0.0f;
    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
    {
        assert(std::isfinite(priors[action]) && priors[action] >= 0.0f &&
               "Evaluator supplied a prior that is not part of a distribution");
        prior_total += priors[action];
    }
    assert(std::fabs(prior_total - 1.0f) < PRIOR_SUM_TOLERANCE &&
           "Evaluator priors do not sum to one");

    // One child per action, contiguous, so the whole set is reachable from the
    // index of the first. Every field but the prior comes from the struct's own
    // defaults - a child differs from a fresh node in exactly one respect.
    const int first_child = static_cast<int>(tree.nodes.size());
    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
    {
        Node child;
        child.prior = priors[action];
        tree.nodes.push_back(child);
    }

    // Written after the pushes, and read through the arena rather than through a
    // reference taken earlier: growing the vector invalidates references, so
    // nothing may hold one across an expansion.
    tree.nodes[node_index].first_child = first_child;

    assert(tree.nodes[node_index].first_child.value() + SnakeEnv::ACTION_COUNT ==
               static_cast<int>(tree.nodes.size()) &&
           "the children this expansion created are not the last block in the arena");
}

void MonteCarloSearch::backup(Tree& tree, float leaf_value)
{
    // A NaN here would be added into value_sum at every node on the path and then
    // into every ancestor on every later simulation, and NaN survives addition -
    // so one bad evaluation permanently poisons the tree while the search goes on
    // returning policies that look ordinary. Caught at the entrance, because after
    // this point there is no telling which simulation introduced it.
    assert(std::isfinite(leaf_value) && "backup handed a leaf value that is not a finite number");

    // Every descent pushes the root before it does anything else, so a path is
    // never empty. An empty one would make this a silent no-op: the simulation
    // would be spent and no statistic anywhere would record it.
    assert(!tree.path.empty() && "backup on an empty path - the simulation would vanish");

    float carried = leaf_value;
    for (int position = static_cast<int>(tree.path.size()) - 1; position >= 0; position--)
    {
        assert(tree.path[position] >= 0 &&
               tree.path[position] < static_cast<int>(tree.nodes.size()) &&
               "a path entry points outside the arena");

        Node& node = tree.nodes[tree.path[position]];
        node.visit_count++;
        node.value_sum += carried;

        // Every node on a path has a real edge count: children are created at one
        // and the descent only raises them, and the root now keeps the same
        // default rather than marking its absent edge with a zero. This was a
        // max(1, edge_steps) clamp, which produced the identical number for the
        // root and hid a zero anywhere else.
        assert(node.edge_steps >= 1 && "backing up across an edge that spans no ticks");

        // Step the return back across the edge that entered this node, over the
        // number of ticks that edge actually spans. Du et al. 2022 write the
        // factor as gamma^(t(s') - t(s)); edge_steps is that exponent, and it is
        // greater than one exactly where forced moves were simulated through
        // rather than given nodes of their own.
        carried = node.reward +
                  std::pow(config_.discount, static_cast<float>(node.edge_steps)) * carried;

        assert(std::isfinite(carried) && "the carried return stopped being a finite number");
    }
}

void MonteCarloSearch::addRootNoise(Tree& tree)
{
    if (config_.root_noise_fraction <= 0.0f)
    {
        return;
    }

    // A fraction above one would make the weight on the network's own prior
    // negative, so a child could bid negatively for attention and expand's
    // sum-to-one check cannot see it - the noise is mixed in after that check has
    // already run.
    assert(config_.root_noise_fraction <= 1.0f && "root noise fraction above one inverts the prior");
    // std::gamma_distribution is undefined for a non-positive shape.
    assert(config_.root_noise_alpha > 0.0f && "Dirichlet concentration must be positive");

    // Only ever called immediately after the root is expanded, so its children
    // exist. An empty optional here would mean noise was being mixed into priors
    // that had not been written yet.
    assert(tree.nodes[0].first_child.has_value() &&
           "root noise applied before the root had children");

    // Draw the Dirichlet weights. Every draw is non-negative, so the only way the
    // total can fail to be positive is all three underflowing to zero - which
    // leaves the distribution undefined rather than merely skewed.
    //
    // This used to return silently when that happened: noise requested, no noise
    // applied, nothing said. That is the failure mode self-play cannot report,
    // because a run with no exploration looks exactly like a run with bad luck.
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

    // The convex combination of two distributions is a distribution, so this has
    // to still hold - and this is the only place that can check it. expand asserts
    // the same property, but it runs before the noise is applied, so the one
    // operation able to break the invariant sat outside the one check for it.
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

    // Grow only when the caller brings more games than last time, and never
    // shrink. This used to clear and resize unconditionally, which destroyed every
    // Tree and with it the node arena, the path and the replay slot - so a search
    // called once per move for a whole game reallocated all of them on every move.
    // Only the contents are reset below; the capacity is what is being kept.
    if (static_cast<int>(trees_.size()) < tree_count)
    {
        trees_.resize(tree_count);
    }
    assert(static_cast<int>(trees_.size()) >= tree_count &&
           "fewer trees than games - some root would go unsearched");

    // Reset by index rather than by range, because trees_ may be longer than this
    // call needs and the tail belongs to a previous, larger batch. Nothing past
    // tree_count is read, but nothing past it may be reset either - resetting it
    // would throw away exactly the capacity this is keeping.
    for (int index = 0; index < tree_count; index++)
    {
        Tree& tree = trees_[index];
        tree.nodes.clear();
        // The root differs from a default node in exactly one respect: it is
        // certain rather than predicted, so its prior is one. No edge enters it,
        // and that is left as the default rather than marked with a zero, because
        // backup computes a return across the root and then discards it - the
        // field is never read for node zero.
        Node root;
        root.prior = 1.0f;
        tree.nodes.push_back(root);
        // Both of these are overwritten for every tree on every simulation before
        // anything reads them, so neither is load-bearing - mutation testing
        // confirms removing either changes no result. They stay because this loop
        // is the one place that says what a fresh tree is, and trees now outlive
        // the call that made them.
        tree.path.clear();
        tree.known_leaf_value.reset();
    }

    for (int simulation = 0; simulation < config_.simulations; simulation++)
    {
        batch_.clear();

        // Selection: every tree walks down to a leaf, replaying the path from
        // its root rather than reading a stored snapshot.
        for (int index = 0; index < tree_count; index++)
        {
            Tree& tree = trees_[index];
            tree.path.clear();

            // Assign into the existing slot rather than clearing and refilling it.
            // Clearing destroyed the SnakeEnv and freed its body and occupancy
            // vectors, and the push_back allocated two more - twice per simulation
            // per tree, which at a few hundred of each is tens of thousands of
            // allocation pairs per move. Copy assignment reuses both buffers.
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
            // A copy carries the root's generator, so every simulation would
            // otherwise draw the same apples and the tree would search a
            // deterministic problem - planning routes to cells it had no way of
            // knowing about. One stream per simulation makes the visit counts an
            // average over where the apple might land, which is what Du et al.
            // get by branching on every empty cell.
            state.reseed(rng_());

            int node_index = 0;
            tree.path.push_back(node_index);

            while (tree.nodes[node_index].first_child.has_value() &&
                   !tree.nodes[node_index].terminal)
            {
                int action = selectChild(tree, node_index);
                int child_index = tree.nodes[node_index].first_child.value() + action;

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

        // One forward pass for every tree's leaf. assign rather than resize, so the
        // buffers keep whatever capacity the largest batch so far needed and the
        // contents are always freshly written.
        if (!batch_.empty())
        {
            priors_.assign(batch_.size() * SnakeEnv::ACTION_COUNT, 0.0f);
            values_.assign(batch_.size(), 0.0f);
            evaluator_.evaluate(batch_, priors_.data(), values_.data());
        }

        // Expansion and backup.
        size_t batch_position = 0;
        for (int index = 0; index < tree_count; index++)
        {
            Tree& tree = trees_[index];

            // Either the value is already known, because the descent finished on a
            // terminal node, or this tree has a leaf in the batch and the value is
            // the network's. The optional is the only thing that decides which, so
            // the two cases cannot both be taken or both be missed.
            float leaf_value = 0.0f;
            if (tree.known_leaf_value.has_value())
            {
                leaf_value = tree.known_leaf_value.value();
            }
            else
            {
                assert(batch_position < values_.size() &&
                       "a tree is owed an evaluation the batch does not contain");
                leaf_value = values_[batch_position];

                const std::span<const float> leaf_priors =
                    std::span<const float>(priors_).subspan(
                        batch_position * SnakeEnv::ACTION_COUNT, SnakeEnv::ACTION_COUNT);
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

        // The batch is consumed in tree order, so every leaf queued during
        // selection must have been claimed by exactly one tree here. A mismatch
        // means priors and values were read against the wrong tree - which would
        // train the network on positions it never saw and leave no other trace.
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

        // Read the root's children once. The optional is tested here and nowhere
        // else in this loop: previously every read repeated the same check against
        // -1, and the two copies had to agree by inspection.
        //
        // Zero simulations leaves the root childless and there is nothing to
        // report, so the policy falls back to uniform. That is a real case rather
        // than a defensive one - a caller may legitimately ask for no search - and
        // it is the only reason the visits array can be all zeros.
        int visits[SnakeEnv::ACTION_COUNT] = {0, 0, 0};
        int total_visits = 0;
        if (tree.nodes[0].first_child.has_value())
        {
            const int first_child = tree.nodes[0].first_child.value();
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
            {
                visits[action] = tree.nodes[first_child + action].visit_count;
                total_visits += visits[action];
            }
        }

        // Argmax seeded with the first action's real count, so no negative stands
        // in for "nothing seen yet". Ties go to the lowest index, which is what
        // makes the choice reproducible on a seed.
        int best_action = 0;
        for (int action = 1; action < SnakeEnv::ACTION_COUNT; action++)
        {
            if (visits[action] > visits[best_action])
            {
                best_action = action;
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

        // What every caller assumes about a Result, checked once here rather than
        // separately at the trainer, the evaluator and the visual demo. The policy
        // is a training target, so a total that has drifted trains the network on a
        // distribution that is not one.
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

    // One result per root, in the caller's order. Every caller indexes the two
    // together, so a short return would silently pair a policy with the wrong game.
    assert(results.size() == roots.size() && "search returned a different number of results");
    return results;
}
