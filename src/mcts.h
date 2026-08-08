#pragma once
#include "evaluator.h"
#include "snake_env.h"
#include <optional>
#include <random>
#include <vector>

// The move the position leaves no choice about, if there is one.
//
// Declared here rather than kept private to the search because it carries a
// subtle contract and nothing could test it while it was file-local: empty means
// the caller has a decision to make, which covers both "two or more moves
// survive" and "none does". It is a pure function of the position.
std::optional<SnakeEnv::Action> forcedAction(const SnakeEnv& state);

// Monte Carlo tree search over the real simulator, run on many games at once.
//
// Batched by construction. The measured network rate is 56k evaluations/s at
// 20x20 with a batch of 1024 and a small fraction of that one position at a
// time, so a search that evaluates a single leaf per forward pass leaves the
// GPU idle. Every tree therefore advances one simulation in lockstep and all
// their leaves are evaluated together.
//
// Tree nodes hold statistics only, never a game state. The simulator runs three
// orders of magnitude faster than the network, so replaying the path from the
// root each descent is far cheaper than storing a snapshot per node - at 20x20
// a snapshot is about 5KB, and 100 simulations across 1024 games would be half
// a gigabyte of state to keep coherent.
//
// Food placement is stochastic. A descent replays from a copy of the root, so
// each simulation samples one outcome from the true distribution and the tree
// averages over samples rather than modelling the chance node explicitly. That
// is an approximation, and it is the one place this search is not exact.
class MonteCarloSearch
{
public:
    struct Config
    {
        int simulations;
        float exploration;  // c_puct
        float discount;
        // Dirichlet noise on root priors, the standard device for keeping
        // self-play from collapsing onto one line. Set fraction to zero when
        // evaluating rather than training.
        float root_noise_fraction;
        float root_noise_alpha;
        unsigned int seed;
    };

    struct Result
    {
        // Visit distribution over actions, the search's improved policy.
        std::vector<float> policy;
        // Root value estimate after search.
        float value;
        SnakeEnv::Action best_action;
    };

    MonteCarloSearch(Evaluator& evaluator, const Config& config);

    // Not copyable and not movable, and the destructor stays compiler-generated:
    // nothing here owns a raw resource, so there is no release to write down.
    // These deletions are an interface decision rather than resource management -
    // a search has no value semantics.
    //
    // Three reasons a copy would be wrong, and the first has already happened
    // one level down. rng_ would be duplicated, so two searches would draw the
    // identical stream - the same defect as a copied SnakeEnv carrying its
    // generator, which made every simulation see the same apples. trees_ holds
    // the descent in flight, so a copy taken mid-search is a half-finished
    // search that looks complete. And evaluator_ is a reference, so both copies
    // would advance one evaluation counter while each reported its own rate.
    //
    // The reference member had already deleted both assignments implicitly,
    // leaving the type copy-constructible but not copy-assignable. Nobody chose
    // that asymmetry; stating all four removes it.
    MonteCarloSearch(const MonteCarloSearch&) = delete;
    MonteCarloSearch& operator=(const MonteCarloSearch&) = delete;
    MonteCarloSearch(MonteCarloSearch&&) = delete;
    MonteCarloSearch& operator=(MonteCarloSearch&&) = delete;

    // Runs the configured simulations on every root at once. Roots must be live;
    // a finished game has nothing to search and is the caller's to filter.
    std::vector<Result> search(const std::vector<const SnakeEnv*>& roots);

private:
    struct Node
    {
        float prior;
        // Discounted sum of every reward collected on the edge entering this
        // node. An edge usually spans one tick, but forced moves are simulated
        // through rather than given their own node, so it can span several.
        float reward;
        float value_sum;
        int visit_count;
        int first_child;  // arena index of action 0's child, or -1
        // Ticks the entering edge covers, so backup discounts by the time
        // actually elapsed. Du et al. 2022 write this as gamma^(t(s')-t(s)).
        int edge_steps;
        bool expanded;
        bool terminal;
    };

    struct Tree
    {
        std::vector<Node> nodes;
        // Scratch for the descent in flight: the path taken and the state it
        // reached, kept between the selection phase and the backup phase that
        // straddle the batched evaluation.
        std::vector<int> path;
        std::vector<SnakeEnv> replay;  // holds at most one live copy
        bool awaiting_evaluation;
    };

    Evaluator& evaluator_;
    Config config_;
    std::vector<Tree> trees_;
    std::mt19937 rng_;

    int selectChild(const Tree& tree, int node_index) const;
    void expand(Tree& tree, int node_index, const float* priors);
    void backup(Tree& tree, float leaf_value);
    void addRootNoise(Tree& tree);
};
