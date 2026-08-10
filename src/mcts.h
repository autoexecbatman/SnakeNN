#pragma once

#include <optional>
#include <random>
#include <span>
#include <vector>

#include "evaluator.h"
#include "snake_env.h"

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
    // Every field is set explicitly by the caller. The initializers exist so that
    // forgetting one is a wrong number rather than undefined behaviour - a garbage
    // exploration constant reads as a plausible run that cannot be reproduced.
    struct Config
    {
        int simulations{ 0 };
        float exploration{ 0.0f };  // c_puct
        float discount{ 0.0f };
        // Paid on every tick, so the search prices delay the same way the training
        // target does. Zero reproduces the behaviour before it existed, in which a
        // slow route to an apple scored the same as a fast one.
        float step_reward{ 0.0f };
        // How far below the best visit count an action may sit and still be taken
        // when it finishes sooner. Zero disables the tie-break entirely.
        float steps_tiebreak_margin{ 0.0f };
        // Whether to refuse a root move that seals the head away from its own
        // tail. Off reproduces the search as it was.
        //
        // This is knowledge the search cannot reach on its own: the seal only
        // proves fatal tens of moves later, past any horizon 200 simulations buy.
        // It vetoes, never suggests - the search still chooses among what is left.
        bool trap_guard{ false };
        // Whether to count the seals without acting on them. Independent of the
        // veto, because the count is the measurement of whether the network has
        // learned to avoid them and the veto is what stops that being measurable:
        // a guard that corrects the move erases the evidence it was needed.
        bool trap_report{ false };
        // Dirichlet noise on root priors, the standard device for keeping
        // self-play from collapsing onto one line. Set fraction to zero when
        // evaluating rather than training.
        float root_noise_fraction{ 0.0f };
        float root_noise_alpha{ 0.0f };
        unsigned int seed{ 0 };
    };

    struct Result
    {
        // Visit distribution over actions, the search's improved policy.
        std::vector<float> policy;
        // Root value estimate after search.
        float value{ 0.0f };
        SnakeEnv::Action best_action{ SnakeEnv::Action::STRAIGHT };
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

    // How many times the trap guard has overruled the search since construction.
    //
    // Reported rather than kept internal, because a guard that fires silently is
    // indistinguishable from one that never fires - and the number falling over a
    // run is the evidence the network is learning what the guard knows.
    long long trapGuardFires() const { return trap_guard_fires_; }

    // How many root moves the search chose that would seal the head away from its
    // tail, counted whether or not the guard was there to refuse them. This is
    // the quantity that falls as the network learns; trapGuardFires is only the
    // subset the guard managed to do something about.
    long long sealedChoices() const { return sealed_choices_; }

private:
    struct Node
    {
        // Every field carries its default here, so a node is fully formed the
        // moment it is declared. It used to be filled in field by field at two
        // separate sites - expand() for children and search() for the root - and
        // those two had already drifted apart on edge_steps. A field added later
        // would have had to be remembered in both.
        float prior{ 0.0f };
        // Discounted sum of every reward collected on the edge entering this
        // node. An edge usually spans one tick, but forced moves are simulated
        // through rather than given their own node, so it can span several.
        float reward{ 0.0f };
        float value_sum{ 0.0f };
        // Steps still needed from this node, summed over visits, as a fraction of
        // the game's budget. The mean of this is a search-derived estimate: the
        // ticks along the descent are real, and only the leaf is predicted, so it
        // is strictly better informed than the network's own steps_to_go.
        float steps_sum{ 0.0f };
        int visit_count{ 0 };
        // Arena index of action 0's child, once this node has children. Empty
        // until then, and empty is the whole answer - there is no -1 standing in
        // for it that could be added to an action and used as an index.
        //
        // This is also the only record of whether the node is expanded. A separate
        // bool used to carry that, which stored one fact twice and could disagree
        // with this field; having children and being expanded are the same thing,
        // so they are now the same field.
        std::optional<int> first_child;
        // Ticks the entering edge covers, so backup discounts by the time
        // actually elapsed. Du et al. 2022 write this as gamma^(t(s')-t(s)). One
        // is the floor for any real edge. The root has no entering edge at all,
        // and rather than marking that with a zero nothing reads, it keeps the
        // default - backup computes a return across the root and discards it.
        int edge_steps{ 1 };
        bool terminal{ false };
        // The network's steps-to-go for this node, as a fraction of the budget,
        // stored when the node was expanded. One is the pessimistic default and
        // is what an unexpanded node keeps, so a node the search never looked at
        // never wins the tie-break.
        float steps_to_go{ 1.0f };
    };

    struct Tree
    {
        std::vector<Node> nodes;
        // Scratch for the descent in flight: the path taken and the state it
        // reached, kept between the selection phase and the backup phase that
        // straddle the batched evaluation.
        std::vector<int> path;
        // Holds exactly one live copy once the first simulation has run. A vector
        // rather than a bare member because SnakeEnv has no default constructor,
        // and assigned into rather than cleared and refilled - assignment reuses
        // the body and occupancy allocations, and this is per simulation per tree.
        std::vector<SnakeEnv> replay;
        // The leaf's value, when it is already known. Engaged means the descent
        // ended on a terminal node, which is owed no evaluation and contributes
        // only what its edge already paid; empty means this tree has a leaf in the
        // batch and its value is arriving from the network.
        //
        // One field, because this used to be a bool paired with a parallel vector
        // of floats indexed by tree - two things saying one thing, kept in step by
        // hand.
        std::optional<float> known_leaf_value;
    };

    Evaluator& evaluator_;
    Config config_;
    std::vector<Tree> trees_;
    std::mt19937 rng_;

    // Scratch reused across calls, not per-call temporaries. A search runs a few
    // hundred simulations over a few hundred trees and is called once per move for
    // the length of a game, so anything allocated per call is allocated tens of
    // thousands of times per game. These keep their capacity; only their contents
    // are reset. Their size is a function of the batch, so they are cleared and
    // refilled rather than resized away.
    std::vector<const SnakeEnv*> batch_;
    std::vector<float> priors_;
    std::vector<float> values_;
    std::vector<float> steps_;

    // The PUCT score for one child, given how much attention the parent has to
    // distribute. Separated from the argmax so that the formula and the choosing
    // are readable on their own, and so the assertions on a child's statistics
    // sit next to the arithmetic that consumes them.
    float actionScore(const Node& child, float parent_weight) const;
    int selectChild(const Tree& tree, int node_index) const;
    // Priors as a span rather than a bare pointer: the length travelled with the
    // caller's intent and nowhere else, so the callee could not check it. Now it
    // can, and does.
    void expand(Tree& tree, int node_index, std::span<const float> priors);
    void backup(Tree& tree, float leaf_value, float leaf_steps);

    // The budget the steps accumulator is a fraction of, taken from the roots.
    float steps_budget_{ 1.0f };
    long long trap_guard_fires_{ 0 };
    long long sealed_choices_{ 0 };
    void addRootNoise(Tree& tree);
};
