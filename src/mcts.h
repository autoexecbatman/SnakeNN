#pragma once

// Monte Carlo tree search over the real simulator. The network supplies a prior and
// a value; this does the lookahead and returns an improved policy.
//
// Holds forcedAction, a free function for the no-choice case, and MonteCarloSearch.
// A call searches many games at once - every tree advances one simulation in
// lockstep so all their leaves reach the network in one forward pass, which is the
// difference between a saturated GPU and an idle one.
//
// Usage, from az_trainer.cpp and selfplay.cpp:
//
//     MonteCarloSearch::Config search_config;      // every field is set explicitly
//     search_config.simulations = 200;             // lookaheads per move
//     search_config.exploration = 0.5f;            // c_puct
//     search_config.discount = 0.98f;
//     search_config.step_reward = -0.02f;          // paid every tick
//     search_config.steps_tiebreak_margin = 0.05f; // 0 disables the tie-break
//     search_config.trap_guard = false;            // veto moves that seal the tail
//     search_config.trap_report = true;            // count them without vetoing
//     search_config.alias_report = false;          // az_evaluate sets this one true
//     search_config.average_edges = false;         // read an edge's mean, not its last write
//     search_config.death_cap = false;             // refuse near-certain death
//     search_config.death_cap_threshold = 0.5f;
//     search_config.root_noise_fraction = 0.25f;   // zero when evaluating
//     search_config.root_noise_alpha = 0.3f;
//     search_config.seed = 12345;
//
//     MonteCarloSearch search(evaluator, search_config);   // evaluator must outlive it
//
//     std::vector<const SnakeEnv*> roots{ &game_a, &game_b };  // live games only
//     std::vector<MonteCarloSearch::Result> results = search.search(roots);
//     results[0].policy;         // visit shares over the 3 actions, the training target
//     results[0].best_action;    // the move to play, after tie-break and any veto
//     results[0].value;          // the root's mean return after search
//     results[0].death_risk;     // backed-up risk per action, in [0, 1]
//
//     long long sealed = search.sealedChoices();   // counters run since construction
//
// Refuses a null root, a finished game, and fewer than one simulation.

#include <optional>
#include <random>
#include <span>
#include <vector>

#include "evaluator.h"
#include "snake_env.h"
#include "value_range.h"

// The move the position leaves no choice about, if there is one. Empty covers both
// "two or more survive" and "none does" - either way the caller decides.
std::optional<SnakeEnv::Action> forcedAction(const SnakeEnv& state);

// Monte Carlo tree search over the real simulator, run on many games at once. Every
// tree advances one simulation in lockstep so all their leaves evaluate together.
//
// Nodes hold statistics, never a game state - the descent replays from the root,
// which is cheaper than a snapshot per node. Food placement is sampled per descent
// rather than branched on, and that is the one place this search is not exact.
class MonteCarloSearch
{
public:
    // Every field is set by the caller; the initializers make a miss a wrong number.
    struct Config
    {
        int simulations{ 0 };
        float exploration{ 0.0f };  // c_puct
        float discount{ 0.0f };
        // Paid every tick, so the search prices delay as the training target does.
        float step_reward{ 0.0f };
        // How far below the best visit count a faster action may sit; zero disables.
        float steps_tiebreak_margin{ 0.0f };
        // Whether to refuse a root move that seals the head away from its own tail.
        bool trap_guard{ false };
        // Whether to count seals without acting; the veto would erase the evidence.
        bool trap_report{ false };
        // Whether to count how often two simulations disagree about a shared edge.
        bool alias_report{ false };
        // Whether selection reads an edge's mean over traversals rather than its last
        // write - the Leurent and Maillard 2019 estimator. Backup is unaffected.
        bool average_edges{ false };
        // Whether the root refuses an action over death_cap_threshold, per Fatemi et
        // al. 2019. Refuses only when another action survives; a lost position is not.
        bool death_cap{ false };
        // The risk above which an action is refused, in [0, 1]; read only if capped.
        float death_cap_threshold{ 0.0f };
        // Whether selection normalises a value against the range this tree has seen
        // before comparing it with the prior term, so c_puct means what it does in the
        // paper. Backup is unaffected.
        bool normalize_values{ false };
        // Dirichlet noise on root priors; set the fraction to zero when evaluating.
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
        // Backed-up death risk per root action, in [0, 1]; reported even when uncapped.
        std::vector<float> death_risk;
        // Whether every root action was visited, which is what makes death_risk worth
        // training on; an unvisited action keeps its start value and reads as safe.
        bool all_actions_visited{ false };
        SnakeEnv::Action best_action{ SnakeEnv::Action::STRAIGHT };
    };

    MonteCarloSearch(Evaluator& evaluator, const Config& config);

    // A search has no value semantics: a copy would duplicate the random stream and
    // the descent in flight, and share one evaluator counter between two rate reports.
    MonteCarloSearch(const MonteCarloSearch&) = delete;
    MonteCarloSearch& operator=(const MonteCarloSearch&) = delete;
    MonteCarloSearch(MonteCarloSearch&&) = delete;
    MonteCarloSearch& operator=(MonteCarloSearch&&) = delete;

    // Runs the configured simulations on every root at once. Roots must be live.
    std::vector<Result> search(const std::vector<const SnakeEnv*>& roots);

    // How many times the trap guard has overruled the search since construction.
    long long trapGuardFires() const { return trap_guard_fires_; }

    // Root moves that would seal the head away from its tail, guard or no guard.
    // This is what falls as the network learns; trapGuardFires is only its subset.
    long long sealedChoices() const { return sealed_choices_; }

    // Edge traversals reaching a node whose edge an earlier simulation recorded.
    // Zero when alias_report is off, and zero for a search of one simulation.
    long long revisitedEdges() const { return revisited_edges_; }

    // The subset whose recomputed edge differed; bounded above by revisitedEdges().
    long long aliasedEdges() const { return aliased_edges_; }

    // Narrowed to rewards differing by more than half an apple - one simulation ate
    // and another did not. Only this counter says whether aliasing could move a rate.
    long long materiallyAliasedEdges() const { return materially_aliased_edges_; }

    // The same two counted per node rather than per traversal, which says how much
    // of the tree is affected rather than how much attention it drew.
    long long revisitedNodes() const { return revisited_nodes_; }
    long long aliasedNodes() const { return aliased_nodes_; }

    // Root actions the cap has refused since construction; zero when it is off.
    long long deathCapFires() const { return death_cap_fires_; }

private:
    struct Node
    {
        // Every field carries its default, so a node is formed where it is declared.
        float prior{ 0.0f };
        // Discounted rewards on the entering edge, which forced moves can stretch.
        float reward{ 0.0f };
        float value_sum{ 0.0f };
        // Steps still needed, summed over visits, as a fraction of the budget.
        float steps_sum{ 0.0f };
        // Chance the entering action leads to an unavoidable death, in [0, 1]. The
        // minimum over the node's actions, undiscounted, so it carries no horizon.
        float death_risk{ 0.0f };
        int visit_count{ 0 };
        // Arena index of action 0's child, and the only record that this is expanded.
        std::optional<int> first_child;
        // Ticks the entering edge covers, so backup discounts by elapsed time. One is
        // the floor; the root keeps the default and its computed return is discarded.
        int edge_steps{ 1 };
        bool terminal{ false };
        // The first traversal's edge, kept for the alias probe and read by nothing else.
        float first_reward{ 0.0f };
        int first_edge_steps{ 1 };
        bool edge_recorded{ false };
        // Whether the per-node totals have already counted this node.
        bool revisit_counted{ false };
        bool alias_counted{ false };
        // The entering edge summed over traversals, and their count; the ratios are
        // what selection reads under average_edges. The discount factor is summed
        // rather than the ticks, since gamma^k is convex - docs/prove_discount_jensen.py.
        // Counted apart from visit_count, which a different phase increments.
        float reward_sum{ 0.0f };
        float discount_sum{ 0.0f };
        int edge_traversals{ 0 };
        // The network's steps-to-go, as a fraction of the budget; 1 until expanded, so
        // a node the search never looked at never wins the tie-break.
        float steps_to_go{ 1.0f };
    };

    struct Tree
    {
        std::vector<Node> nodes;
        // The path taken, kept between selection and the backup after evaluation.
        std::vector<int> path;
        // Exactly one live copy; a vector because SnakeEnv has no default constructor.
        std::vector<SnakeEnv> replay;
        // Engaged when the descent ended on a terminal, which is owed no evaluation;
        // empty means this tree has a leaf in the batch awaiting the network.
        std::optional<float> known_leaf_value;
        // The values selection has compared in this tree, for normalising the next
        // comparison. Per tree rather than per search: two trees hold different games.
        ValueRange value_range;
    };

    Evaluator& evaluator_;
    Config config_;
    std::vector<Tree> trees_;
    std::mt19937 rng_;

    // Scratch reused across calls, keeping its capacity - a per-call allocation here
    // would be made tens of thousands of times a game.
    std::vector<const SnakeEnv*> batch_;
    std::vector<float> priors_;
    std::vector<float> values_;
    std::vector<float> steps_;
    std::vector<float> death_risks_;

    // What one visited child is worth: what its edge pays, plus the discounted mean
    // return from where it lands. One definition, because selection both compares this
    // and widens the range with it. Asserts the child has been visited.
    float rawActionValue(const Node& child) const;
    // The PUCT score for one child, given the parent's attention to distribute and the
    // range to measure its value against. An unestablished range leaves the value alone.
    float actionScore(const Node& child, float parent_weight, const ValueRange& range) const;
    // Widens the tree's range with what it compares, so the next descent normalises
    // against values this tree has actually produced.
    int selectChild(Tree& tree, int node_index);
    // Spans rather than bare pointers, so the callee can check the lengths.
    void expand(Tree& tree, int node_index, std::span<const float> priors,
                std::span<const float> death_risks);
    void backup(Tree& tree, float leaf_value, float leaf_steps);

    // The minimum risk over a node's actions: doomed only when every action is.
    // Leaves a childless node alone, so it keeps the estimate it already had.
    void refreshDeathRisk(Tree& tree, int node_index);

    // The budget the steps accumulator is a fraction of, taken from the roots.
    float steps_budget_{ 1.0f };
    long long trap_guard_fires_{ 0 };
    long long sealed_choices_{ 0 };
    long long revisited_edges_{ 0 };
    long long aliased_edges_{ 0 };
    long long materially_aliased_edges_{ 0 };
    long long revisited_nodes_{ 0 };
    long long aliased_nodes_{ 0 };
    long long death_cap_fires_{ 0 };
    void addRootNoise(Tree& tree);
};
