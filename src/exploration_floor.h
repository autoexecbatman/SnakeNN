#pragma once

// How often selection ignores its own scores and picks an action uniformly.
//
// PUCT scores an action as its value plus c_puct * prior * sqrt(N) / (1 + n). Every term
// of the exploration half multiplies the prior, so a prior near zero yields an exploration
// term near zero and no constant recovers it. Measured on az10_death368: 46 percent of
// positions have a top prior above 0.999, the search visits 1.13 root actions of 3, and
// sweeping c_puct over a hundredfold moved that to 1.24.
//
// Xiao et al. 2019 (MENTS, NeurIPS) mix the tree policy with a uniform one instead:
// pi(a|s) = (1 - lambda) f(Q)(a) + lambda / |A|, with lambda = eps |A| / log(sum N + 1).
// The floor is additive, so it does not ask the network's permission. This is that
// mixture in the shape a PUCT search can use - a weight saying how often to bypass the
// argmax entirely, since flooring the prior inside the argmax would leave the value term
// dominating and change nothing.
//
// Usage:
//
//     const float weight = explorationMixWeight(0.1f, 3, 200);  // eps, actions, visits
//     // 0.0566 - so about 11 of 200 selections at this node go uniform, ~3.8 per action
//
//     if (uniform_draw < weight) { /* pick an action uniformly */ }
//     else                       { /* the usual PUCT argmax */ }
//
//     explorationMixWeight(0.0f, 3, 200);   // 0.0 exactly - the floor is off
//     explorationMixWeight(0.1f, 3, 0);     // 1.0 - nothing visited, so explore outright
//
// Decay is 1/log(N), which is slow on purpose: the floor is meant to outlast the point
// where the policy has made up its mind. The result is clamped to 1, because at small
// visit counts the formula exceeds it.
//
// Throws std::invalid_argument on a negative epsilon, an action count below 1, or a
// negative visit count.
float explorationMixWeight(float epsilon, int action_count, int total_visits);

// The mix weight for one node of a descent: the floor applies at the root and nowhere
// below it.
//
// The weight above decays in the node's own visit count, so it is 0.057 at a root with
// 200 visits and 0.433 at a node with one - and 200 simulations leave most of the tree
// at one visit. A descent that goes uniform at two of every five deep steps is a random
// walk, and it measures as one: epsilon 0.1 at every node took self-play from 97.5
// apples to 3.8 with no wins, while the labels the floor was built for did arrive, 126
// of 128 per batch. The root is where the training label is read, so it is the only
// place the coverage has to exist.
//
// Usage, from the descent in mcts.cpp:
//
//     const float weight = descentMixWeight(0.1f, 3, node.visit_count, depth);
//     // depth 0, 200 visits -> 0.0566;  any depth above 0 -> 0.0 exactly
//
// Throws std::invalid_argument on a negative depth, and on whatever
// explorationMixWeight refuses.
float descentMixWeight(float epsilon, int action_count, int node_visits, int depth);
