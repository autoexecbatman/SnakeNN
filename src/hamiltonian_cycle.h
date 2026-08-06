#pragma once
#include "snake_logic.h"  // Use existing Position struct
#include <vector>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>

struct PositionHash {
    size_t operator()(const Position& pos) const {
        return std::hash<int>()(pos.x) ^ (std::hash<int>()(pos.y) << 1);
    }
};

// Hamiltonian Cycle Generator for Snake Game
// Based on academic research: Umans & Lenhart (IEEE 1997)
class HamiltonianCycle {
public:
    HamiltonianCycle(int width, int height);
    
    // Generate Hamiltonian cycle using Prim's algorithm approach
    bool generateCycle();
    
    // Get next position in cycle from current position
    Position getNext(const Position& current) const;
    
    // Get distance between two positions in the cycle
    int getCycleDistance(const Position& from, const Position& to) const;
    
    // Check if taking a shortcut is safe (maintains snake ordering)
    bool isShortcutSafe(const Position& head, const Position& tail, 
                       const Position& newPos, int snakeLength) const;
    
    // Get the cycle index of a position
    int getCycleIndex(const Position& pos) const;
    
    // Debug: print the cycle
    void printCycle() const;
    
private:
    int width_, height_;
    std::vector<std::vector<int>> cycle_index_;  // Maps position to cycle index
    std::vector<Position> cycle_path_;           // Ordered cycle positions
    std::unordered_map<Position, Position, PositionHash> next_in_cycle_;
    
    // Generate MST using Prim's algorithm (half-size grid)
    bool generateMST();
    
    // Convert MST to Hamiltonian cycle using wall-following
    bool mstToCycle();
    
    // Check if grid dimensions are valid for Hamiltonian cycle
    bool isValidGrid() const;
    
    // Get neighbors of a position
    std::vector<Position> getNeighbors(const Position& pos) const;
    
    // MST data structures
    struct MSTNode {
        bool visited = false;
        bool canGoRight = false;
        bool canGoDown = false;
    };
    
    std::vector<std::vector<MSTNode>> mst_grid_;
    std::mt19937 rng_;
};
