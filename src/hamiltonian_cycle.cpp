#include "hamiltonian_cycle.h"
#include <iostream>
#include <algorithm>
#include <stack>

HamiltonianCycle::HamiltonianCycle(int width, int height) 
    : width_(width), height_(height), rng_(std::random_device{}()) {
    
    if (!isValidGrid()) {
        throw std::invalid_argument("Grid must be at least 2x2 with an even number of rows for this Hamiltonian cycle construction");
    }
    
    cycle_index_.resize(height_, std::vector<int>(width_, -1));
    
    // Initialize MST grid (half dimensions)
    int mst_width = width_ / 2;
    int mst_height = height_ / 2;
    mst_grid_.resize(mst_height, std::vector<MSTNode>(mst_width));
}

bool HamiltonianCycle::isValidGrid() const {
    // The construction in mstToCycle reserves column 0 as the return path and
    // serpentines across columns 1..width-1. Its last row must end adjacent to
    // column 0, which happens only when the row count is even. An even cell
    // count is necessary for any Hamiltonian cycle on a grid but is not
    // sufficient here - a 4-wide, 5-high grid has 20 cells and this
    // construction fails on it, so height parity is what must be checked.
    return width_ >= 2 && height_ >= 2 && height_ % 2 == 0;
}

bool HamiltonianCycle::generateCycle() {
    if (!generateMST()) {
        std::cerr << "Failed to generate MST" << std::endl;
        return false;
    }
    
    if (!mstToCycle()) {
        std::cerr << "Failed to convert MST to Hamiltonian cycle" << std::endl;
        return false;
    }
    
    std::cout << "✓ Generated Hamiltonian cycle with " << cycle_path_.size() << " positions" << std::endl;
    return true;
}

bool HamiltonianCycle::generateMST() {
    int mst_width = width_ / 2;
    int mst_height = height_ / 2;
    
    if (mst_width == 0 || mst_height == 0) {
        return false;
    }
    
    std::cout << "Generating MST for " << mst_width << "x" << mst_height << " grid" << std::endl;
    
    // Prim's algorithm with random weights
    std::vector<std::vector<bool>> in_tree(mst_height, std::vector<bool>(mst_width, false));
    
    // Start from (0,0)
    in_tree[0][0] = true;
    mst_grid_[0][0].visited = true;
    
    int edges_added = 0;
    int total_nodes = mst_width * mst_height;
    
    // Continue until we have a spanning tree
    while (edges_added < total_nodes - 1) {
        // Find minimum weight edge connecting tree to non-tree vertex
        int best_weight = INT_MAX;
        int best_x = -1, best_y = -1;
        bool best_is_right = false;
        
        // Check all possible edges from tree vertices to non-tree vertices
        for (int y = 0; y < mst_height; y++) {
            for (int x = 0; x < mst_width; x++) {
                if (!in_tree[y][x]) continue; // Only from tree vertices
                
                // Check right edge
                if (x < mst_width - 1 && !in_tree[y][x + 1]) {
                    int weight = rng_() % 1000;
                    if (weight < best_weight) {
                        best_weight = weight;
                        best_x = x;
                        best_y = y;
                        best_is_right = true;
                    }
                }
                
                // Check down edge
                if (y < mst_height - 1 && !in_tree[y + 1][x]) {
                    int weight = rng_() % 1000;
                    if (weight < best_weight) {
                        best_weight = weight;
                        best_x = x;
                        best_y = y;
                        best_is_right = false;
                    }
                }
            }
        }
        
        // If no edge found, we're done (shouldn't happen in connected grid)
        if (best_x == -1) {
            std::cout << "No more edges found, stopping at " << edges_added << " edges" << std::endl;
            break;
        }
        
        // Add the best edge to MST
        int next_x = best_is_right ? best_x + 1 : best_x;
        int next_y = best_is_right ? best_y : best_y + 1;
        
        if (best_is_right) {
            mst_grid_[best_y][best_x].canGoRight = true;
        } else {
            mst_grid_[best_y][best_x].canGoDown = true;
        }
        
        // Add target node to tree
        in_tree[next_y][next_x] = true;
        mst_grid_[next_y][next_x].visited = true;
        edges_added++;
        
        if (edges_added % 10 == 0) {
            std::cout << "MST progress: " << edges_added << "/" << (total_nodes - 1) << " edges" << std::endl;
        }
    }
    
    std::cout << "✓ Generated MST with " << edges_added << " edges (needed " << (total_nodes - 1) << ")" << std::endl;
    return edges_added == total_nodes - 1;
}

bool HamiltonianCycle::mstToCycle() {
    // Serpentine across columns 1..width-1, then return up column 0.
    //
    // A plain boustrophedon over the whole grid is a Hamiltonian PATH, not a
    // cycle: on a 20x20 grid it starts at (0,0) and ends at (0,19), which are
    // nineteen cells apart, so closing it produces one move the snake cannot
    // make. Reserving column 0 as a dedicated return path closes the loop.
    //
    // Rows are traversed alternately rightward and leftward over columns
    // 1..width-1. With an even row count the final row is traversed leftward
    // and so ends at column 1, one step from column 0; the walk then runs up
    // column 0 to (0,0), which is adjacent to the start cell (1,0).

    cycle_path_.clear();
    next_in_cycle_.clear();

    for (int row = 0; row < height_; row++) {
        if (row % 2 == 0) {
            for (int col = 1; col < width_; col++) {
                cycle_path_.push_back({col, row});
            }
        } else {
            for (int col = width_ - 1; col >= 1; col--) {
                cycle_path_.push_back({col, row});
            }
        }
    }

    // Return path: up the reserved column, from the last row back to row 0.
    for (int row = height_ - 1; row >= 0; row--) {
        cycle_path_.push_back({0, row});
    }

    // Build the cycle index mapping and next position mapping
    for (size_t i = 0; i < cycle_path_.size(); i++) {
        Position pos = cycle_path_[i];
        cycle_index_[pos.y][pos.x] = static_cast<int>(i);
        
        // Set next position in cycle
        Position next_pos = cycle_path_[(i + 1) % cycle_path_.size()];
        next_in_cycle_[pos] = next_pos;
    }
    
    std::cout << "Created serpentine Hamiltonian cycle with " << cycle_path_.size() << " positions" << std::endl;
    std::cout << "Start: (" << cycle_path_[0].x << "," << cycle_path_[0].y << ")" << std::endl;
    std::cout << "End: (" << cycle_path_.back().x << "," << cycle_path_.back().y << ")" << std::endl;
    std::cout << "End connects to start: (" << next_in_cycle_[cycle_path_.back()].x << "," << next_in_cycle_[cycle_path_.back()].y << ")" << std::endl;
    
    return cycle_path_.size() == width_ * height_;
}

Position HamiltonianCycle::getNext(const Position& current) const {
    auto it = next_in_cycle_.find(current);
    if (it != next_in_cycle_.end()) {
        return it->second;
    }
    return current; // Fallback
}

int HamiltonianCycle::getCycleDistance(const Position& from, const Position& to) const {
    int from_idx = getCycleIndex(from);
    int to_idx = getCycleIndex(to);
    
    if (from_idx == -1 || to_idx == -1) return -1;
    
    int total_size = static_cast<int>(cycle_path_.size());
    if (to_idx >= from_idx) {
        return to_idx - from_idx;
    } else {
        return (total_size - from_idx) + to_idx;
    }
}

int HamiltonianCycle::getCycleIndex(const Position& pos) const {
    if (pos.x >= 0 && pos.x < width_ && pos.y >= 0 && pos.y < height_) {
        return cycle_index_[pos.y][pos.x];
    }
    return -1;
}

bool HamiltonianCycle::isShortcutSafe(const Position& head, const Position& tail, 
                                    const Position& newPos, int snakeLength) const {
    int head_idx = getCycleIndex(head);
    int tail_idx = getCycleIndex(tail);
    int new_idx = getCycleIndex(newPos);
    
    if (head_idx == -1 || tail_idx == -1 || new_idx == -1) return false;
    
    // Check if snake remains ordered in cycle: tail < body < head
    int total_size = static_cast<int>(cycle_path_.size());
    
    // Calculate space available after shortcut
    int space_after_head = getCycleDistance({newPos.x, newPos.y}, tail);
    
    // Need enough space for snake body + growth buffer
    return space_after_head >= snakeLength + 2;
}

std::vector<Position> HamiltonianCycle::getNeighbors(const Position& pos) const {
    std::vector<Position> neighbors;
    
    // Up, Down, Left, Right
    std::vector<Position> dirs = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
    
    for (const auto& dir : dirs) {
        Position next = {pos.x + dir.x, pos.y + dir.y};
        if (next.x >= 0 && next.x < width_ && next.y >= 0 && next.y < height_) {
            neighbors.push_back(next);
        }
    }
    
    return neighbors;
}

void HamiltonianCycle::printCycle() const {
    std::cout << "Hamiltonian Cycle (" << width_ << "x" << height_ << "):" << std::endl;
    
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            int idx = cycle_index_[y][x];
            if (idx < 10) std::cout << " ";
            if (idx < 100) std::cout << " ";
            std::cout << idx << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
