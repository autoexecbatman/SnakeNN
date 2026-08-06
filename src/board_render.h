#pragma once
#include <raylib.h>
#include "snake_logic.h"  // Direction, Position
#include <vector>

// Shared presentation for anything that draws a snake board.
//
// Extracted from the Hamiltonian demo when the learned agent needed the same
// picture: two copies of a renderer diverge, and the second one always ends up
// being the one nobody updates. Layout, palette, fonts and the board drawing
// live here; each demo keeps only its own panel, because the numbers worth
// showing differ between a fixed policy and a searching one.
namespace ui {

constexpr int CELL = 22;
constexpr int GAP = 16;
constexpr int MARGIN = GAP + GAP / 2;
constexpr int HEADER_HEIGHT = 74;

constexpr Color BACKGROUND = {13, 16, 22, 255};
constexpr Color SURFACE = {21, 26, 34, 255};
constexpr Color SURFACE_EDGE = {38, 46, 58, 255};
constexpr Color GRID_LINE = {29, 35, 45, 255};
constexpr Color TEXT_PRIMARY = {233, 238, 245, 255};
constexpr Color TEXT_MUTED = {150, 163, 182, 255};
constexpr Color MINT = {84, 224, 168, 255};
constexpr Color MINT_DEEP = {24, 108, 88, 255};
constexpr Color AMBER = {245, 176, 66, 255};

constexpr const char* DISPLAY_FONT_PATH = "C:/Windows/Fonts/segoeui.ttf";
constexpr const char* LABEL_FONT_PATH = "C:/Windows/Fonts/segoeuib.ttf";
constexpr const char* MONO_FONT_PATH = "C:/Windows/Fonts/CascadiaMono.ttf";
constexpr int FONT_BAKE_SIZE = 64;

// The three faces every panel here uses: display text, bold small caps for
// labels, and a monospace for figures so digits hold their column while
// counters run.
struct Fonts {
    Font display{};
    Font label{};
    Font mono{};
};

Fonts loadFonts();
void unloadFonts(Fonts& fonts);

int boardPixelWidth(int columns);
int boardPixelHeight(int rows);

void drawLabel(const Font& font, const char* text, float x, float y, Color color);
void drawText(const Font& font, const char* text, float x, float y, float size, float spacing,
              Color color);
float textWidth(const Font& font, const char* text, float size, float spacing);

// Draws the grid, the food and the snake at the standard board origin. `body`
// runs head-first, as both game classes store it. Pass show_food false on a won
// board, where the food no longer exists.
void drawBoard(int columns, int rows, const std::vector<Position>& body, Position food,
               Direction heading, bool show_food);

}  // namespace ui
