#pragma once

#include <raylib.h>

#include <vector>

#include "snake_logic.h"  // Direction, Position

// Layout, palette, fonts and the board drawing, shared by every program that renders a
// snake. A caller supplies its own panel - the numbers worth showing differ between a
// fixed policy and a searching one - and takes the board from here.
//
// Usage:
//
//     InitWindow(ui::boardPixelWidth(20), ui::boardPixelHeight(20) + ui::HEADER_HEIGHT,
//                "snake");
//     const ui::Fonts fonts = ui::loadFonts();   // after InitWindow, before drawing
//
//     BeginDrawing();
//     ClearBackground(ui::BACKGROUND);
//     ui::drawBoard(20, 20, game.body(), game.food(), game.heading(), true);
//     ui::drawLabel(fonts.label, "SCORE", ui::MARGIN, 20.0f, ui::TEXT_MUTED);
//     EndDrawing();
//
//     ui::unloadFonts(fonts);   // before CloseWindow
namespace ui {

// One cell's side in pixels; every board dimension is a multiple of it.
constexpr int CELL = 22;
// The spacing unit panels are laid out on. MARGIN is the page edge, one and a half of it.
constexpr int GAP = 16;
constexpr int MARGIN = GAP + GAP / 2;
// Height reserved above the board for a caller's own panel.
constexpr int HEADER_HEIGHT = 74;

// The palette, darkest first: the window behind everything, a panel on it, that panel's
// border, and the lines between cells.
constexpr Color BACKGROUND = {13, 16, 22, 255};
constexpr Color SURFACE = {21, 26, 34, 255};
constexpr Color SURFACE_EDGE = {38, 46, 58, 255};
constexpr Color GRID_LINE = {29, 35, 45, 255};

// Body text, and the dimmer weight labels are drawn in.
constexpr Color TEXT_PRIMARY = {233, 238, 245, 255};
constexpr Color TEXT_MUTED = {150, 163, 182, 255};

// The snake's head, its body, and the food.
constexpr Color MINT = {84, 224, 168, 255};
constexpr Color MINT_DEEP = {24, 108, 88, 255};
constexpr Color AMBER = {245, 176, 66, 255};

// System faces loadFonts reads. A missing one falls back to raylib's built-in face with
// a warning rather than failing.
constexpr const char* DISPLAY_FONT_PATH = "C:/Windows/Fonts/segoeui.ttf";
constexpr const char* LABEL_FONT_PATH = "C:/Windows/Fonts/segoeuib.ttf";
constexpr const char* MONO_FONT_PATH = "C:/Windows/Fonts/CascadiaMono.ttf";

// Glyphs are baked at this size and filtered down, which keeps them sharp at every size
// drawn here.
constexpr int FONT_BAKE_SIZE = 64;

// The three faces every panel here uses. loadFonts fills it; the fields say what each
// one is for.
struct Fonts {
    // Body and headings.
    Font display{};
    // Bold, for the small-caps labels above figures.
    Font label{};
    // Fixed width, so a running counter does not shift its column.
    Font mono{};
};

// Loads the three faces. Call after InitWindow: raylib needs a GL context to bake a
// texture.
Fonts loadFonts();
// Releases them, before CloseWindow. Leaves the built-in fallback face alone.
void unloadFonts(Fonts& fonts);

// The board's width in pixels: CELL per column, with no margin.
int boardPixelWidth(int columns);
// Its height, likewise per row. A window needs HEADER_HEIGHT on top of this.
int boardPixelHeight(int rows);

// A small-caps label at the house size and spacing - the caller chooses only where and
// what colour.
void drawLabel(const Font& font, const char* text, float x, float y, Color color);
// Any text, with the size and letter spacing left to the caller.
void drawText(const Font& font, const char* text, float x, float y, float size, float spacing,
              Color color);
// What drawText would occupy, for laying a figure out against its label.
float textWidth(const Font& font, const char* text, float size, float spacing);

// Draws the grid, the food and the snake at the standard board origin. `body`
// runs head-first, as both game classes store it. Pass show_food false on a won
// board, where the food no longer exists.
void drawBoard(int columns, int rows, const std::vector<Position>& body, Position food,
               Direction heading, bool show_food);

}  // namespace ui
