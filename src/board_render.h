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
namespace ui
{

// One cell's side in pixels; every board dimension is a multiple of it.
constexpr int CELL = 22;
// The spacing unit panels are laid out on.
constexpr int GAP = 16;
// The page edge: the gap between the window border and anything drawn.
constexpr int MARGIN = GAP + GAP / 2;
// Height reserved above the board for a caller's own panel.
constexpr int HEADER_HEIGHT = 74;

// The window behind everything.
constexpr Color BACKGROUND = { 13, 16, 22, 255 };
// A panel drawn on that background - the board itself is one.
constexpr Color SURFACE = { 21, 26, 34, 255 };
// The one-pixel outline around a panel.
constexpr Color SURFACE_EDGE = { 38, 46, 58, 255 };
// The lines between cells.
constexpr Color GRID_LINE = { 29, 35, 45, 255 };

// Body text and figures.
constexpr Color TEXT_PRIMARY = { 233, 238, 245, 255 };
// The dimmer weight, for the small-caps labels above figures.
constexpr Color TEXT_MUTED = { 150, 163, 182, 255 };

// The head, and the near end of the body.
constexpr Color MINT = { 84, 224, 168, 255 };
// The tail end. The body is a gradient from MINT to this, so length is visible.
constexpr Color MINT_DEEP = { 24, 108, 88, 255 };
// The food.
constexpr Color AMBER = { 245, 176, 66, 255 };

// The face loadFonts reads for Fonts::display. A missing file falls back to raylib's
// built-in face with a warning rather than failing, and the same holds for the two below.
constexpr const char* DISPLAY_FONT_PATH = "C:/Windows/Fonts/segoeui.ttf";
// The bold face, for Fonts::label.
constexpr const char* LABEL_FONT_PATH = "C:/Windows/Fonts/segoeuib.ttf";
// The fixed-width face, for Fonts::mono.
constexpr const char* MONO_FONT_PATH = "C:/Windows/Fonts/CascadiaMono.ttf";

// Glyphs are baked at this size and filtered down, which keeps them sharp at every size
// drawn here.
constexpr int FONT_BAKE_SIZE = 64;

// The three faces every panel here uses. loadFonts fills it; the fields say what each
// one is for.
struct Fonts
{
    // Body and headings.
    Font display{};
    // Bold, for the small-caps labels above figures.
    Font label{};
    // Fixed width, so a running counter does not shift its column.
    Font mono{};
};

// Loads the three faces. Call after InitWindow: raylib needs a GL context to bake a
// texture. Pair every call with unloadFonts.
//
//     const Fonts fonts = loadFonts();
Fonts loadFonts();
// Releases them, before CloseWindow. Leaves the built-in fallback face alone, so it is
// safe on a Fonts whose files were all missing.
//
//     unloadFonts(fonts);
void unloadFonts(Fonts& fonts);

// The board's width in pixels: CELL per column, with no margin.
//
//     boardPixelWidth(20)   // 440, at CELL == 22
int boardPixelWidth(int columns);
// Its height, likewise per row. A window needs HEADER_HEIGHT on top of this.
//
//     InitWindow(boardPixelWidth(20), boardPixelHeight(20) + HEADER_HEIGHT, "snake");
int boardPixelHeight(int rows);

// A small-caps label at the house size and spacing - the caller chooses only where and
// what colour. Pass Fonts::label; the text is drawn as given, so write it upper case.
//
//     drawLabel(fonts.label, "SCORE", MARGIN, 20.0f, TEXT_MUTED);
void drawLabel(const Font& font, const char* text, float x, float y, Color color);
// Any text, with the size and letter spacing left to the caller. Use Fonts::mono for a
// figure that changes, so it does not shift its column as digits change width.
//
//     // font, text, x, y, size, letter spacing, colour
//     drawText(fonts.mono, "137", MARGIN, 40.0f, 27.0f, 0.5f, TEXT_PRIMARY);
void drawText(const Font& font, const char* text, float x, float y, float size, float spacing,
              Color color);
// How wide drawText would draw that text, for laying a figure out against its label.
// Takes the same size and spacing, or the answer is for different text than you draw.
//
//     // Right-align a figure against the panel's right edge.
//     const float x = right_edge - textWidth(fonts.mono, "137", 27.0f, 0.5f);
float textWidth(const Font& font, const char* text, float size, float spacing);

// Draws the grid, the food and the snake at the standard board origin. `body` runs
// head-first, as both game classes store it. Call between BeginDrawing and EndDrawing.
//
//     // columns, rows, segments head-first, apple, heading, draw the apple
//     drawBoard(20, 20, game.body(), game.food(), game.heading(), true);
//
//     // A won board has no apple left, so the last argument goes false.
//     drawBoard(20, 20, game.body(), game.food(), game.heading(), false);
void drawBoard(int columns, int rows, const std::vector<Position>& body, Position food,
               Direction heading, bool show_food);

}  // namespace ui
