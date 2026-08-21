// Implementation of the shared board renderer. What it draws and how to call it are in
// board_render.h.
//
// Every drawing call here goes through raylib and needs a window: loadFonts bakes a
// texture, so nothing in this file works before InitWindow.

#include <cmath>
#include <iostream>

#include "board_render.h"

namespace ui {

namespace {

Font loadOne(const char* path) {
    if (!FileExists(path)) {
        std::cerr << "[UI] Font not found, falling back to the built-in face: " << path
                  << std::endl;
        return GetFontDefault();
    }
    // Baked well above display size and filtered down, which is what keeps
    // glyphs sharp at every size drawn here.
    Font font = LoadFontEx(path, FONT_BAKE_SIZE, nullptr, 0);
    SetTextureFilter(font.texture, TEXTURE_FILTER_BILINEAR);
    return font;
}

void unloadOne(Font& font) {
    if (font.texture.id != GetFontDefault().texture.id) {
        UnloadFont(font);
    }
}

Rectangle cellRect(const Position& cell, float inset) {
    return {MARGIN + cell.x * (float)CELL + inset, HEADER_HEIGHT + cell.y * (float)CELL + inset,
            CELL - inset * 2.0f, CELL - inset * 2.0f};
}

}  // namespace

Fonts loadFonts() {
    Fonts fonts;
    fonts.display = loadOne(DISPLAY_FONT_PATH);
    fonts.label = loadOne(LABEL_FONT_PATH);
    fonts.mono = loadOne(MONO_FONT_PATH);
    return fonts;
}

void unloadFonts(Fonts& fonts) {
    unloadOne(fonts.display);
    unloadOne(fonts.label);
    unloadOne(fonts.mono);
}

int boardPixelWidth(int columns) { return CELL * columns; }
int boardPixelHeight(int rows) { return CELL * rows; }

void drawLabel(const Font& font, const char* text, float x, float y, Color color) {
    DrawTextEx(font, text, {x, y}, 12.0f, 1.4f, color);
}

void drawText(const Font& font, const char* text, float x, float y, float size, float spacing,
              Color color) {
    DrawTextEx(font, text, {x, y}, size, spacing, color);
}

float textWidth(const Font& font, const char* text, float size, float spacing) {
    return MeasureTextEx(font, text, size, spacing).x;
}

void drawBoard(int columns, int rows, const std::vector<Position>& body, Position food,
               Direction heading, bool show_food) {
    const float board_width = (float)boardPixelWidth(columns);
    const float board_height = (float)boardPixelHeight(rows);

    Rectangle board = {MARGIN - 2.0f, HEADER_HEIGHT - 2.0f, board_width + 4.0f,
                       board_height + 4.0f};
    DrawRectangleRounded(board, 0.02f, 8, SURFACE);
    DrawRectangleRoundedLines(board, 0.02f, 8, SURFACE_EDGE);

    // Separators this close in value to the surface read as texture rather than
    // as a table.
    for (int column = 1; column < columns; column++) {
        float x = MARGIN + column * (float)CELL;
        DrawLineV({x, (float)HEADER_HEIGHT}, {x, HEADER_HEIGHT + board_height}, GRID_LINE);
    }
    for (int row = 1; row < rows; row++) {
        float y = HEADER_HEIGHT + row * (float)CELL;
        DrawLineV({(float)MARGIN, y}, {MARGIN + board_width, y}, GRID_LINE);
    }

    if (show_food) {
        // Food pulses so the eye can find it on a board that is mostly snake.
        Rectangle food_cell = cellRect(food, 0.0f);
        Vector2 centre = {food_cell.x + food_cell.width / 2.0f,
                          food_cell.y + food_cell.height / 2.0f};
        float pulse = 0.5f + 0.5f * sinf((float)GetTime() * 4.0f);
        DrawCircleV(centre, CELL * (0.46f + 0.10f * pulse), Fade(AMBER, 0.16f));
        DrawCircleV(centre, CELL * 0.26f, AMBER);
    }

    if (body.empty()) {
        return;
    }

    // Tail first, so the head draws on top. The fade from mint at the head to
    // deep teal at the tail makes direction of travel readable in a still frame.
    for (size_t index = body.size(); index-- > 0; ) {
        float along = body.size() > 1 ? (float)index / (float)(body.size() - 1) : 0.0f;
        Color segment = {(unsigned char)(MINT.r + (MINT_DEEP.r - MINT.r) * along),
                         (unsigned char)(MINT.g + (MINT_DEEP.g - MINT.g) * along),
                         (unsigned char)(MINT.b + (MINT_DEEP.b - MINT.b) * along), 255};
        DrawRectangleRounded(cellRect(body[index], 1.5f), 0.35f, 6, segment);
    }

    Rectangle head = cellRect(body[0], 1.0f);
    DrawRectangleRounded(head, 0.35f, 6, MINT);
    Vector2 head_centre = {head.x + head.width / 2.0f, head.y + head.height / 2.0f};
    Vector2 facing = {0.0f, 0.0f};
    switch (heading) {
        case Direction::UP: facing = {0.0f, -1.0f}; break;
        case Direction::DOWN: facing = {0.0f, 1.0f}; break;
        case Direction::LEFT: facing = {-1.0f, 0.0f}; break;
        case Direction::RIGHT: facing = {1.0f, 0.0f}; break;
    }
    Vector2 across = {-facing.y, facing.x};
    for (float side : {-1.0f, 1.0f}) {
        Vector2 eye = {head_centre.x + facing.x * CELL * 0.18f + across.x * CELL * 0.20f * side,
                       head_centre.y + facing.y * CELL * 0.18f + across.y * CELL * 0.20f * side};
        DrawCircleV(eye, CELL * 0.09f, BACKGROUND);
    }
}

}  // namespace ui
