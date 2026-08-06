@echo off
cd /d D:\repo\snakeNN
echo Building Snake AI with fixed collision detection...
echo.

if not exist build mkdir build
cd build

cmake .. -DCMAKE_TOOLCHAIN_FILE=E:\dev\vcpkg\scripts\buildsystems\vcpkg.cmake
if %errorlevel% neq 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

cmake --build . --config Release --target PerfectSnakeAI
if %errorlevel% neq 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo Build successful! Running PerfectSnakeAI...
cd Release
PerfectSnakeAI.exe

pause
