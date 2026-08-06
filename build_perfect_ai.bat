@echo off
echo Building Perfect Snake AI with Hamiltonian Cycle Implementation...
echo Based on academic research: Umans and Lenhart IEEE 1997

cd /d "D:\repo\snakeNN\build"

echo Configuring CMake...
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE="E:\dev\vcpkg\scripts\buildsystems\vcpkg.cmake"

echo Building PerfectSnakeAI...
cmake --build . --config Debug --target PerfectSnakeAI

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ Build successful! Running Academic Perfect Snake AI...
    echo.
    ".\Debug\PerfectSnakeAI.exe"
) else (
    echo.
    echo ✗ Build failed! Check errors above.
    pause
)
