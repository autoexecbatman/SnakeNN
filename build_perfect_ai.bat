@echo off
REM Configures, builds and runs PerfectSnakeAI - the visual agent that follows a
REM Hamiltonian cycle and fills the board.
REM
REM   build_perfect_ai.bat            builds Release
REM   build_perfect_ai.bat Debug      builds Debug, where assertions are live
REM
REM Paths are relative to this file, so the repository can be cloned anywhere.
REM The vcpkg toolchain is the one thing that is not: override VCPKG_TOOLCHAIN in
REM the environment if yours lives elsewhere.

setlocal

set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=Release"

set "ROOT=%~dp0"
if "%VCPKG_TOOLCHAIN%"=="" set "VCPKG_TOOLCHAIN=E:\dev\vcpkg\scripts\buildsystems\vcpkg.cmake"

echo Configuring...
cmake -S "%ROOT%." -B "%ROOT%build" -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE="%VCPKG_TOOLCHAIN%"
if errorlevel 1 (
    echo [FAILED] CMake configuration.
    exit /b 1
)

echo Building PerfectSnakeAI in %CONFIG%...
cmake --build "%ROOT%build" --config %CONFIG% --target PerfectSnakeAI
if errorlevel 1 (
    echo [FAILED] Build.
    exit /b 1
)

echo [OK] Running PerfectSnakeAI...
"%ROOT%build\%CONFIG%\PerfectSnakeAI.exe"

endlocal
