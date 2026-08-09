cmake_minimum_required(VERSION 3.18)
project(SnakeNN CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# src/ holds what the project still measures with: the AlphaZero stack, and the
# winning path (snake_logic, hamiltonian_cycle, cycle_agent) with the instruments
# that measure it. legacy/ holds one main() per file, each the record of one past
# run. Both directories are on the include path because the legacy programs include
# "snake_logic.h" from their own directory, and because perfect_snake_ai.cpp still
# includes the legacy "neural_network.h" for three comparison scorers it never
# calls. That one dependency is the only thing crossing the boundary in that
# direction, and nothing in the build stops a new one being added - the split is a
# convention here, not an enforced barrier.
include_directories(${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/legacy)

# Find vcpkg packages
find_package(raylib CONFIG REQUIRED)

# Direct LibTorch paths - bypass problematic cmake scripts
set(TORCH_INCLUDE_DIRS 
    "D:/libtorch-cuda/libtorch/include"
    "D:/libtorch-cuda/libtorch/include/torch/csrc/api/include"
)
set(TORCH_LIBRARIES 
    "D:/libtorch-cuda/libtorch/lib/torch.lib"
    "D:/libtorch-cuda/libtorch/lib/torch_cpu.lib"
    "D:/libtorch-cuda/libtorch/lib/torch_cuda.lib"
    "D:/libtorch-cuda/libtorch/lib/c10.lib"
    "D:/libtorch-cuda/libtorch/lib/c10_cuda.lib"
)

# Function to copy PyTorch DLLs
function(copy_pytorch_dlls target_name)
    add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "D:/libtorch-cuda/libtorch/lib/torch.dll"
            "D:/libtorch-cuda/libtorch/lib/torch_cpu.dll"
            "D:/libtorch-cuda/libtorch/lib/torch_cuda.dll"
            "D:/libtorch-cuda/libtorch/lib/c10.dll"
            "D:/libtorch-cuda/libtorch/lib/c10_cuda.dll"
            "D:/libtorch-cuda/libtorch/lib/torch_global_deps.dll"
            "D:/libtorch-cuda/libtorch/lib/asmjit.dll"
            "D:/libtorch-cuda/libtorch/lib/fbgemm.dll"
            "D:/libtorch-cuda/libtorch/lib/libiomp5md.dll"
            "D:/libtorch-cuda/libtorch/lib/uv.dll"
            "D:/libtorch-cuda/libtorch/lib/zlibwapi.dll"
            "D:/libtorch-cuda/libtorch/lib/cudart64_12.dll"
            "D:/libtorch-cuda/libtorch/lib/cublas64_12.dll"
            "D:/libtorch-cuda/libtorch/lib/cublasLt64_12.dll"
            "D:/libtorch-cuda/libtorch/lib/curand64_10.dll"
            "D:/libtorch-cuda/libtorch/lib/cupti64_2025.1.0.dll"
            "D:/libtorch-cuda/libtorch/lib/cufft64_11.dll"
            "D:/libtorch-cuda/libtorch/lib/cusolver64_11.dll"
            "D:/libtorch-cuda/libtorch/lib/cusparse64_12.dll"
            "D:/libtorch-cuda/libtorch/lib/nvrtc64_120_0.dll"
            "D:/libtorch-cuda/libtorch/lib/nvJitLink_120_0.dll"
            "D:/libtorch-cuda/libtorch/lib/nvToolsExt64_1.dll"
            "D:/libtorch-cuda/libtorch/lib/cudnn64_9.dll"
            "D:/libtorch-cuda/libtorch/lib/cudnn_ops64_9.dll"
            "D:/libtorch-cuda/libtorch/lib/cudnn_cnn64_9.dll"
            "D:/libtorch-cuda/libtorch/lib/cudnn_adv64_9.dll"
            "D:/libtorch-cuda/libtorch/lib/cudnn_engines_precompiled64_9.dll"
            "D:/libtorch-cuda/libtorch/lib/cudnn_engines_runtime_compiled64_9.dll"
            "D:/libtorch-cuda/libtorch/lib/cudnn_graph64_9.dll"
            "D:/libtorch-cuda/libtorch/lib/cudnn_heuristic64_9.dll"
            $<TARGET_FILE_DIR:${target_name}>
        COMMENT "Copying PyTorch DLLs for ${target_name}")
endfunction()

# Snake game library (no PyTorch dependency)
add_library(SnakeGameLib
    src/snake_logic.cpp
)

target_link_libraries(SnakeGameLib raylib)

# Simple trainer test - minimal version
add_executable(SimpleTrainer
    legacy/snake_trainer.cpp
    legacy/network_nn.cpp
)

target_include_directories(SimpleTrainer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(SimpleTrainer SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET SimpleTrainer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(SimpleTrainer)

# Define CUDA support
target_compile_definitions(SimpleTrainer PRIVATE USE_CUDA=1)
target_compile_definitions(SimpleTrainer PRIVATE TORCH_CUDA=1)

# Windows CUDA linker fix
if(WIN32)
    target_link_options(SimpleTrainer PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

# Test executables
add_executable(NeuralNetworkTest
    legacy/minimal_test.cpp
)

target_include_directories(NeuralNetworkTest PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(NeuralNetworkTest ${TORCH_LIBRARIES})
set_property(TARGET NeuralNetworkTest PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(NeuralNetworkTest)

add_executable(SnakeGame
    legacy/snake_game.cpp
)

target_include_directories(SnakeGame PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(SnakeGame SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET SnakeGame PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(SnakeGame)

add_executable(SnakeTest
    legacy/snake_test.cpp
    legacy/network_nn.cpp
)

target_include_directories(SnakeTest PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(SnakeTest SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET SnakeTest PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(SnakeTest)

# Diagnostic test executable
add_executable(DiagnosticTest
    legacy/diagnostic_test.cpp
    legacy/network_nn.cpp
)

target_include_directories(DiagnosticTest PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(DiagnosticTest SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET DiagnosticTest PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(DiagnosticTest)

# Simple agent test (no neural network)
add_executable(SimpleAgentTest
    legacy/simple_agent_test.cpp
)

target_link_libraries(SimpleAgentTest SnakeGameLib)
set_property(TARGET SimpleAgentTest PROPERTY CXX_STANDARD 20)

# Simple neural agent test
add_executable(SimpleNeuralTest
    legacy/simple_neural_test.cpp
    legacy/network_nn.cpp
)

target_include_directories(SimpleNeuralTest PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(SimpleNeuralTest SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET SimpleNeuralTest PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(SimpleNeuralTest)

# Fixed trainer with all improvements
add_executable(FixedTrainer
    legacy/fixed_trainer_clean.cpp
    legacy/network_nn.cpp
)

target_include_directories(FixedTrainer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(FixedTrainer SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET FixedTrainer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(FixedTrainer)

# Simple fixed trainer
add_executable(SimpleFixed
    legacy/simple_fixed.cpp
    legacy/network_nn.cpp
)

target_include_directories(SimpleFixed PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(SimpleFixed SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET SimpleFixed PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(SimpleFixed)

# Model tester
add_executable(ModelTester
    legacy/model_tester.cpp
    legacy/network_nn.cpp
)

target_include_directories(ModelTester PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(ModelTester SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET ModelTester PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(ModelTester)

# Visual demo
add_executable(VisualDemo
    legacy/visual_demo.cpp
    legacy/network_nn.cpp
)

target_include_directories(VisualDemo PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(VisualDemo SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET VisualDemo PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(VisualDemo)

# Enhanced trainer for 90%+ performance
add_executable(EnhancedTrainer
    legacy/enhanced_trainer.cpp
    legacy/network_nn.cpp
)

target_include_directories(EnhancedTrainer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(EnhancedTrainer SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET EnhancedTrainer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(EnhancedTrainer)

# Optimized trainer - builds on proven 96% success
add_executable(OptimizedTrainer
    legacy/optimized_trainer.cpp
    legacy/network_nn.cpp
)

target_include_directories(OptimizedTrainer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(OptimizedTrainer SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET OptimizedTrainer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(OptimizedTrainer)

# Aggressive trainer for high scores (15+ foods per game)
add_executable(AggressiveTrainer
    legacy/aggressive_trainer.cpp
    legacy/network_nn.cpp
)

target_include_directories(AggressiveTrainer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(AggressiveTrainer SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET AggressiveTrainer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(AggressiveTrainer)

# Behavior analyzer to understand AI strategy
add_executable(BehaviorAnalyzer
    legacy/behavior_analyzer.cpp
    legacy/network_nn.cpp
)

target_include_directories(BehaviorAnalyzer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(BehaviorAnalyzer SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET BehaviorAnalyzer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(BehaviorAnalyzer)

# Self-aware trainer for collision avoidance and longer survival
add_executable(SelfAwareTrainer
    legacy/selfaware_trainer.cpp
    legacy/network_nn.cpp
)

target_include_directories(SelfAwareTrainer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(SelfAwareTrainer SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET SelfAwareTrainer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(SelfAwareTrainer)

# Minimal improved trainer - keeps what works, minimal changes
add_executable(MinimalImprovedTrainer
    legacy/minimal_improved_trainer.cpp
    legacy/network_nn.cpp
)

target_include_directories(MinimalImprovedTrainer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(MinimalImprovedTrainer SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET MinimalImprovedTrainer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(MinimalImprovedTrainer)

# Advanced Snake AI Trainer - Next Generation
add_executable(AdvancedTrainer
    legacy/advanced_trainer_main.cpp
    legacy/advanced_snake_network.cpp
)

target_include_directories(AdvancedTrainer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(AdvancedTrainer SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET AdvancedTrainer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(AdvancedTrainer)

# Define CUDA support for advanced trainer
target_compile_definitions(AdvancedTrainer PRIVATE USE_CUDA=1)
target_compile_definitions(AdvancedTrainer PRIVATE TORCH_CUDA=1)

# Windows CUDA linker fix
if(WIN32)
    target_link_options(AdvancedTrainer PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

# Comprehensive evaluator - proper performance metrics
add_executable(ComprehensiveEvaluator
    legacy/comprehensive_evaluator.cpp
    legacy/network_nn.cpp
)

target_include_directories(ComprehensiveEvaluator PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(ComprehensiveEvaluator SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET ComprehensiveEvaluator PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(ComprehensiveEvaluator)

# Define CUDA support for comprehensive evaluator
target_compile_definitions(ComprehensiveEvaluator PRIVATE USE_CUDA=1)
target_compile_definitions(ComprehensiveEvaluator PRIVATE TORCH_CUDA=1)

# Windows CUDA linker fix for comprehensive evaluator
if(WIN32)
    target_link_options(ComprehensiveEvaluator PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

# Extended Optimized trainer for longer training sessions (20K+ episodes)
add_executable(ExtendedOptimizedTrainer
    legacy/extended_optimized_trainer.cpp
    legacy/network_nn.cpp
)

target_include_directories(ExtendedOptimizedTrainer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(ExtendedOptimizedTrainer SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET ExtendedOptimizedTrainer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(ExtendedOptimizedTrainer)

# Define CUDA support for extended trainer
target_compile_definitions(ExtendedOptimizedTrainer PRIVATE USE_CUDA=1)
target_compile_definitions(ExtendedOptimizedTrainer PRIVATE TORCH_CUDA=1)

# Windows CUDA linker fix for extended trainer
if(WIN32)
    target_link_options(ExtendedOptimizedTrainer PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

# Quality trainer - optimizes for game length and score quality
add_executable(QualityTrainer
    legacy/quality_trainer.cpp
    legacy/network_nn.cpp
)

target_include_directories(QualityTrainer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(QualityTrainer SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET QualityTrainer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(QualityTrainer)

# Define CUDA support for quality trainer
target_compile_definitions(QualityTrainer PRIVATE USE_CUDA=1)
target_compile_definitions(QualityTrainer PRIVATE TORCH_CUDA=1)

# Windows CUDA linker fix for quality trainer
if(WIN32)
    target_link_options(QualityTrainer PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

# Resume trainer - continues training from existing model
add_executable(ResumeTrainer
    legacy/resume_trainer.cpp
    legacy/network_nn.cpp
)

target_include_directories(ResumeTrainer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(ResumeTrainer SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET ResumeTrainer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(ResumeTrainer)

# Define CUDA support for resume trainer
target_compile_definitions(ResumeTrainer PRIVATE USE_CUDA=1)
target_compile_definitions(ResumeTrainer PRIVATE TORCH_CUDA=1)

# Windows CUDA linker fix for resume trainer
if(WIN32)
    target_link_options(ResumeTrainer PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

# Research-optimized trainer with academic parameters
add_executable(ResearchOptimizedTrainer
    legacy/research_optimized_trainer.cpp
    legacy/network_nn.cpp
)

target_include_directories(ResearchOptimizedTrainer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(ResearchOptimizedTrainer SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET ResearchOptimizedTrainer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(ResearchOptimizedTrainer)

# Define CUDA support for research trainer
target_compile_definitions(ResearchOptimizedTrainer PRIVATE USE_CUDA=1)
target_compile_definitions(ResearchOptimizedTrainer PRIVATE TORCH_CUDA=1)

# Windows CUDA linker fix for research trainer
if(WIN32)
    target_link_options(ResearchOptimizedTrainer PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

# Winning trainer - optimizes for high scores (15+ foods)
add_executable(WinningTrainer
    legacy/winning_trainer.cpp
    legacy/network_nn.cpp
)

target_include_directories(WinningTrainer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(WinningTrainer SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET WinningTrainer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(WinningTrainer)

# Define CUDA support for winning trainer
target_compile_definitions(WinningTrainer PRIVATE USE_CUDA=1)
target_compile_definitions(WinningTrainer PRIVATE TORCH_CUDA=1)

# Windows CUDA linker fix for winning trainer
if(WIN32)
    target_link_options(WinningTrainer PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

# Simplified winning trainer - proven 8 features with score rewards
add_executable(SimplifiedWinningTrainer
    legacy/simplified_winning_trainer.cpp
    legacy/network_nn.cpp
)

target_include_directories(SimplifiedWinningTrainer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(SimplifiedWinningTrainer SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET SimplifiedWinningTrainer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(SimplifiedWinningTrainer)

# Define CUDA support for simplified winning trainer
target_compile_definitions(SimplifiedWinningTrainer PRIVATE USE_CUDA=1)
target_compile_definitions(SimplifiedWinningTrainer PRIVATE TORCH_CUDA=1)

# Windows CUDA linker fix for simplified winning trainer
if(WIN32)
    target_link_options(SimplifiedWinningTrainer PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

# Model evaluator - comprehensive testing of trained models
add_executable(ModelEvaluator
    legacy/model_evaluator.cpp
    legacy/network_nn.cpp
)

target_include_directories(ModelEvaluator PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(ModelEvaluator SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET ModelEvaluator PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(ModelEvaluator)

# Define CUDA support for model evaluator
target_compile_definitions(ModelEvaluator PRIVATE USE_CUDA=1)
target_compile_definitions(ModelEvaluator PRIVATE TORCH_CUDA=1)

# Windows CUDA linker fix for model evaluator
if(WIN32)
    target_link_options(ModelEvaluator PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

# Hybrid AI - Neural network + pathfinding + space management
add_executable(HybridAI
    legacy/hybrid_ai.cpp
    legacy/network_nn.cpp
)

target_include_directories(HybridAI PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(HybridAI SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET HybridAI PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(HybridAI)

# Define CUDA support for hybrid AI
target_compile_definitions(HybridAI PRIVATE USE_CUDA=1)
target_compile_definitions(HybridAI PRIVATE TORCH_CUDA=1)

# Windows CUDA linker fix for hybrid AI
if(WIN32)
    target_link_options(HybridAI PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

# Visual hybrid demo - watch AI play in real-time
add_executable(VisualHybridDemo
    legacy/visual_hybrid_demo.cpp
    legacy/network_nn.cpp
)

target_include_directories(VisualHybridDemo PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(VisualHybridDemo SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET VisualHybridDemo PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(VisualHybridDemo)

# Define CUDA support for visual demo
target_compile_definitions(VisualHybridDemo PRIVATE USE_CUDA=1)
target_compile_definitions(VisualHybridDemo PRIVATE TORCH_CUDA=1)

# Windows CUDA linker fix for visual demo
if(WIN32)
    target_link_options(VisualHybridDemo PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

# Safe Snake AI - guaranteed collision avoidance
add_executable(SafeSnakeAI
    legacy/safe_snake_ai.cpp
    legacy/network_nn.cpp
)

target_include_directories(SafeSnakeAI PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(SafeSnakeAI SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET SafeSnakeAI PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(SafeSnakeAI)

# Define CUDA support for safe AI
target_compile_definitions(SafeSnakeAI PRIVATE USE_CUDA=1)
target_compile_definitions(SafeSnakeAI PRIVATE TORCH_CUDA=1)

# Windows CUDA linker fix for safe AI
if(WIN32)
    target_link_options(SafeSnakeAI PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

# Universal Safe AI - loads any compatible model architecture
add_executable(UniversalSafeAI
    legacy/universal_safe_ai.cpp
    legacy/network_nn.cpp
)

target_include_directories(UniversalSafeAI PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(UniversalSafeAI SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET UniversalSafeAI PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(UniversalSafeAI)

# Define CUDA support for universal AI
target_compile_definitions(UniversalSafeAI PRIVATE USE_CUDA=1)
target_compile_definitions(UniversalSafeAI PRIVATE TORCH_CUDA=1)

# Windows CUDA linker fix for universal AI
if(WIN32)
    target_link_options(UniversalSafeAI PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

# Smart Safe AI - trap avoidance with flood fill
add_executable(SmartSafeAI
    legacy/smart_safe_ai.cpp
    legacy/network_nn.cpp
)

target_include_directories(SmartSafeAI PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(SmartSafeAI SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET SmartSafeAI PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(SmartSafeAI)

# Define CUDA support for smart AI
target_compile_definitions(SmartSafeAI PRIVATE USE_CUDA=1)
target_compile_definitions(SmartSafeAI PRIVATE TORCH_CUDA=1)

# Windows CUDA linker fix for smart AI
if(WIN32)
    target_link_options(SmartSafeAI PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

# Simple Snake AI with Fixed Collision Detection
add_executable(SimpleSnakeAIFixed
    legacy/simple_snake_ai_fixed.cpp
    legacy/network_nn.cpp
)

target_include_directories(SimpleSnakeAIFixed PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(SimpleSnakeAIFixed SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET SimpleSnakeAIFixed PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(SimpleSnakeAIFixed)

# Define CUDA support for fixed AI
target_compile_definitions(SimpleSnakeAIFixed PRIVATE USE_CUDA=1)
target_compile_definitions(SimpleSnakeAIFixed PRIVATE TORCH_CUDA=1)

# Windows CUDA linker fix for fixed AI
if(WIN32)
    target_link_options(SimpleSnakeAIFixed PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

# Tests. None of these link LibTorch or raylib, so they build and run without
# the CUDA toolchain, and each returns non-zero on failure. Run them with
# `ctest -C Debug` from the build directory, or build and run a target directly.
enable_testing()

# /W4 with warnings as errors, for the targets measured clean under it.
#
# Not applied to the LibTorch or raylib targets - their headers warn - nor to
# Benchmark and PerfectSnakeAI, because snake_logic.cpp has two narrowing
# conversions (C4244 at :146, C4267 at :181), nor to anything in legacy/.
# /external:W0 keeps the standard library's own warnings out.
#
# Core Guidelines analysis is a separate pass, not part of the build: see
# docs/analyze.ps1. Header self-containment is a third: docs/check_headers.ps1.
function(enable_strict_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE
            /W4 /WX /permissive- /external:anglebrackets /external:W0)
    endif()
endfunction()

# Cycle validity test - no LibTorch, no raylib, runs anywhere
add_executable(CycleTest
    src/cycle_test.cpp
    src/hamiltonian_cycle.cpp
)

set_property(TARGET CycleTest PROPERTY CXX_STANDARD 20)
enable_strict_warnings(CycleTest)
add_test(NAME cycle_is_hamiltonian COMMAND CycleTest)

# Training environment properties - the simulator the learner will be trained
# against, checked before anything is trained against it.
add_executable(EnvTest
    src/snake_env_test.cpp
    src/snake_env.cpp
    src/hamiltonian_cycle.cpp
)

set_property(TARGET EnvTest PROPERTY CXX_STANDARD 20)
enable_strict_warnings(EnvTest)
add_test(NAME snake_env_properties COMMAND EnvTest)

# Batched environment - the layer everything the learner concludes rests on.
add_executable(VectorEnvTest
    src/vector_env_test.cpp
    src/vector_env.cpp
    src/snake_env.cpp
)

set_property(TARGET VectorEnvTest PROPERTY CXX_STANDARD 20)
enable_strict_warnings(VectorEnvTest)
add_test(NAME vector_env_properties COMMAND VectorEnvTest)

# Search correctness, checked against hand-written evaluators rather than a
# network, so a failure here is the search's and not the model's. No LibTorch.
add_executable(SearchTest
    src/mcts_test.cpp
    src/mcts.cpp
    src/snake_env.cpp
)

set_property(TARGET SearchTest PROPERTY CXX_STANDARD 20)
enable_strict_warnings(SearchTest)
add_test(NAME search_properties COMMAND SearchTest)

# Network properties, including that its weights are board-size independent,
# which is what makes the curriculum possible. Needs LibTorch.
add_executable(NetworkTest
    src/az_network_test.cpp
    src/az_network.cpp
    src/snake_env.cpp
)

target_include_directories(NetworkTest PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(NetworkTest ${TORCH_LIBRARIES})
set_property(TARGET NetworkTest PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(NetworkTest)

if(WIN32)
    target_link_options(NetworkTest PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

add_test(NAME network_properties COMMAND NetworkTest)

# Training and evaluation seed ranges are disjoint, and the guard that enforces
# it actually fires. They overlapped by 86 percent until 2026-08-08, which
# invalidated every held-out number measured before then. No LibTorch.
add_executable(SeedPolicyTest
    src/seed_policy_test.cpp
)

set_property(TARGET SeedPolicyTest PROPERTY CXX_STANDARD 20)
enable_strict_warnings(SeedPolicyTest)
add_test(NAME seed_policy COMMAND SeedPolicyTest)

# What the search sees of chance: every simulation copies the root environment
# and its random generator, so one action sequence always yields one apple. The
# paper branches over every empty cell instead. No LibTorch.
add_executable(ChanceNodeTest
    src/chance_node_test.cpp
    src/snake_env.cpp
)

set_property(TARGET ChanceNodeTest PROPERTY CXX_STANDARD 20)
enable_strict_warnings(ChanceNodeTest)
add_test(NAME chance_nodes COMMAND ChanceNodeTest)

# Whether a one-cell change survives the 4x4 pooling and reaches the heads.
# The pooling is what makes the weights board-size independent; this measures
# what it costs. Optional argument: a checkpoint, which enables the property
# that a trained policy's preferred action depends on where the food is.
add_executable(PoolingProbeTest
    src/pooling_probe_test.cpp
    src/az_network.cpp
    src/snake_env.cpp
)

target_include_directories(PoolingProbeTest PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(PoolingProbeTest ${TORCH_LIBRARIES})
set_property(TARGET PoolingProbeTest PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(PoolingProbeTest)

if(WIN32)
    target_link_options(PoolingProbeTest PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

add_test(NAME pooling_probe COMMAND PoolingProbeTest)

# Self-play record generation - the value targets no later stage can check.
add_executable(SelfPlayTest
    src/selfplay_test.cpp
    src/selfplay.cpp
    src/mcts.cpp
    src/snake_env.cpp
)

set_property(TARGET SelfPlayTest PROPERTY CXX_STANDARD 20)
enable_strict_warnings(SelfPlayTest)
add_test(NAME selfplay_properties COMMAND SelfPlayTest)

# The trainer's settings and the two strings it prints. Extracted from
# az_trainer.cpp, which was one main() and therefore untestable: it accepted
# "--games 0" and divided by the game count eight hundred seconds later, and
# warned about a mistyped flag on stderr before training with the default. No
# LibTorch, so assertions here are reachable in a debug build - the ones in any
# Torch-linked file are not.
add_executable(TrainerOptionsTest
    src/trainer_options_test.cpp
    src/trainer_options.cpp
    src/az_parameters.cpp
)

set_property(TARGET TrainerOptionsTest PROPERTY CXX_STANDARD 20)
enable_strict_warnings(TrainerOptionsTest)
add_test(NAME trainer_options COMMAND TrainerOptionsTest)

# The paper's hyperparameters and the step limit derived from the board area.
# No LibTorch.
add_executable(ParametersTest
    src/az_parameters_test.cpp
    src/az_parameters.cpp
)

set_property(TARGET ParametersTest PROPERTY CXX_STANDARD 20)
enable_strict_warnings(ParametersTest)
add_test(NAME az_parameters COMMAND ParametersTest)

# Command-line number parsing, shared by the trainer, the evaluator and the
# visual. No LibTorch, so its assertions and its throws are reachable in Debug.
add_executable(FlagParserTest
    src/flag_parser_test.cpp
    src/flag_parser.cpp
)

set_property(TARGET FlagParserTest PROPERTY CXX_STANDARD 20)
enable_strict_warnings(FlagParserTest)
add_test(NAME flag_parser COMMAND FlagParserTest)

# The evaluator's settings. Extracted from az_evaluate.cpp, whose parser warned
# about an unknown flag and then scored the run with the default, and reached
# std::stoi for every number. Every figure this project quotes comes out of that
# program. No LibTorch, so its assertions and its throws are reachable in Debug.
add_executable(EvalOptionsTest
    src/eval_options_test.cpp
    src/eval_options.cpp
    src/az_parameters.cpp
    src/flag_parser.cpp
)

set_property(TARGET EvalOptionsTest PROPERTY CXX_STANDARD 20)
enable_strict_warnings(EvalOptionsTest)
add_test(NAME eval_options COMMAND EvalOptionsTest)

# AlphaZero training loop.
add_executable(AlphaZeroTrainer
    src/az_trainer.cpp
    src/trainer_options.cpp
    src/az_parameters.cpp
    src/selfplay.cpp
    src/mcts.cpp
    src/network_evaluator.cpp
    src/az_network.cpp
    src/snake_env.cpp
)

target_include_directories(AlphaZeroTrainer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(AlphaZeroTrainer ${TORCH_LIBRARIES})
set_property(TARGET AlphaZeroTrainer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(AlphaZeroTrainer)

target_compile_definitions(AlphaZeroTrainer PRIVATE USE_CUDA=1)
target_compile_definitions(AlphaZeroTrainer PRIVATE TORCH_CUDA=1)

if(WIN32)
    target_link_options(AlphaZeroTrainer PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

# Scores a saved network on held-out seeds: greedy, no root noise, win rate.
add_executable(AlphaZeroEvaluate
    src/az_evaluate.cpp
    src/eval_options.cpp
    src/az_parameters.cpp
    src/flag_parser.cpp
    src/mcts.cpp
    src/network_evaluator.cpp
    src/az_network.cpp
    src/snake_env.cpp
)

target_include_directories(AlphaZeroEvaluate PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(AlphaZeroEvaluate ${TORCH_LIBRARIES})
set_property(TARGET AlphaZeroEvaluate PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(AlphaZeroEvaluate)

target_compile_definitions(AlphaZeroEvaluate PRIVATE USE_CUDA=1)
target_compile_definitions(AlphaZeroEvaluate PRIVATE TORCH_CUDA=1)

if(WIN32)
    target_link_options(AlphaZeroEvaluate PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

# Watch a trained network play, under the same conditions it is scored under.
add_executable(AlphaZeroVisual
    src/az_visual.cpp
    src/board_render.cpp
    src/mcts.cpp
    src/network_evaluator.cpp
    src/az_network.cpp
    src/snake_env.cpp
)

target_include_directories(AlphaZeroVisual PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(AlphaZeroVisual raylib ${TORCH_LIBRARIES})
set_property(TARGET AlphaZeroVisual PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(AlphaZeroVisual)

target_compile_definitions(AlphaZeroVisual PRIVATE USE_CUDA=1)
target_compile_definitions(AlphaZeroVisual PRIVATE TORCH_CUDA=1)

if(WIN32)
    target_link_options(AlphaZeroVisual PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

# Throughput measurement - simulator steps and network evaluations per second.
# The only target in this group that needs LibTorch.
add_executable(ThroughputBenchmark
    src/throughput_benchmark.cpp
    src/az_network.cpp
    src/vector_env.cpp
    src/snake_env.cpp
    src/hamiltonian_cycle.cpp
)

target_include_directories(ThroughputBenchmark PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(ThroughputBenchmark ${TORCH_LIBRARIES})
set_property(TARGET ThroughputBenchmark PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(ThroughputBenchmark)

target_compile_definitions(ThroughputBenchmark PRIVATE USE_CUDA=1)
target_compile_definitions(ThroughputBenchmark PRIVATE TORCH_CUDA=1)

if(WIN32)
    target_link_options(ThroughputBenchmark PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

# Headless benchmark - win rate over seeded games. No LibTorch, no raylib, so
# it compiles snake_logic.cpp directly rather than linking SnakeGameLib.
add_executable(Benchmark
    src/benchmark.cpp
    src/cycle_agent.cpp
    src/hamiltonian_cycle.cpp
    src/snake_logic.cpp
)

set_property(TARGET Benchmark PROPERTY CXX_STANDARD 20)

# Perfect Snake AI - complete game winner with Hamiltonian Cycle
add_executable(PerfectSnakeAI
    src/perfect_snake_ai.cpp
    src/board_render.cpp
    src/cycle_agent.cpp
    src/hamiltonian_cycle.cpp
    legacy/network_nn.cpp
)

target_include_directories(PerfectSnakeAI PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(PerfectSnakeAI SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET PerfectSnakeAI PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(PerfectSnakeAI)

# Define CUDA support for perfect AI
target_compile_definitions(PerfectSnakeAI PRIVATE USE_CUDA=1)
target_compile_definitions(PerfectSnakeAI PRIVATE TORCH_CUDA=1)

# Windows CUDA linker fix for perfect AI
if(WIN32)
    target_link_options(PerfectSnakeAI PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()
