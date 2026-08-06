cmake_minimum_required(VERSION 3.18)
project(SnakeNN CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
    src/snake_trainer.cpp
    src/network_nn.cpp
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
    src/minimal_test.cpp
)

target_include_directories(NeuralNetworkTest PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(NeuralNetworkTest ${TORCH_LIBRARIES})
set_property(TARGET NeuralNetworkTest PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(NeuralNetworkTest)

add_executable(SnakeGame
    src/snake_game.cpp
)

target_include_directories(SnakeGame PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(SnakeGame SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET SnakeGame PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(SnakeGame)

add_executable(SnakeTest
    src/snake_test.cpp
    src/network_nn.cpp
)

target_include_directories(SnakeTest PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(SnakeTest SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET SnakeTest PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(SnakeTest)

# Diagnostic test executable
add_executable(DiagnosticTest
    src/diagnostic_test.cpp
    src/network_nn.cpp
)

target_include_directories(DiagnosticTest PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(DiagnosticTest SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET DiagnosticTest PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(DiagnosticTest)

# Simple agent test (no neural network)
add_executable(SimpleAgentTest
    src/simple_agent_test.cpp
)

target_link_libraries(SimpleAgentTest SnakeGameLib)
set_property(TARGET SimpleAgentTest PROPERTY CXX_STANDARD 20)

# Simple neural agent test
add_executable(SimpleNeuralTest
    src/simple_neural_test.cpp
    src/network_nn.cpp
)

target_include_directories(SimpleNeuralTest PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(SimpleNeuralTest SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET SimpleNeuralTest PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(SimpleNeuralTest)

# Fixed trainer with all improvements
add_executable(FixedTrainer
    src/fixed_trainer_clean.cpp
    src/network_nn.cpp
)

target_include_directories(FixedTrainer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(FixedTrainer SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET FixedTrainer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(FixedTrainer)

# Simple fixed trainer
add_executable(SimpleFixed
    src/simple_fixed.cpp
    src/network_nn.cpp
)

target_include_directories(SimpleFixed PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(SimpleFixed SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET SimpleFixed PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(SimpleFixed)

# Model tester
add_executable(ModelTester
    src/model_tester.cpp
    src/network_nn.cpp
)

target_include_directories(ModelTester PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(ModelTester SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET ModelTester PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(ModelTester)

# Visual demo
add_executable(VisualDemo
    src/visual_demo.cpp
    src/network_nn.cpp
)

target_include_directories(VisualDemo PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(VisualDemo SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET VisualDemo PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(VisualDemo)

# Enhanced trainer for 90%+ performance
add_executable(EnhancedTrainer
    src/enhanced_trainer.cpp
    src/network_nn.cpp
)

target_include_directories(EnhancedTrainer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(EnhancedTrainer SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET EnhancedTrainer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(EnhancedTrainer)

# Optimized trainer - builds on proven 96% success
add_executable(OptimizedTrainer
    src/optimized_trainer.cpp
    src/network_nn.cpp
)

target_include_directories(OptimizedTrainer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(OptimizedTrainer SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET OptimizedTrainer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(OptimizedTrainer)

# Aggressive trainer for high scores (15+ foods per game)
add_executable(AggressiveTrainer
    src/aggressive_trainer.cpp
    src/network_nn.cpp
)

target_include_directories(AggressiveTrainer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(AggressiveTrainer SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET AggressiveTrainer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(AggressiveTrainer)

# Behavior analyzer to understand AI strategy
add_executable(BehaviorAnalyzer
    src/behavior_analyzer.cpp
    src/network_nn.cpp
)

target_include_directories(BehaviorAnalyzer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(BehaviorAnalyzer SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET BehaviorAnalyzer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(BehaviorAnalyzer)

# Self-aware trainer for collision avoidance and longer survival
add_executable(SelfAwareTrainer
    src/selfaware_trainer.cpp
    src/network_nn.cpp
)

target_include_directories(SelfAwareTrainer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(SelfAwareTrainer SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET SelfAwareTrainer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(SelfAwareTrainer)

# Minimal improved trainer - keeps what works, minimal changes
add_executable(MinimalImprovedTrainer
    src/minimal_improved_trainer.cpp
    src/network_nn.cpp
)

target_include_directories(MinimalImprovedTrainer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(MinimalImprovedTrainer SnakeGameLib ${TORCH_LIBRARIES})
set_property(TARGET MinimalImprovedTrainer PROPERTY CXX_STANDARD 20)
copy_pytorch_dlls(MinimalImprovedTrainer)

# Advanced Snake AI Trainer - Next Generation
add_executable(AdvancedTrainer
    src/advanced_trainer_main.cpp
    src/advanced_snake_network.cpp
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
    src/comprehensive_evaluator.cpp
    src/network_nn.cpp
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
    src/extended_optimized_trainer.cpp
    src/network_nn.cpp
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
    src/quality_trainer.cpp
    src/network_nn.cpp
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
    src/resume_trainer.cpp
    src/network_nn.cpp
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
    src/research_optimized_trainer.cpp
    src/network_nn.cpp
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
    src/winning_trainer.cpp
    src/network_nn.cpp
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
    src/simplified_winning_trainer.cpp
    src/network_nn.cpp
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
    src/model_evaluator.cpp
    src/network_nn.cpp
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
    src/hybrid_ai.cpp
    src/network_nn.cpp
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
    src/visual_hybrid_demo.cpp
    src/network_nn.cpp
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
    src/safe_snake_ai.cpp
    src/network_nn.cpp
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
    src/universal_safe_ai.cpp
    src/network_nn.cpp
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
    src/smart_safe_ai.cpp
    src/network_nn.cpp
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
    src/simple_snake_ai_fixed.cpp
    src/network_nn.cpp
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

# Perfect Snake AI - complete game winner with Hamiltonian Cycle
add_executable(PerfectSnakeAI
    src/perfect_snake_ai.cpp
    src/hamiltonian_cycle.cpp
    src/network_nn.cpp
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
