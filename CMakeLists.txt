cmake_minimum_required(VERSION 3.20)

# 1. 项目名称, 版本及编程语言
project(edgeflow VERSION 0.1.0 LANGUAGES CXX)

# 2. 强制使用标准 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 3. 功能开关
option(EDGEFLOW_WITH_RKNN "Enable RKNN backend" ON)
option(EDGEFLOW_WITH_RGA "Enable RGA preprocessing" ON)

# 4. 获取当前 Git Commit
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE EDGEFLOW_GIT_SHA
    RESULT_VARIABLE EDGEFLOW_GIT_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if (NOT EDGEFLOW_GIT_RESULT EQUAL 0 OR EDGEFLOW_GIT_SHA STREQUAL "")
    set(EDGEFLOW_GIT_SHA "unknown")
endif()

# 5. 定义核心接口库
# 当前还没有核心源文件，先使用 INTERFACE
# INTERFACE 是“纯头文件库”, 不编译任何 .cpp
# 编译源文件
add_library(
    edgeflow_core STATIC
    src/core/status.cpp
    src/core/system_probe.cpp
)

# 头文件
target_include_directories(
    edgeflow_core PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)

# 6. 将 CMake 配置传入 C++ 编译器
target_compile_definitions(
    edgeflow_core PUBLIC
    "EDGEFLOW_VERSION=\"${PROJECT_VERSION}\""
    "EDGEFLOW_GIT_COMMIT=\"${EDGEFLOW_GIT_SHA}\""
    "EDGEFLOW_BUILD_TYPE=\"$<IF:$<BOOL:$<CONFIG>>,$<CONFIG>,Unknown>\""
    "EDGEFLOW_TARGET_ARCH=\"${CMAKE_SYSTEM_PROCESSOR}\""
    "EDGEFLOW_WITH_RKNN=$<BOOL:${EDGEFLOW_WITH_RKNN}>"
    "EDGEFLOW_WITH_RGA=$<BOOL:${EDGEFLOW_WITH_RGA}>"
)

# 7. 开启编译警告
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
        edgeflow_core PUBLIC
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

# 8. 定义命令行程序
add_executable(
    edgeflow
    apps/edgeflow.cpp
)

target_link_libraries(
    edgeflow PRIVATE
    edgeflow_core
)

# 9. 开启自动化测试
enable_testing()

add_test(
    NAME edgeflow_version
    COMMAND edgeflow version
)

add_executable(
    edgeflow_unit_tests
    tests/unit/test_core.cpp
)
target_link_libraries(
    edgeflow_unit_tests PRIVATE
    edgeflow_core
)
add_test(
    NAME edgeflow_unit_tests
    COMMAND edgeflow_unit_tests
)