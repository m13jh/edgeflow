cmake_minimum_required(VERSION 3.20)

# 1. 项目配置
project(edgeflow VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 功能开关
option(EDGEFLOW_WITH_RKNN "Enable RKNN backend" ON)
option(EDGEFLOW_WITH_RGA "Enable RGA preprocessing" ON)

# 3. 查找公共依赖
find_package(JPEG REQUIRED)

# 4. 获取当前 Git Commit
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE EDGEFLOW_GIT_SHA
    RESULT_VARIABLE EDGEFLOW_GIT_RESULT # 进程的返回结果
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if (NOT EDGEFLOW_GIT_RESULT EQUAL 0 OR EDGEFLOW_GIT_SHA STREQUAL "")
    set(EDGEFLOW_GIT_SHA "unknown")
endif()

# 5. 创建核心静态库
add_library(edgeflow_core STATIC
    src/core/status.cpp
    src/core/system_probe.cpp
    src/core/image_io.cpp
    src/postprocess/classification.cpp
    src/core/latency_stats.cpp
)

target_include_directories(edgeflow_core
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)

# JPEG 是 Core 的内部实现依赖
target_link_libraries(edgeflow_core
    PRIVATE
        JPEG::JPEG
)

# 6. 编译宏
target_compile_definitions(edgeflow_core
    PUBLIC
        "EDGEFLOW_VERSION=\"${PROJECT_VERSION}\""
        "EDGEFLOW_GIT_COMMIT=\"${EDGEFLOW_GIT_SHA}\""
        # 项目构建方式, debug 或 release
        "EDGEFLOW_BUILD_TYPE=\"$<IF:$<BOOL:$<CONFIG>>,$<CONFIG>,Unknown>\""
        # 处理器架构
        "EDGEFLOW_TARGET_ARCH=\"${CMAKE_SYSTEM_PROCESSOR}\""
        "EDGEFLOW_WITH_RKNN=$<BOOL:${EDGEFLOW_WITH_RKNN}>"
        "EDGEFLOW_WITH_RGA=$<BOOL:${EDGEFLOW_WITH_RGA}>"
)

# 7. 编译警告
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(edgeflow_core
        PUBLIC
            -Wall
            -Wextra
            -Wpedantic
    )
endif()

# 8. RKNN 后端
if (EDGEFLOW_WITH_RKNN)
    find_library(RKNN_RUNTIME_LIB
        NAMES rknnrt
        PATHS /lib /usr/lib
    )

    if (NOT RKNN_RUNTIME_LIB)
        message(FATAL_ERROR
            "RKNN runtime library librknnrt.so not found"
        )
    endif()

    add_library(edgeflow_rknn STATIC
        src/backend/rknn/rknn_context.cpp
    )

    # SDK 头文件只对 RKNN 目标可见
    target_include_directories(edgeflow_rknn
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknn/include"
    )

    target_link_libraries(edgeflow_rknn
        PUBLIC
            edgeflow_core
        PRIVATE
            ${RKNN_RUNTIME_LIB}
    )

    message(STATUS "RKNN runtime: ${RKNN_RUNTIME_LIB}")
endif()

# 9. 命令行程序
add_executable(edgeflow
    apps/edgeflow.cpp
)

target_link_libraries(edgeflow
    PRIVATE edgeflow_core
)

# 仅 RKNN=ON 时链接 RKNN 后端
if (EDGEFLOW_WITH_RKNN)
    target_link_libraries(edgeflow
        PRIVATE edgeflow_rknn
    )
endif()

# 10. 自动化测试
enable_testing()

add_test(
    NAME edgeflow_version
    COMMAND edgeflow version
)

add_executable(edgeflow_unit_tests
    tests/unit/test_core.cpp
)

target_link_libraries(edgeflow_unit_tests
    PRIVATE edgeflow_core
)

add_test(
    NAME edgeflow_unit_tests
    COMMAND edgeflow_unit_tests
)

if (EDGEFLOW_WITH_RKNN)
    add_executable(edgeflow_rknn_tests
        tests/hardware/test_rknn_context.cpp
    )

    target_link_libraries(edgeflow_rknn_tests
        PRIVATE edgeflow_rknn
    )

    add_test(
        NAME edgeflow_rknn_tests
        COMMAND edgeflow_rknn_tests
    )
endif()