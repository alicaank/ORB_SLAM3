cmake_minimum_required(VERSION 3.10)
project(DepthInferenceOverlappingDemo)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/../include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# Add executable
add_executable(overlapping_demo overlapping_demo.cpp)

# Link libraries
target_link_libraries(overlapping_demo
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    cudart
    cublas
    curand
    cudnn
)

# Set CUDA properties
set_target_properties(overlapping_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Compiler flags
target_compile_options(overlapping_demo PRIVATE
    ${OpenCV_COMPILE_OPTIONS}
    ${CUDA_NVCC_FLAGS}
)

# Add definition for CUDA
target_compile_definitions(overlapping_demo PRIVATE
    CUDA_ENABLED
)

# Copy the main library if it exists
if(EXISTS ${CMAKE_SOURCE_DIR}/../lib/libDepthInference.a)
    add_custom_command(TARGET overlapping_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/../lib/libDepthInference.a
        $<TARGET_FILE_DIR:overlapping_demo>/
    )
endif()

# Install target
install(TARGETS overlapping_demo
    RUNTIME DESTINATION bin
) 