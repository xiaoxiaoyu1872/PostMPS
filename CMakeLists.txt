cmake_minimum_required(VERSION 3.0)

project(Visualization)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

find_package(VTK COMPONENTS vtkFiltersModeling vtkIOGeometry REQUIRED)
include(${VTK_USE_FILE})

IF(VTK_FOUND)
  message("found VTK. Version:" ${VTK_VERSION}. VTK_DIR: ${VTK_DIR})
ELSE(VTK_FOUND)
  MESSAGE(FATAL_ERROR
    "Cannot build the executable without VTK. Please set the VTK_DIR")
ENDIF(VTK_FOUND)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS -O2; -G; -g;-std=c++14)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -fopenmp-simd")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${CUDA_INCLUDE_DIRS} /usr/local/cuda/samples/common/inc) 

add_subdirectory(src)
