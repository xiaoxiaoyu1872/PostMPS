cmake_minimum_required(VERSION 3.18)

project(Visualization LANGUAGES CUDA CXX C)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(GLEW REQUIRED)
find_package(VTK REQUIRED
  COMPONENTS
    FiltersModeling
    IOGeometry
)

if(VTK_FOUND)
  message("found VTK. Version:" ${VTK_VERSION}. VTK_DIR: ${VTK_DIR})
else()
  message(FATAL_ERROR
    "Cannot build the executable without VTK. Please set the VTK_DIR")
endif()

find_path(CUDA_SAMPLES_INCLUDE
  helper_cuda.h
  PATHS
    "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v11.4/common/inc/"
    /usr/local/cuda/samples/common/inc
  REQUIRED
)

add_subdirectory(src)
