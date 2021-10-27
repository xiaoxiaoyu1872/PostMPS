file(GLOB cpp_SOURCES  *.cpp  ./inc/marchingcube/*.cpp)
file(GLOB cuda_SOURCES ./cuda/*.cu)

add_executable(VisGPU
    ${cpp_SOURCES}
    ${cuda_SOURCES}
    inc/inireader/ini.c
    inc/inireader/INIReader.cpp
    inc/tinyply/tinyply.cpp
)

include_directories(VisGPU
  PUBLIC
    .
    ./cuda
    ./inc
    ./inc/inireader
    ./inc/marchingcube
    ./inc/tinyply/
    ${CUDA_SAMPLES_INCLUDE}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(VisGPU
  PUBLIC
    -lstdc++fs
    VTK::FiltersModeling
    VTK::IOGeometry
)

vtk_module_autoinit(
  TARGETS VisGPU
  MODULES VTK::FiltersModeling VTK::IOGeometry
)

target_compile_features(VisGPU PUBLIC cxx_std_14 cuda_std_14)

set_target_properties(VisGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(EXECUTABLE_OUTPUT_PATH ../../bin)

set_target_properties(VisGPU PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}")
