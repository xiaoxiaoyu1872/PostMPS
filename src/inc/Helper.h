#pragma once

#include <iostream>
#include <math.h>
#include <vector_types.h> //cuda
#include <vector_functions.h> //cuda
#include <cuda_runtime_api.h>

#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_functions.h>

typedef unsigned int uint;

inline __host__ std::ostream &operator<<(std::ostream &os, const float3 &a)
{ 
    os <<a.x <<"  " << a.y << "  " << a.z;
    return os;            
}

inline __host__ std::ostream &operator<<(std::ostream &os, const uint3 &a)
{ 
    os <<a.x <<"  " << a.y << "  " << a.z;
    return os;            
}




