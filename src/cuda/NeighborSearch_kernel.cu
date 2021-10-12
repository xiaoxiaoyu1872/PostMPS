#include "GPUmemory.h"
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
// #include <myCuda.cu>
__constant__  GridParams  dGridParams;

__global__
void boundGridBuilding(
	BoundParticle* particles,
	uint numParticles,
	BoundGrid *boundGrid,
	SpatialGrid* spatialGrid)
{
	uint index = getthreadIdGlobal();
	if (index >= numParticles)
		return;

	float3 curPos = particles[index].pos;
	uint3 Index3Dsp = getIndex3D(curPos, dGridParams.minPos, dGridParams.spGridSize);

	uint3 spatialGridRes = dGridParams.spresolution;

	// uint hashValue = index3DTo1D(make_uint3(Index3Dsp.x, Index3Dsp.y, Index3Dsp.z), dGridParams.spresolution);

	uint3 lower = make_uint3(Index3Dsp.x - 1, Index3Dsp.y - 1, Index3Dsp.z - 1);
	uint3 upper = make_uint3(Index3Dsp.x + 1, Index3Dsp.y + 1, Index3Dsp.z + 1);
	lower = clamp(lower, make_uint3(0, 0, 0), make_uint3(spatialGridRes.x - 1, spatialGridRes.y - 1, spatialGridRes.z - 1));
	upper = clamp(upper, make_uint3(0, 0, 0), make_uint3(spatialGridRes.x - 1, spatialGridRes.y - 1, spatialGridRes.z - 1));

#pragma unroll 3
	for (uint z = lower.z; z <= upper.z; ++z)
	{
#pragma unroll 3
		for (uint y = lower.y; y <= upper.y; ++y)
		{
#pragma unroll 3
			for (uint x = lower.x; x <= upper.x; ++x)
			{
				uint3 neighbor = make_uint3(x, y, z); 
				uint index = index3DTo1D(neighbor, spatialGridRes);  				
				boundGrid[index].bound = 1;
				spatialGrid[index].classify = 2;
			}
		}
	}
}


__global__
void calculateHash(
    uint* particleHash, 
	FluidParticle* particles,
	uint numParticles,
	SpatialGrid* spatialGrid)
{
	uint index = getthreadIdGlobal();
	if (index >= numParticles)
		return;

	float3 curPos = particles[index].pos;

	uint3 gridPos = getIndex3D(curPos, dGridParams.minPos, dGridParams.spGridSize);

	uint hashValue = index3DTo1D(make_uint3(gridPos.x, gridPos.y, gridPos.z), dGridParams.spresolution);
	
    particleHash[index] = hashValue;

	spatialGrid[hashValue].fluid = 1;
	spatialGrid[hashValue].inner = 1;

	spatialGrid[hashValue].classify = 2; //surface and inner fluid
}

__global__
void calCellRange(
	IndexRange* particlesIndexRange, 
	uint numParticles, 
	uint *particleHash,
	FluidParticle* particles) 
{
    uint index = getthreadIdGlobal();
    if (index >= numParticles)
        return;

	extern __shared__ unsigned int sharedHash[];
	
	uint hashValue;

    hashValue = particleHash[index];
    sharedHash[threadIdx.x + 1] = hashValue;

    if (index > 0 && threadIdx.x == 0)
        sharedHash[0] = particleHash[index - 1];

	__syncthreads();  

	if (index == 0 || hashValue != sharedHash[threadIdx.x])
	{
		particlesIndexRange[hashValue].start = index;
		if (index > 0)
			particlesIndexRange[sharedHash[threadIdx.x]].end = index;
	}

	if (index == numParticles - 1)
		particlesIndexRange[hashValue].end = index + 1;
}


// __global__
// void findCellRangeKernel(
// 	IndexRange* particlesIndexRange, 
// 	uint numParticles, 
// 	uint *particleHash,
// 	FluidParticle* particles) 
// {
//     uint index = getthreadIdGlobal();
//     if (index >= numParticles)
//         return;
	
// 	uint hashValue = particleHash[index];;

// 	uint hashValue_before = 0;
//     if (index > 0)
//         hashValue_before = particleHash[index - 1];

// 	if (index == 0 || hashValue != hashValue_before)
// 	{
// 		particlesIndexRange[hashValue].start = index;
// 		if (index > 0)
// 			particlesIndexRange[hashValue_before].end = index;
// 	}

// 	if (index == numParticles - 1)
// 		particlesIndexRange[hashValue].end = index + 1;

// }
