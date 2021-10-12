#include "NeighborSearch.h"
#include "NeighborSearch_kernel.cu"
#include "Thrust.cuh"

NeighborSearch::NeighborSearch(GPUmemory *_gMemory, Params *_params)
{
    gMemory = _gMemory;
    params = _params;
    constantMemCopy_Grid();
}

void NeighborSearch::constantMemCopy_Grid()
{
	// checkCudaErrors(cudaMemcpyToSymbol(dGridParams, &params->mGridParams, sizeof(GridParams)));
}

void NeighborSearch::BoundGridBuilding()
{
    checkCudaErrors(cudaMemcpyToSymbol(dGridParams, &params->mGridParams, sizeof(GridParams)));
    uint NumBoundParticles = gMemory->NumBoundParticles;

    dim3 gridDim, blockDim;
    calcGridDimBlockDim(NumBoundParticles,gridDim, blockDim);

    boundGridBuilding<<< gridDim, blockDim, 0, 0>>>
    (gMemory->dBoundParticle, 
    NumBoundParticles, 
    gMemory->dBoundGrid,
    gMemory->dSpatialGrid);
	getLastCudaError("boundGridBuilding");

    // gMemory->Memfree_bound();
    cudaDeviceSynchronize();
}  


void NeighborSearch::SpatialGridBuilding()
{
    checkCudaErrors(cudaMemcpyToSymbol(dGridParams, &params->mGridParams, sizeof(GridParams)));
    uint NumParticles = gMemory->NumParticles;
    uint *dParticleHash;
    
	checkCudaErrors(cudaMalloc((void**)&dParticleHash, NumParticles * sizeof(uint)));
   
    dim3 gridDim, blockDim;
    calcGridDimBlockDim(NumParticles,gridDim, blockDim);

    calculateHash<<< gridDim, blockDim, 0, 0>>>(dParticleHash, gMemory->dFluidParticle, NumParticles, gMemory->dSpatialGrid);
	getLastCudaError("calculateHash");
    cudaDeviceSynchronize();

    ThrustSort(gMemory->dFluidParticle, dParticleHash, NumParticles);
    getLastCudaError("thrustSort");
    cudaDeviceSynchronize();

    // cudaEvent_t start, stop;
    // float elapsedTime = 0.0;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start, 0);

    uint memSize = sizeof(uint) * (numThreads + 1);
    calCellRange <<< gridDim, blockDim, memSize , 0>>> (gMemory->dIndexRange, NumParticles, dParticleHash, gMemory->dFluidParticle);
	getLastCudaError("calCellRange");
    cudaDeviceSynchronize();

    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // std::cout << "elasped time:" << elapsedTime << std::endl; 

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    
    safeCudaFree((void**)&dParticleHash);
}  