#include <string>
#include <iostream>
#include "GPUPOST.h"

GPUPOST::GPUPOST()
{
    params = new Params();
    gMemory  = new GPUmemory();
}


GPUPOST::~GPUPOST()
{
    delete params;
    delete gMemory;
    delete fileData;
}


void GPUPOST::initialize()
{
    // checkMemUsed();
    params->setFilname();
    params->setParams();
    params->printInfo();

    params->setGPUId();
    params->printGPUInfo();

    gMemory->initGPUmemory(params);
}


void GPUPOST::runSimulation()
{
    fileData = new FileData(params,gMemory);

    NeighborSearch neighborSearch(gMemory, params);

    SurfaceReconstruction surfaceReconstruction;
    if (params->mConfigParams.isSurface){
        surfaceReconstruction.Init(gMemory, params, fileData);
    }

    DiffuseGeneration diffuseGeneration;
    if (params->mConfigParams.isDiffuse)
    {
        diffuseGeneration.Init(gMemory, params, fileData);
    }

    for (uint frameIndex = params->mConfigParams.frameStart; 
              frameIndex <= params->mConfigParams.frameEnd; 
              frameIndex += params->mConfigParams.frameStep)
    {
        // gMemory->Memreset();
        float3 motion = {0,0,0};
        // fileData->loadMotionFile(frameIndex, motion);
        fileData->loadFluidFile(frameIndex, motion);
        //fileData->loadMotionDatFile(frameIndex, motion);
        // fileData->loadFluidDatFile(frameIndex, motion);
        // checkMemUsed();

        neighborSearch.SpatialGridBuilding();

        if (params->mConfigParams.isSurface){
            surfaceReconstruction.runsimulation();
        }

        if (params->mConfigParams.isDiffuse){
            diffuseGeneration.runsimulation();
        }
        
        if(params->mConfigParams.isSurface){
            gMemory->MethodYuMemFree();
            gMemory->MCMemFree();
        }

        if (params->mConfigParams.isDiffuse){
            gMemory->DiffuseMemFree();
        }
        gMemory->Memfree();
    }

    if(params->mConfigParams.isSurface){
        surfaceReconstruction.Destory();
    }

    if(params->mConfigParams.isDiffuse){
        diffuseGeneration.Destory();
    }
} 


void GPUPOST::finalize()
{
    gMemory->finaGPUmemory();
}