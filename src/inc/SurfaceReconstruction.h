#ifndef SURFACERECONSTRUCTION_H_
#define SURFACERECONSTRUCTION_H_

#include "Params.h"
#include "GPUmemory.h"
#include "myCuda.cuh"
#include "FileData.h"

#include "Anisotropic.h"
#include "MarchingCube.h"

#include <ctime>

class SurfaceReconstruction
{
public:
    SurfaceReconstruction();
    ~SurfaceReconstruction();
    void Init(GPUmemory *gMemory, Params *params, FileData *fileData);
    void Destory();
    void runsimulation();
    void saveMiddleFile();
private:
    GPUmemory* gMemory;
    Params* params;
    FileData* fileData;

    Anisotropic* anisotropicMethod;
    MarchingCube* marchingCube;

    std::vector<record*> records;
};

#endif 