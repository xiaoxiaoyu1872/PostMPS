#ifndef GPUPOST_H_
#define GPUPOST_H_

#include <sstream>
#include <string>
#include "SurfaceReconstruction.h"
#include "DiffuseGeneration.h"
#include "GPUmemory.h"
#include "Params.h"
#include "FileData.h"
#include "NeighborSearch.h"

class GPUPOST 
{
public:
    GPUPOST();  
    ~GPUPOST();      
    void initialize();
    void runSimulation();
    void finalize();      
private:
    GPUmemory*             gMemory; 
    Params*                params;
    FileData*              fileData;
};

#endif 