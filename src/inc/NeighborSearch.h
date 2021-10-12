#ifndef HASHSPATIALGRID_H
#define HASHSPATIALGRID_H

#include "GPUmemory.h"
#include "myCuda.cuh"


class NeighborSearch
{
public:
    NeighborSearch(GPUmemory *gMemory, Params* params);

    void BoundGridBuilding();
    void SpatialGridBuilding();

    void constantMemCopy_Grid();
private:
    GPUmemory* gMemory;
    Params* params;
};

#endif