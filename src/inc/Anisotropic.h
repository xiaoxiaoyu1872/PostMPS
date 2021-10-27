#ifndef METHODOFYU_H_
#define METHODOFYU_H_

#include "Params.h"
#include "GPUmemory.h"
#include "myCuda.cuh"

class Anisotropic
{
public:
    Anisotropic(GPUmemory *gMemory, Params *params);
    ~Anisotropic();

    void processingOfParticles(record* paper);
    void processingOfVertices(record* paper);
    void estimationOfscalarField();
    
private:
    GPUmemory* gMemory;
    Params* params;

    cudaStream_t s1, s2;
    
    uint NumSurfaceParticles;
    uint NumInvolveParticles;

    uint NumSurfaceVertices;

    void extractionOfSurfaceAndInvolveParticles();
    void thrustscan_particles();
    void streamcompact_particles();
    void memallocation_particles();
    void smoothedparticles();
    void transformmatrices();

    void extractionOfSurfaceVertices();
    void thrustscan_vertices();
	void memallocation_vertices();
    void streamcompact_vertices();
	void memallocation_scalarvalue();

    void scalarfield();

    void constantMemCopy();
    void constantMemSurAndInvPar_Num();
    void constantMemSurVer_Num();

    record* record_single;
};

#endif 