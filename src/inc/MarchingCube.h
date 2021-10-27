#ifndef PARTICLE_H_
#define PARTICLE_H_

#include "Params.h"
#include "GPUmemory.h"

#include "myCuda.cuh"
#include "Define.h"

class MarchingCube
{
public:
    MarchingCube(GPUmemory *gMemory, Params *params);
    ~MarchingCube();
    void triangulation(record* record_single);

    std::vector<fVector3> mVertexArray;							//! vertex array of surface mesh.
	std::vector<fVector3> mNormalArray;							//! normal array of surface mesh.

	std::vector<uint> mSurfaceVerticesIndexArray;
	std::vector<uint> mSurfaceParticlesIndexArray;
	std::vector<uint> mValidSurfaceCubesIndexArray;

private:
    GPUmemory* gMemory;
    Params* params;

    uint NumSurfaceVertices;
    uint NumValidSurfaceCubes;
    uint NumSurfaceMeshVertices;

    void memallocation_cubes();
    void detectionOfValidCubes();
    void thrustscan_cubes();
    void streamcompact_cubes();
    void memallocation_triangles();
    void marchingcubes();

    void memallocation_scalarvalue();
    void scalarvalue();

    void constantMemCopy();
    void BindingTexMem();
    void constantMemSurVer_Num();
    void constantMemValCube_Num();

    void generationOfSurfaceMeshUsingMCForAkinci();

    record* record_single;
};



#endif