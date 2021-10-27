#include <GPUmemory.h>

__constant__  SimParams      dSimParams;
__constant__  SurfaceParams  dSurfaceParams;
__constant__  GridParams     dGridParams;
__constant__  DiffuseParams  dDiffuseParams;

__constant__ uint     dNumSurfaceParticles;
__constant__ uint     dNumInvolveParticles;
__constant__ uint     dNumSurfaceVertices;

__constant__ uint     dNumFreeSurfaceParticles;
__constant__ uint     dGeneratedNumDiffuseParticles;
__constant__ uint     dNumIsDiffuseParticles;

__constant__ uint     dOldNumDiffuseParticles;
__constant__ uint     dNumParticles;

__constant__ uint     dNumValidSurfaceCubes;
__constant__ uint     dNumSurfaceMeshVertices;
