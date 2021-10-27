#ifndef _GPUMEMORY_H_
#define _GPUMEMORY_H_
#include "Define.h"
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>

#include "myCuda.cuh"
#include "Params.h"

class GPUmemory
{
public:
    GPUmemory();
    ~GPUmemory();

public:      
    uint NumParticles;
    uint NumBoundParticles;

    uint NumSurfaceParticles;
    uint NumInvolveParticles;

    uint NumSurfaceVertices;

    uint NumValidSurfaceCubes;

    uint NumSurfaceMeshVertices;

    uint NumFreeSurfaceParticles;

    uint GeneratedNumDiffuseParticles;
    uint NumIsDiffuseParticles;
    uint OldNumDiffuseParticles;

    Params *params;

    BoundGrid *dBoundGrid;
    SpatialGrid *dSpatialGrid;

    ScalarFieldGrid *dScalarFiled;

    IsSurface *dIsSurfaceVertices;
    IsSurfaceScan *dIsSurfaceVerticesScan;

    IndexRange *dIndexRange;

    FluidParticle* dFluidParticle;
    BoundParticle* dBoundParticle;

    IsSurface* dIsSurfaceParticle;
    IsSurfaceScan* dIsSurfaceParticleScan;

    NumParticleGrid* dNumSurParticleGrid;
    NumParticleGrid* dNumInvParticleGrid;
    NumParticleGrid* dNumFreeSurParticleGrid;

    NumParticleGridScan* dNumSurParticleGridScan;
    NumParticleGridScan* dNumInvParticleGridScan;
    NumParticleGridScan* dNumFreeSurParticleGridScan;

    IsValid* dIsValidSurfaceCube;
    IsValidScan* dIsValidSurfaceCubeScan;

    NumVertexCube*      dNumVertexCube;
    NumVertexCubeScan*  dNumVertexCubeScan;

    CubeFlag* dCubeFlag;

    Index* dSurfaceParticlesIndex;
    Index* dInvolveParticlesIndex;
    Index* dFreeSurfaceParticlesIndex;
    Index* dDiffuseParticlesIndex;

    DiffuseParticle* dDiffuseParticle;
    DiffuseParticle* dDiffuseParticle_old;

    Index* dSurfaceVerticesIndex;
    Index* dValidCubesIndex;


    SmoothedPos* dSurfaceParticlesSmoothed;
    SmoothedPos* dInvolveParticlesSmoothed;

    MeanPos* dSurfaceParticlesMean;

    MatrixValue* dSVDMatrices;

    Vertex* dVertex;
    Normal* dNormal;

    ScalarValue* dScalarValue;

    DiffusePotential* dDiffusePotential;
    // ThinFeature*      dThinFeature;
    float3*      dThinFeature;

    IsDiffuse* dIsDiffuse;
    IsDiffuseScan* dIsDiffuseScan;

    NumDiffuseParticle*    dNumDiffuseParticle;
    NumDiffuseParticleScan*  dNumDiffuseParticleScan;

    ColorField* dColorField;

    uint* dEdgeTable;									
    int* dEdgeIndicesOfTriangleTable;					
    uint* dNumVerticesTable;							
    uint* dVertexIndicesOfEdgeTable;

    void initGPUmemory(Params *params);

    void SpatialGridMem();
    void Memreset();
    void Memfree();

    void finaGPUmemory();

    void memAlcandCpy_fluid(std::vector<FluidParticle> mFluidParticle);
    void memAlcandCpy_bound(std::vector<BoundParticle> mBoundParticle);

    //------------------------surface---------------------------
    void SurfaceAlloFixedMem();
    void SurfaceFreeFixedMem();

    void AlloTextureMem();
    void BindTextures();
    void FreeTextureMem();

    void SurfaceMemreset();

    void MethodYuMemFree();
    void MCMemFree();

    void memAllocation_particles();
    void memAllocation_vertices();
    void memAllocation_cubes();
    void memAllocation_triangles();
    void memAllocation_scalarvalues();

    //------------------------diffuse---------------------------
    void DiffuseAlloFixedMem();
    void DiffuseFreeFixedMem();

    void DiffuseMemreset();

    void DiffuseMemFree();

    void DiffuseMemFinal();
    void Memfree_bound();
    
    void memallocation_freeparticles();
    void memallocation_potional();
    void memallocation_diffuseparticles();
    void memallocation_olddiffuseparticles(std::vector<DiffuseParticle> mDiffuse);

    //------------------------feal---------------------------
    void memallocation_feal();
    //------------------------cpu---------------------------
    void memAllocation_cubes_cpu();
};


#endif