#include "GPUmemory.h"
#include "MarchingCubesHelper.h"

GPUmemory::GPUmemory()
{

}
GPUmemory::~GPUmemory()
{

}


void GPUmemory::initGPUmemory(Params *_params)
{
    params = _params;
	SpatialGridMem();
}

void GPUmemory::SpatialGridMem()
{
	// if (params->mConfigParams.isDiffuse)
	// {
	// 	cudaMallocMemset((void**)&dBoundGrid, 0.0f, params->mGridParams.spSize*sizeof(BoundGrid));
	// }	

	// cudaMallocMemset((void**)&dSpatialGrid, 0.0f, params->mGridParams.spSize*sizeof(SpatialGrid));
	// cudaMallocMemset((void**)&dIndexRange, 0.0f, params->mGridParams.spSize*sizeof(IndexRange));	
}

void GPUmemory::Memreset()
{
	// GridParams mGridParams = params->mGridParams;

	// checkCudaErrors(cudaMemset(dSpatialGrid, 0.0f, 
    // mGridParams.spSize * sizeof(SpatialGrid)));

    // checkCudaErrors(cudaMemset(dIndexRange, 0.0f, 
    // mGridParams.spSize * sizeof(IndexRange)));
}

void GPUmemory::Memfree()
{
	safeCudaFree((void**)&dFluidParticle);
}

void GPUmemory::memAlcandCpy_fluid(std::vector<FluidParticle> _mFluidParticle)
{
//-------------------------------------------------------------------------------------------
	cudaMallocMemset((void**)&dScalarFiled, 0.0f, params->mGridParams.scSize*sizeof(ScalarFieldGrid));
	cudaMallocMemset((void**)&dIsSurfaceVertices, 0.0f, params->mGridParams.scSize*sizeof(IsSurface));
	cudaMallocMemset((void**)&dIsSurfaceVerticesScan, 0.0f, params->mGridParams.scSize*sizeof(IsSurfaceScan));

	cudaMallocMemset((void**)&dNumSurParticleGrid, 0.0f, params->mGridParams.spSize*sizeof(NumParticleGrid));
	cudaMallocMemset((void**)&dNumInvParticleGrid, 0.0f, params->mGridParams.spSize*sizeof(NumParticleGrid));

	cudaMallocMemset((void**)&dNumSurParticleGridScan, 0.0f, params->mGridParams.spSize*sizeof(NumParticleGridScan));
	cudaMallocMemset((void**)&dNumInvParticleGridScan, 0.0f, params->mGridParams.spSize*sizeof(NumParticleGridScan));

	cudaMallocMemset((void**)&dSpatialGrid, 0.0f, params->mGridParams.spSize*sizeof(SpatialGrid));
	cudaMallocMemset((void**)&dIndexRange, 0.0f, params->mGridParams.spSize*sizeof(IndexRange));	
//-------------------------------------------------------------------------------------------
	NumParticles = _mFluidParticle.size();
    checkCudaErrors(cudaMalloc((void**)&dFluidParticle, 
    (size_t)NumParticles*sizeof(FluidParticle)));
    
    checkCudaErrors(cudaMemcpy(dFluidParticle, static_cast<void*>(_mFluidParticle.data()),
			NumParticles * sizeof(FluidParticle), cudaMemcpyHostToDevice));
//-------------------------------------------------------------------------------------------
	cudaMallocMemset((void**)&dIsSurfaceParticle, 0, (size_t)NumParticles*sizeof(IsSurface));
	cudaMallocMemset((void**)&dIsSurfaceParticleScan, 0, (size_t)NumParticles*sizeof(IsSurfaceScan));
}

void GPUmemory::memAlcandCpy_bound(std::vector<BoundParticle> _mBoundParticle)
{
	NumBoundParticles = _mBoundParticle.size();

    checkCudaErrors(cudaMalloc((void**)&dBoundParticle, 
    (size_t)NumBoundParticles*sizeof(BoundParticle)));

    checkCudaErrors(cudaMemcpy(dBoundParticle, 
    static_cast<void*>(_mBoundParticle.data()),
			NumBoundParticles * sizeof(BoundParticle), cudaMemcpyHostToDevice));
}

void GPUmemory::finaGPUmemory()
{
	// safeCudaFree((void**)&dSpatialGrid);
	// safeCudaFree((void**)&dIndexRange);
}

//------------------------surface---------------------------
void GPUmemory::SurfaceAlloFixedMem()
{
	// cudaMallocMemset((void**)&dScalarFiled, 0.0f, params->mGridParams.scSize*sizeof(ScalarFieldGrid));
	// cudaMallocMemset((void**)&dIsSurfaceVertices, 0.0f, params->mGridParams.scSize*sizeof(IsSurface));
	// cudaMallocMemset((void**)&dIsSurfaceVerticesScan, 0.0f, params->mGridParams.scSize*sizeof(IsSurfaceScan));

	// cudaMallocMemset((void**)&dNumSurParticleGrid, 0.0f, params->mGridParams.spSize*sizeof(NumParticleGrid));
	// cudaMallocMemset((void**)&dNumInvParticleGrid, 0.0f, params->mGridParams.spSize*sizeof(NumParticleGrid));

	// cudaMallocMemset((void**)&dNumSurParticleGridScan, 0.0f, params->mGridParams.spSize*sizeof(NumParticleGridScan));
	// cudaMallocMemset((void**)&dNumInvParticleGridScan, 0.0f, params->mGridParams.spSize*sizeof(NumParticleGridScan));
}

void GPUmemory::AlloTextureMem()
{
	cudaMallocMemcpy((void**)&dEdgeTable, (void*)MarchingCubesHelper::edgeFlags, 256 * sizeof(uint));
	cudaMallocMemcpy((void**)&dEdgeIndicesOfTriangleTable, (void*)MarchingCubesHelper::edgeIndexesOfTriangle, 256 * 16 * sizeof(int));
	cudaMallocMemcpy((void**)&dNumVerticesTable, (void*)MarchingCubesHelper::numVertices, 256 * sizeof(uint));
	cudaMallocMemcpy((void**)&dVertexIndicesOfEdgeTable, (void*)MarchingCubesHelper::vertexIndexesOfEdge, 12 * 2 * sizeof(int));
}

void GPUmemory::FreeTextureMem()
{
	safeCudaFree((void**)&dEdgeTable);
	safeCudaFree((void**)&dEdgeIndicesOfTriangleTable);
	safeCudaFree((void**)&dNumVerticesTable);
	safeCudaFree((void**)&dVertexIndicesOfEdgeTable);
}

void GPUmemory::SurfaceMemreset()
{
	GridParams mGridParams = params->mGridParams;

	// checkCudaErrors(cudaMemset(dNumSurParticleGrid, 0.0f, 
    // mGridParams.spSize * sizeof(NumParticleGrid)));
    // checkCudaErrors(cudaMemset(dNumInvParticleGrid, 0.0f, 
    // mGridParams.spSize * sizeof(NumParticleGrid)));

	// checkCudaErrors(cudaMemset(dNumSurParticleGridScan, 0.0f, 
    // mGridParams.spSize * sizeof(NumParticleGridScan)));
    // checkCudaErrors(cudaMemset(dNumInvParticleGridScan, 0.0f, 
    // mGridParams.spSize * sizeof(NumParticleGridScan)));

	// checkCudaErrors(cudaMemset(dIsSurfaceVertices, 0.0f, 
    // mGridParams.scSize * sizeof(IsSurface)));
	// checkCudaErrors(cudaMemset(dIsSurfaceVerticesScan, 0.0f, 
    // mGridParams.scSize * sizeof(IsSurfaceScan)));

	// checkCudaErrors(cudaMemset(dScalarFiled, 0.0f, 
    // mGridParams.scSize * sizeof(ScalarFieldGrid)));
}

void GPUmemory::MethodYuMemFree()
{
	safeCudaFree((void**)&dIsSurfaceParticle);
	safeCudaFree((void**)&dIsSurfaceParticleScan);

	safeCudaFree((void**)&dSurfaceParticlesIndex);
	safeCudaFree((void**)&dInvolveParticlesIndex);

	safeCudaFree((void**)&dSurfaceParticlesSmoothed);
	safeCudaFree((void**)&dInvolveParticlesSmoothed);

	safeCudaFree((void**)&dSurfaceParticlesMean);

	safeCudaFree((void**)&dSVDMatrices);
//-------------------------------------------------------------------------------------------

	safeCudaFree((void**)&dNumSurParticleGrid);
	safeCudaFree((void**)&dNumInvParticleGrid);

	safeCudaFree((void**)&dNumSurParticleGridScan);
	safeCudaFree((void**)&dNumInvParticleGridScan);
}

void GPUmemory::MCMemFree()
{
	safeCudaFree((void**)&dSurfaceVerticesIndex);

	safeCudaFree((void**)&dIsValidSurfaceCube);
	safeCudaFree((void**)&dIsValidSurfaceCubeScan);

	safeCudaFree((void**)&dNumVertexCube);
	safeCudaFree((void**)&dNumVertexCubeScan);
	
	safeCudaFree((void**)&dCubeFlag);

	safeCudaFree((void**)&dValidCubesIndex);
	safeCudaFree((void**)&dVertex);
	safeCudaFree((void**)&dNormal);
	safeCudaFree((void**)&dScalarValue);
	
//-------------------------------------------------------------------------------------------
	safeCudaFree((void**)&dScalarFiled);
	safeCudaFree((void**)&dIsSurfaceVertices);
	safeCudaFree((void**)&dIsSurfaceVerticesScan);

//-------------------------------------------------------------------------------------------
	safeCudaFree((void**)&dSpatialGrid);
	safeCudaFree((void**)&dIndexRange);
}

void GPUmemory::SurfaceFreeFixedMem()
{
	// safeCudaFree((void**)&dScalarFiled);
	// safeCudaFree((void**)&dIsSurfaceVertices);
	// safeCudaFree((void**)&dIsSurfaceVerticesScan);

	// safeCudaFree((void**)&dNumSurParticleGrid);
	// safeCudaFree((void**)&dNumInvParticleGrid);

	// safeCudaFree((void**)&dNumSurParticleGridScan);
	// safeCudaFree((void**)&dNumInvParticleGridScan);
}


void GPUmemory::memAllocation_particles()
{
	cudaMallocMemset((void**)&dSurfaceParticlesIndex, 0.0f, NumSurfaceParticles*sizeof(Index));
	cudaMallocMemset((void**)&dSurfaceParticlesSmoothed, 0.0f, NumSurfaceParticles*sizeof(SmoothedPos));
	
	cudaMallocMemset((void**)&dInvolveParticlesSmoothed, 0.0f, NumInvolveParticles*sizeof(SmoothedPos));
	cudaMallocMemset((void**)&dInvolveParticlesIndex, 0.0f, NumInvolveParticles*sizeof(Index));

	cudaMallocMemset((void**)&dSurfaceParticlesMean, 0.0f, NumSurfaceParticles*sizeof(MeanPos));

	cudaMallocMemset((void**)&dSVDMatrices, 0.0f, NumSurfaceParticles*sizeof(MatrixValue));
}

void GPUmemory::memAllocation_vertices()
{
	cudaMallocMemset((void**)&dSurfaceVerticesIndex, 0.0f, NumSurfaceVertices*sizeof(Index));
}

void GPUmemory::memAllocation_cubes()
{
	cudaMallocMemset((void**)&dIsValidSurfaceCube, 0.0f, 
	NumSurfaceVertices*sizeof(IsValid));

	cudaMallocMemset((void**)&dIsValidSurfaceCubeScan, 0.0f, 
	NumSurfaceVertices*sizeof(IsValidScan));

	cudaMallocMemset((void**)&dNumVertexCube, 0.0f, 
	NumSurfaceVertices*sizeof(NumVertexCube));
	cudaMallocMemset((void**)&dNumVertexCubeScan, 0.0f, 
	NumSurfaceVertices*sizeof(NumVertexCubeScan));

	cudaMallocMemset((void**)&dCubeFlag, 0.0f, 
	NumSurfaceVertices*sizeof(CubeFlag));
}

void GPUmemory::memAllocation_cubes_cpu()
{
	cudaMallocMemset((void**)&dIsValidSurfaceCube, 0.0f, 
	NumSurfaceVertices*sizeof(IsValid));
}


void GPUmemory::memAllocation_triangles()
{
	cudaMallocMemset((void**)&dValidCubesIndex, 0.0f, NumValidSurfaceCubes*sizeof(Index));
	cudaMallocMemset((void**)&dVertex, 0.0f, NumSurfaceMeshVertices*sizeof(Vertex));
	cudaMallocMemset((void**)&dNormal, 0.0f, NumSurfaceMeshVertices*sizeof(Normal));
}

void GPUmemory::memAllocation_scalarvalues()
{
	cudaMallocMemset((void**)&dScalarValue, 0.0f, NumSurfaceMeshVertices*sizeof(ScalarValue));
}


//------------------------diffuse---------------------------
void GPUmemory::DiffuseAlloFixedMem()
{
	cudaMallocMemset((void**)&dNumFreeSurParticleGrid, 0.0f, params->mGridParams.spSize*sizeof(NumParticleGrid));
	cudaMallocMemset((void**)&dNumFreeSurParticleGridScan, 0.0f, params->mGridParams.spSize*sizeof(NumParticleGridScan));
}

void GPUmemory::DiffuseFreeFixedMem()
{
	safeCudaFree((void**)&dNumFreeSurParticleGrid);
	safeCudaFree((void**)&dNumFreeSurParticleGridScan);
}


void GPUmemory::DiffuseMemreset()
{
	// GridParams mGridParams = params->mGridParams;
	// checkCudaErrors(cudaMemset(dNumFreeSurParticleGrid, 0.0f, 
    // mGridParams.spSize * sizeof(NumParticleGrid)));
	// checkCudaErrors(cudaMemset(dNumFreeSurParticleGridScan, 0.0f, 
    // mGridParams.spSize * sizeof(NumParticleGridScan)));
}

void GPUmemory::DiffuseMemFree()
{
	safeCudaFree((void**)&dFreeSurfaceParticlesIndex);

	safeCudaFree((void**)&dThinFeature);
	safeCudaFree((void**)&dDiffusePotential);
	safeCudaFree((void**)&dColorField);

	safeCudaFree((void**)&dIsDiffuse);
	safeCudaFree((void**)&dIsDiffuseScan);

	safeCudaFree((void**)&dNumDiffuseParticle);
	safeCudaFree((void**)&dNumDiffuseParticleScan);

	safeCudaFree((void**)&dDiffuseParticlesIndex);
	safeCudaFree((void**)&dDiffuseParticle);
}

void GPUmemory::Memfree_bound()
{
	safeCudaFree((void**)&dBoundGrid);
}

void GPUmemory::DiffuseMemFinal()
{
	safeCudaFree((void**)&dDiffuseParticle_old);
}


void GPUmemory::memallocation_freeparticles()
{
	cudaMallocMemset((void**)&dFreeSurfaceParticlesIndex, 0.0f, NumFreeSurfaceParticles*sizeof(Index));

	// cudaMallocMemset((void**)&dThinFeature, 0.0f, NumFreeSurfaceParticles*sizeof(ThinFeature));
	// cudaMallocMemset((void**)&dColorField, 0.0f, NumParticles*sizeof(ColorField));
}

void GPUmemory::memallocation_potional()
{
	cudaMallocMemset((void**)&dDiffusePotential, 0.0f, NumParticles*sizeof(DiffusePotential));

	cudaMallocMemset((void**)&dIsDiffuse, 0.0f, NumParticles*sizeof(IsDiffuse));
	cudaMallocMemset((void**)&dIsDiffuseScan, 0.0f, NumParticles*sizeof(IsDiffuseScan));

	cudaMallocMemset((void**)&dNumDiffuseParticle, 0.0f, NumParticles*sizeof(NumDiffuseParticle));
	cudaMallocMemset((void**)&dNumDiffuseParticleScan, 0.0f, NumParticles*sizeof(NumDiffuseParticleScan));
}

void GPUmemory::memallocation_diffuseparticles()
{
	cudaMallocMemset((void**)&dDiffuseParticlesIndex, 0.0f, GeneratedNumDiffuseParticles*sizeof(Index));
	cudaMallocMemset((void**)&dDiffuseParticle, 0.0f, GeneratedNumDiffuseParticles*sizeof(DiffuseParticle));
}	

void GPUmemory::memallocation_olddiffuseparticles(std::vector<DiffuseParticle> mDiffuse)
{
	safeCudaFree((void**)&dDiffuseParticle_old);
	cudaMallocMemset((void**)&dDiffuseParticle_old, 0.0f, OldNumDiffuseParticles*sizeof(DiffuseParticle));
	checkCudaErrors(cudaMemcpy(dDiffuseParticle_old, static_cast<void*>(mDiffuse.data()),
			OldNumDiffuseParticles * sizeof(DiffuseParticle), cudaMemcpyHostToDevice));
}	


//-----------------------------------
void GPUmemory::memallocation_feal()
{
	cudaMallocMemset((void**)&dColorField, 0.0f, NumParticles*sizeof(ColorField));
	cudaMallocMemset((void**)&dDiffusePotential, 0.0f, NumParticles*sizeof(DiffusePotential));

	cudaMallocMemset((void**)&dIsDiffuse, 0.0f, NumParticles*sizeof(IsDiffuse));
	cudaMallocMemset((void**)&dIsDiffuseScan, 0.0f, NumParticles*sizeof(IsDiffuseScan));

	cudaMallocMemset((void**)&dNumDiffuseParticle, 0.0f, NumParticles*sizeof(NumDiffuseParticle));
	cudaMallocMemset((void**)&dNumDiffuseParticleScan, 0.0f, NumParticles*sizeof(NumDiffuseParticleScan));

	//-----------------------------------------------
	cudaMallocMemset((void**)&dThinFeature, 0.0f, NumParticles*sizeof(float3));
}