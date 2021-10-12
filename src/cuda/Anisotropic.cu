#include "Anisotropic.h"
#include "Anisotropic_kernel.cu"
#include "Thrust.cuh"

Anisotropic::Anisotropic(GPUmemory *_gMemory, Params *_params)
{
    params = _params;
    gMemory = _gMemory;

	gMemory->SurfaceAlloFixedMem();
    // constantMemCopy();

    int priority_high, priority_low;
	cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);

	cudaStreamCreateWithPriority(&s1, cudaStreamDefault, priority_high);
	cudaStreamCreateWithPriority(&s2, cudaStreamDefault, priority_low);
}

Anisotropic::~Anisotropic()
{
    cudaStreamDestroy(s1);
	cudaStreamDestroy(s2);

	gMemory->SurfaceFreeFixedMem();
	std::cout << "~~Anisotropic" << std::endl;
}


void Anisotropic::processingOfParticles(record* _record)
{
	record_single = _record;
	constantMemCopy();
    extractionOfSurfaceAndInvolveParticles();
    thrustscan_particles();
    memallocation_particles();
    streamcompact_particles();
    smoothedparticles();
    transformmatrices();
}


void Anisotropic::processingOfVertices(record* _record)
{
	record_single = _record;
	extractionOfSurfaceVertices();
    thrustscan_vertices();
	memallocation_vertices();
    streamcompact_vertices();
}


void Anisotropic::estimationOfscalarField()
{
	scalarfield();
}


void Anisotropic::extractionOfSurfaceAndInvolveParticles()
{
    dim3 gridDim, blockDim;
	calcGridDimBlockDim(params->mGridParams.spSize, gridDim, blockDim); 

    estimationOfSurfaceParticles <<< gridDim, blockDim >>> (
		gMemory->dSpatialGrid,
		gMemory->dNumSurParticleGrid,
		gMemory->dNumInvParticleGrid,
		gMemory->dIndexRange);

	cudaDeviceSynchronize();

	// estimationOfInvolveParticles  <<< gridDim, blockDim >>> (
	// 	gMemory->dNumSurParticleGrid,
	// 	gMemory->dNumInvParticleGrid);
	// cudaDeviceSynchronize();
}

void Anisotropic::thrustscan_particles()
{
    NumSurfaceParticles = ThrustExclusiveScan(
		gMemory->dNumSurParticleGridScan,
		gMemory->dNumSurParticleGrid,
		(uint)params->mGridParams.spSize);

	gMemory->NumSurfaceParticles = NumSurfaceParticles;

	if (NumSurfaceParticles == 0)
	{
		std::cerr << "No surface particle detected!\n";
		return;
	}

	std::cout << "NumSurfaceParticles =  " << NumSurfaceParticles << std::endl;

	record_single->surpar = static_cast<double>(NumSurfaceParticles)
		/ gMemory->NumParticles;

	// NumInvolveParticles = ThrustExclusiveScan(
	// 	gMemory->dNumInvParticleGridScan,
	// 	gMemory->dNumInvParticleGrid,
	// 	(uint)params->mGridParams.spSize);

	// gMemory->NumInvolveParticles = NumInvolveParticles;

	// if (NumInvolveParticles == 0)
	// {
	// 	std::cerr << "No involve particle detected!\n";
	// 	return;
	// }

	// std::cout << "mNumInvolveParticles =  " << NumInvolveParticles << std::endl;
}


void Anisotropic::streamcompact_particles()
{
    dim3 gridDim, blockDim;
	calcGridDimBlockDim(params->mGridParams.spSize, gridDim, blockDim);

	compactationOfParticles << < gridDim, blockDim, 0, s1>> > (
		gMemory->dNumSurParticleGrid,
		gMemory->dNumSurParticleGridScan,
		gMemory->dIndexRange,
		gMemory->dSurfaceParticlesIndex);	
	cudaStreamSynchronize(s1);

	// compactationOfParticles << < gridDim, blockDim, 0,  s2>> > (
	// 	gMemory->dNumInvParticleGrid,
	// 	gMemory->dNumInvParticleGridScan,
	// 	gMemory->dIndexRange,
	// 	gMemory->dInvolveParticlesIndex);
	// cudaStreamSynchronize(s2);
}

void Anisotropic::smoothedparticles()
{
    dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumSurfaceParticles, gridDim, blockDim);
	calculationOfSmoothed << < gridDim, blockDim , 0, s1>> > (
		gMemory->dFluidParticle,
		gMemory->dSurfaceParticlesMean,
		gMemory->dSurfaceParticlesSmoothed,
		gMemory->dIndexRange,
		gMemory->dSurfaceParticlesIndex,
		gMemory->dSpatialGrid);
	cudaStreamSynchronize(s1);
	
	// calcGridDimBlockDim(NumInvolveParticles, gridDim, blockDim);
	// calculationOfSmoothedforInvovle << < gridDim, blockDim , 0, s2>> > (
	// 	gMemory->dFluidParticle,
	// 	gMemory->dInvolveParticlesSmoothed,
	// 	gMemory->dIndexRange,
	// 	gMemory->dInvolveParticlesIndex);
	// cudaStreamSynchronize(s2);

	calcGridDimBlockDim(gMemory->NumParticles, gridDim, blockDim);
	calculationOfDensity << < gridDim, blockDim>> > (
		gMemory->dFluidParticle,
		gMemory->dIndexRange,
		gMemory->dSpatialGrid,
		gMemory->NumParticles);
}


void Anisotropic::transformmatrices()
{
    dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumSurfaceParticles, gridDim, blockDim);
	calculationOfTransformMatrices << < gridDim, blockDim, 0, s1 >> > (
		gMemory->dSurfaceParticlesMean,
		gMemory->dFluidParticle,
		gMemory->dIndexRange,
		gMemory->dSurfaceParticlesIndex,
		gMemory->dSVDMatrices);

	cudaStreamSynchronize(s1);
}


void Anisotropic::extractionOfSurfaceVertices()
{
	dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumSurfaceParticles, gridDim, blockDim);
	estimationOfSurfaceVertices << < gridDim, blockDim>> > (
		gMemory->dFluidParticle,
		gMemory->dSurfaceParticlesSmoothed,
		gMemory->dIndexRange,
		gMemory->dSurfaceParticlesIndex,
		gMemory->dIsSurfaceVertices,
		gMemory->dSVDMatrices,
		gMemory->dSpatialGrid,
		gMemory->dScalarFiled);

	cudaDeviceSynchronize();
}


void Anisotropic::thrustscan_vertices()
{
	NumSurfaceVertices = ThrustExclusiveScan(
		gMemory->dIsSurfaceVerticesScan,
		gMemory->dIsSurfaceVertices,
		(uint)params->mGridParams.scSize);

	gMemory->NumSurfaceVertices = NumSurfaceVertices;

	if (NumSurfaceVertices == 0)
	{
		std::cerr << "No surface vertex detected!\n";
		return;
	}
	std::cout << "NumSurfaceVertices =  " << NumSurfaceVertices << std::endl;

	record_single->surver = static_cast<double>(NumSurfaceVertices) /
		(params->mGridParams.scSize);
}


void Anisotropic::streamcompact_vertices()
{
	dim3 gridDim, blockDim;
	calcGridDimBlockDim(params->mGridParams.scSize, gridDim, blockDim);

	compactationOfSurfaceVertices << < gridDim, blockDim>> > (
		gMemory->dIsSurfaceVertices,
		gMemory->dIsSurfaceVerticesScan,
		gMemory->dSurfaceVerticesIndex);

	cudaDeviceSynchronize();
}


void Anisotropic::scalarfield()
{
	dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumSurfaceVertices, gridDim, blockDim);

	computationOfScalarFieldGrid << < gridDim, blockDim>> > (
		gMemory->dFluidParticle,
		gMemory->dSurfaceParticlesSmoothed,
		gMemory->dInvolveParticlesSmoothed,
		gMemory->dIndexRange,
		gMemory->dSurfaceParticlesIndex,
		gMemory->dInvolveParticlesIndex,
		gMemory->dSurfaceVerticesIndex,
		gMemory->dNumSurParticleGrid,
		gMemory->dNumSurParticleGridScan,
		gMemory->dNumInvParticleGrid,
		gMemory->dNumInvParticleGridScan,
		gMemory->dSVDMatrices,
		gMemory->dScalarFiled);

	cudaDeviceSynchronize();
}


void Anisotropic::constantMemCopy()
{
    checkCudaErrors(cudaMemcpyToSymbol(dSurfaceParams, &params->mSurfaceParams, sizeof(SurfaceParams)));

	checkCudaErrors(cudaMemcpyToSymbol(dSimParams, &params->mSimParams, sizeof(SimParams)));

	checkCudaErrors(cudaMemcpyToSymbol(dGridParams, &params->mGridParams, sizeof(GridParams)));
}


void Anisotropic::constantMemSurAndInvPar_Num()
{
	checkCudaErrors(cudaMemcpyToSymbol(dNumSurfaceParticles, 
	&gMemory->NumSurfaceParticles, sizeof(uint)));
	checkCudaErrors(cudaMemcpyToSymbol(dNumInvolveParticles, 
	&gMemory->NumInvolveParticles, sizeof(uint)));	
}


void Anisotropic::constantMemSurVer_Num()
{
	checkCudaErrors(cudaMemcpyToSymbol(dNumSurfaceVertices, 
	&gMemory->NumSurfaceVertices, sizeof(uint)));
}

void Anisotropic::memallocation_particles()
{
    gMemory->memAllocation_particles();
	constantMemSurAndInvPar_Num();
}

void Anisotropic::memallocation_vertices()
{
	gMemory->memAllocation_vertices();
	constantMemSurVer_Num();
}

