#include "Params.h"
Params::Params()
{
    // mSurfaceParams = new SurfaceParams();
    // mDiffuseParams = new DiffuseParams();
    // mGridParams  =  new GridParams();
    // mConfigParams = new ConfigParams();
}
Params::~Params()
{
    // delete mSurfaceParams;
    // delete mDiffuseParams;
    // delete mGridParams;
    // delete mConfigParams;
}

void Params::setFilname()
{
    mConfigParams.Directory_Param = "";
    mConfigParams.FileName_Param = "config.txt";
    mConfigParams.FileName_Param = mConfigParams.Directory_Param + mConfigParams.FileName_Param;
}

void Params::setParams()
{
    INIReader reader(mConfigParams.FileName_Param);
    if (reader.ParseError() < 0){
       std::cerr << "error in Params.cpp:Can't load file "<< mConfigParams.FileName_Param << std::endl;
    }

    mConfigParams.gpu_id = reader.GetReal("Config", "GpuId", 0);
    mConfigParams.frameStart = reader.GetReal("Config", "Framefrom", 0);
    mConfigParams.frameEnd = reader.GetReal("Config", "Frameto", 10);
    mConfigParams.frameStep = reader.GetReal("Config", "FrameStep", 1);

    mConfigParams.nzeros = reader.GetReal("Config", "NumZeros", 4);

    mConfigParams.fluidPath = reader.Get("Config", "FluidPath", "../flow/particles/");
    mConfigParams.fluidPrefix = reader.Get("Config", "FluidPrefix", "PartFluid_");

    mConfigParams.boundPath = reader.Get("Config", "BoundPath", "../flow/");
    mConfigParams.boundPrefix = reader.Get("Config", "BoundPrefix", "CaseFloating_");

    mConfigParams.surfacePath = reader.Get("Config", "SurfacePath", "../flow/particles/");
    mConfigParams.surfacePrefix = reader.Get("Config", "SurfacePrefix", "PartSurfaceTest_");

    mConfigParams.diffusePath = reader.Get("Config", "DiffusePath", "../flow/particles/");
    mConfigParams.diffusePrefix = reader.Get("Config", "DiffusePrefix", "PartDiffuse_");

    mConfigParams.isDiffuse = reader.GetBoolean("Config", "IsDiffuseGeneration", true);
    mConfigParams.isSurface = reader.GetBoolean("Config", "IsSurfaceReconstruct", false);

    mSimParams.mass = reader.GetReal("Sim", "Mass", 0); 
    mSimParams.particleSpacing = reader.GetReal("Sim", "ParticleSpacing", 0.01);

    //fluid surface reconstruction
	mSurfaceParams.smoothingRadiusRatio = reader.GetReal("Surface", "SmoothingRadiusRatio", 4);

	mSurfaceParams.smoothingRadius = mSurfaceParams.smoothingRadiusRatio*mSimParams.particleSpacing;

	mSurfaceParams.smoothingRadiusInv = 1.0 / mSurfaceParams.smoothingRadius;
	mSurfaceParams.smoothingRadiusSq = mSurfaceParams.smoothingRadius * mSurfaceParams.smoothingRadius;
	mSurfaceParams.anisotropicRadius = mSurfaceParams.smoothingRadius * 2; 

    mSurfaceParams.minNumNeighbors = reader.GetInteger("Surface", "MinNumNeighbors", 8);
    mSurfaceParams.isolateNumNeighbors = reader.GetInteger("Surface", "IsolateNumNeighbors", 4);

	mSurfaceParams.lambdaForSmoothed = reader.GetReal("Surface", "LambdaForSmoothed", 0.5);
    mSurfaceParams.isoValue = reader.GetReal("Surface", "IsoValue", 0.5);

    //diffuse particle generation
    // mDiffuseParams.smoothingRadiusRatio = reader.GetReal("Diffuse", "SmoothingRadiusRatioForDiffuse", 2);
	// mDiffuseParams.smoothingRadius = mDiffuseParams.smoothingRadiusRatio*mSimParams.particleSpacing;
    
    mDiffuseParams.smoothingRadius = 1.732*mSimParams.particleSpacing;

	mDiffuseParams.smoothingRadiusInv = 1.0 / mDiffuseParams.smoothingRadius;
	mDiffuseParams.smoothingRadiusSq = mDiffuseParams.smoothingRadius * mDiffuseParams.smoothingRadius;
    mDiffuseParams.anisotropicRadius = mDiffuseParams.smoothingRadius * 2; 

    mDiffuseParams.coefficient = (4 / 3)*PI*powf((mSimParams.particleSpacing/ 2), 3)
    * 315 / (64 * PI*powf(mDiffuseParams.smoothingRadius, 9));

    mDiffuseParams.minNumNeighbors = reader.GetInteger("Surface", "MinNumNeighbors", 25);
    mDiffuseParams.minWaveCrests = reader.GetReal("Diffuse", "MinWaveCrestsThreshold", 0);
	mDiffuseParams.maxWaveCrests = reader.GetReal("Diffuse", "MaxWaveCrestsThreshold", 100);

	mDiffuseParams.minTrappedAir = reader.GetReal("Diffuse", "MinTrappedAirThreshold", 0);
	mDiffuseParams.maxTrappedAir = reader.GetReal("Diffuse", "MaxTrappedAirThreshold", 100);

	mDiffuseParams.minKineticEnergy = reader.GetReal("Diffuse", "MinKineticEnergyThreshold", 0);
	mDiffuseParams.maxKineticEnergy = reader.GetReal("Diffuse", "MaxKineticEnergyThreshold", 100);

	mDiffuseParams.trappedAirMultiplier = reader.GetReal("Diffuse", "DiffuseTrappedAirMultiplier", 800);
	mDiffuseParams.waveCrestsMultiplier = reader.GetReal("Diffuse", "DiffuseWaveCrestsMultiplier", 4000);

	mDiffuseParams.timeStep = reader.GetReal("Diffuse", "TimeStep", 0.01); 

	mDiffuseParams.lifeTime = reader.GetReal("Diffuse", "LifefimeMultiplier", 10); 

	mDiffuseParams.buoyancyControl = reader.GetReal("Diffuse", "BuoyancyControl", 0.8); 
	mDiffuseParams.dragControl = reader.GetReal("Diffuse", "DragControl", 0.5); 

    //domain
	mGridParams.spScale = reader.GetReal("Domain", "SpatialGridSizeScale", 2);
	mGridParams.scScale = reader.GetReal("Domain", "SclarGridSizeScale", 2);

    mGridParams.sptoscScaleInv = 1.0f / (mGridParams.spScale*mGridParams.scScale);

    // mGridParams.spGridSize =  mSimParams.particleSpacing * 1.732;
    mGridParams.spGridSize =  mSimParams.particleSpacing*1.5;
	mGridParams.scGridSize =  mSimParams.particleSpacing / mGridParams.scScale;

	mGridParams.spexpandExtent = ceil(mSurfaceParams.smoothingRadius / mGridParams.spGridSize);
	mGridParams.scexpandExtent = ceil(mSurfaceParams.smoothingRadius / mGridParams.scGridSize);

    mGridParams.spexpandExtent_diffuse = 
    ceil(mDiffuseParams.smoothingRadius / mGridParams.spGridSize);

    mGridParams.minPos.x = reader.GetReal("Domain", "Min_x", 0);
	mGridParams.minPos.y = reader.GetReal("Domain", "Min_y", 0);
	mGridParams.minPos.z = reader.GetReal("Domain", "Min_z", 0);
	mGridParams.maxPos.x = reader.GetReal("Domain", "Max_x", 3);
	mGridParams.maxPos.y = reader.GetReal("Domain", "Max_y", 3);
	mGridParams.maxPos.z = reader.GetReal("Domain", "Max_z", 3);

    //scsize = r
    mGridParams.scminPos = mGridParams.minPos - mSurfaceParams.smoothingRadius;
    mGridParams.scmaxPos = mGridParams.maxPos + mSurfaceParams.smoothingRadius;

    mGridParams.minPos = mGridParams.minPos - 4*mSurfaceParams.smoothingRadius;
    mGridParams.maxPos = mGridParams.maxPos + 4*mSurfaceParams.smoothingRadius;


    mGridParams.spresolution = make_uint3(
	((mGridParams.maxPos.x - mGridParams.minPos.x) / mGridParams.spGridSize),
	((mGridParams.maxPos.y - mGridParams.minPos.y) / mGridParams.spGridSize),
	((mGridParams.maxPos.z - mGridParams.minPos.z) / mGridParams.spGridSize));

    // uint ratio = uint(mGridParams.spGridSize/mGridParams.scGridSize);
    // mGridParams.scresolution = mGridParams.spresolution * ratio;

    mGridParams.scresolution = make_uint3(
	((mGridParams.scmaxPos.x - mGridParams.scminPos.x) / mGridParams.scGridSize) + 0.5,
	((mGridParams.scmaxPos.y - mGridParams.scminPos.y) / mGridParams.scGridSize) + 0.5,
	((mGridParams.scmaxPos.z - mGridParams.scminPos.z) / mGridParams.scGridSize) + 0.5);

    mGridParams.spSize = 
       mGridParams.spresolution.x * mGridParams.spresolution.y * mGridParams.spresolution.z;
    mGridParams.scSize =  
        mGridParams.scresolution.x *  mGridParams.scresolution.y *  mGridParams.scresolution.z;
}


void Params::printInfo()
{
    // std::cout << "particleMass " << mSimParams.mass << std::endl;
    std::cout << "particleSpacing " << mSimParams.particleSpacing << std::endl;
    if (mConfigParams.isSurface)
    {
        std::cout << "SmoothingRadius " << mSurfaceParams.smoothingRadius << std::endl;
        std::cout << "minNumNeighbors " << mSurfaceParams.minNumNeighbors << std::endl;
        std::cout << "isoValue " << mSurfaceParams.isoValue << std::endl;
    }

    if (mConfigParams.isDiffuse)
    {
        std::cout << "diffuseSmoothingRadius " << mDiffuseParams.smoothingRadius << std::endl;
        std::cout << "minWaveCrests " << mDiffuseParams.minWaveCrests << std::endl;
        std::cout << "maxWaveCrests " << mDiffuseParams.maxWaveCrests << std::endl;
        std::cout << "minTrappedAir " << mDiffuseParams.minTrappedAir << std::endl;
        std::cout << "maxTrappedAir " << mDiffuseParams.maxTrappedAir << std::endl;
        std::cout << "minKineticEnergy " << mDiffuseParams.minKineticEnergy << std::endl;
        std::cout << "maxKineticEnergy " << mDiffuseParams.maxKineticEnergy << std::endl;

        std::cout << "trappedAirMultiplier " << mDiffuseParams.trappedAirMultiplier << std::endl;
        std::cout << "waveCrestsMultiplier " << mDiffuseParams.waveCrestsMultiplier << std::endl;
        std::cout << "lifeTime " << mDiffuseParams.lifeTime << std::endl;

        std::cout << "buoyancyControl " << mDiffuseParams.buoyancyControl << std::endl;
        std::cout << "dragControl " << mDiffuseParams.dragControl << std::endl;
    }

    // std::cout << "posMin " << mGridParams.minPos << std::endl;
    // std::cout << "posMax " << mGridParams.maxPos << std::endl;

    // std::cout << "spextent " << mGridParams.spexpandExtent << std::endl;
    // std::cout << "scextent " << mGridParams.scexpandExtent << std::endl;

    // std::cout << "spextent_diffuse " << mGridParams.spexpandExtent_diffuse << std::endl;

    // std::cout << "spresolution " << mGridParams.spresolution << std::endl;
    // std::cout << "scresolution " << mGridParams.scresolution << std::endl;
    
    // std::cout << "spsize " << mGridParams.spSize << std::endl;
    // std::cout << "scsize " << mGridParams.scSize << std::endl;

    // std::cout << "spGridsize " << mGridParams.spGridSize << std::endl;
    // std::cout << "scGridsize " << mGridParams.scGridSize << std::endl;

    std::cout << "frameFrom " << mConfigParams.frameStart << std::endl;
    std::cout << "frameTo " << mConfigParams.frameEnd << std::endl;
}

void Params::setGPUId()
{
    int gpu_id = mConfigParams.gpu_id;
    checkCudaErrors(cudaSetDevice(gpu_id)); 
}

void Params::printGPUInfo()
{
    int gpu_id = mConfigParams.gpu_id;

    checkCudaErrors(cudaGetDevice(&gpu_id));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, gpu_id);

    std::cout << "Device id is " << gpu_id << std::endl;
    std::cout << "DeviceName is " << deviceProp.name << std::endl;
    std::cout << "totalGlobalMem is(MB) " << deviceProp.totalGlobalMem / 1024 / 1024 << std::endl;
    // std::cout << "sharedMemPerBlock is(KB) " << deviceProp.sharedMemPerBlock / 1024 << std::endl;
    // std::cout << "regsPerBlock is "<< deviceProp.regsPerBlock << std::endl;
    // std::cout << "maxThreadsPerBlock is " << deviceProp.maxThreadsPerBlock << std::endl;
    // std::cout << "major of Compute Capability " << deviceProp.major << "." << deviceProp.minor << std::endl;
    // std::cout << "multiProcessorCount is " << deviceProp.multiProcessorCount << std::endl;
}