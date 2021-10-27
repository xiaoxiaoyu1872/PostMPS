#include <omp.h>

#include "DiffuseGeneration.h"
#include "DiffuseGeneration_kernel.cuh"
#include "Thrust.cuh"

#include <random>

clock_t begin;
clock_t finish;
float duration;

clock_t begin_cla;
clock_t finish_cla;

clock_t total_start;
clock_t total_end;

float getduration()
{
    duration = (float) (finish- begin)/CLOCKS_PER_SEC;
    return duration;
}

float getduration_total()
{
    duration = (float) (total_end- total_start)/CLOCKS_PER_SEC;
    return duration;
}

float getduration_cla()
{
    duration = (float) (finish_cla- begin_cla)/CLOCKS_PER_SEC;
    return duration;
}


DiffuseGeneration::DiffuseGeneration()
{

}


void DiffuseGeneration::Init(GPUmemory *_gMemory, Params *_params, FileData *_fileData)
{
    params = _params;
    gMemory = _gMemory;
	fileData = _fileData;

    OldNumDiffuseParticles = 0;

    constantMemCopy_Sim();
	constantMemCopy_Diffuse();
	constantMemCopy_Grid();

    gMemory->DiffuseAlloFixedMem();
}


DiffuseGeneration::~DiffuseGeneration()
{

}

void DiffuseGeneration::Destory()
{
    gMemory->DiffuseFreeFixedMem();
    gMemory->Memfree_bound();
    gMemory->DiffuseMemFinal();
}


void DiffuseGeneration::constantMemCopy_Sim()
{
	checkCudaErrors(cudaMemcpyToSymbol(dSimParams, &params->mSimParams, sizeof(SimParams)));
}

void DiffuseGeneration::constantMemCopy_Diffuse()
{
	checkCudaErrors(cudaMemcpyToSymbol(dDiffuseParams, &params->mDiffuseParams, sizeof(DiffuseParams)));
}

void DiffuseGeneration::constantMemCopy_Grid()
{
	checkCudaErrors(cudaMemcpyToSymbol(dGridParams, &params->mGridParams, sizeof(GridParams)));
}


void DiffuseGeneration::runsimulation()
{
    record record_single;
    gMemory->DiffuseMemreset();

    processingOfFreesurface(record_single);

    estimatingOfPotention(record_single);

    generatingOfDiffuse(record_single);

    updatingOfDiffuse(record_single);

    deleteAndappendParticles(record_single);
}


void DiffuseGeneration::Fealrunsimulation()
{
    record record_single;
    gMemory->DiffuseMemreset();
//----------------------------------------------------------
    // memallocation_feal();

    // Fealtrappedair();

    // Fealgradient();

    // Fealwavecrests();
//----------------------------------------------------------
    processingOfFreesurface(record_single);

    estimatingOfPotention(record_single);
//----------------------------------------------------------

    records.push_back(record_single);

    gMemory->DiffuseMemFree();
}


void DiffuseGeneration::Fealtrappedair()
{
    dim3 gridDim, blockDim;
    NumParticles = gMemory->NumParticles;

	calcGridDimBlockDim(NumParticles, gridDim, blockDim);
    calculationofTrappedairpotential << < gridDim, blockDim>> > (
		gMemory->dFluidParticle,
		gMemory->dIndexRange,
        gMemory->dDiffusePotential);
    cudaDeviceSynchronize();
}


void DiffuseGeneration::Fealgradient()
{
    dim3 gridDim, blockDim;
    calcGridDimBlockDim(NumParticles, gridDim, blockDim);
    calculationofColorField << < gridDim, blockDim>> > (
		gMemory->dFluidParticle,
		gMemory->dIndexRange,
        gMemory->dColorField,
        gMemory->dIsSurfaceParticle);
	cudaDeviceSynchronize();

    calcGridDimBlockDim(NumParticles, gridDim, blockDim);
    calculationofNormal << < gridDim, blockDim>> > (
		gMemory->dFluidParticle,
		gMemory->dIndexRange,
        gMemory->dColorField,
        gMemory->dIsSurfaceParticle);
	cudaDeviceSynchronize();
}


void DiffuseGeneration::Fealwavecrests()
{
    dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumParticles, gridDim, blockDim);
    calculationofWavecrestsFeal << < gridDim, blockDim>> > (
		gMemory->dFluidParticle,
		gMemory->dIndexRange,
        // gMemory->dThinFeature,
        gMemory->dDiffusePotential,
        gMemory->dColorField);
    cudaDeviceSynchronize();
}


void DiffuseGeneration::memallocation_feal()
{
    gMemory->memallocation_feal();
    checkCudaErrors(cudaMemcpyToSymbol(dNumParticles,
	&gMemory->NumParticles, sizeof(uint)));
}


void DiffuseGeneration::processingOfFreesurface(record& record_single)
{
    extractionOfFreeSurfaceParticles();

    thrustscan_freeparticles();

    memallocation_freeparticles();

    streamcompact_freeparticles();

    transformmatrices_freeparticles();
}

void DiffuseGeneration::estimatingOfPotention(record& record_single)
{
    memallocation_potional();


    calculationOftrappedair();


    calculationOfwavecrests();


    cudaDeviceSynchronize();
}

void DiffuseGeneration::generatingOfDiffuse(record& record_single)
{
    calculationOfnumberofdiffuseparticles();

    thrustscan_diffuseparticles();

    memallocation_diffuseparticles();

    streamcompact_diffuseparticles();

    calculationOfdiffuseposition();

    determinationOfdiffusetype();
}


void DiffuseGeneration::extractionOfFreeSurfaceParticles()
{
    uint spSize = params->mGridParams.spSize;
    dim3 gridDim, blockDim;
	calcGridDimBlockDim(spSize, gridDim, blockDim);

    estimationOfFreeSurfaceParticles <<< gridDim, blockDim >>> (
		gMemory->dSpatialGrid,
        // gMemory->dBoundGrid,
        gMemory->dIsSurfaceParticle,
		gMemory->dIndexRange);
}

void DiffuseGeneration::thrustscan_freeparticles()
{
    // NumFreeSurfaceParticles = ThrustExclusiveScan(
    // gMemory->dNumFreeSurParticleGridScan,
    // gMemory->dNumFreeSurParticleGrid,
    // (uint)params->mGridParams.spSize);

    NumParticles = gMemory->NumParticles;
    NumFreeSurfaceParticles = ThrustExclusiveScan(
    gMemory->dIsSurfaceParticleScan,
    gMemory->dIsSurfaceParticle,
    (uint)NumParticles);


    if (NumFreeSurfaceParticles == 0)
	{
		std::cerr << "No free surface particle detected!\n";
		return;
	}
    gMemory->NumFreeSurfaceParticles = NumFreeSurfaceParticles;

    std::cout << "mNumFreeSurfaceParticles =  " << NumFreeSurfaceParticles << std::endl;

	std::cout << "free surface particles ratio: " <<
	static_cast<double>(NumFreeSurfaceParticles)
		/ gMemory->NumParticles << std::endl;

    record_single.surpar = static_cast<float>(NumFreeSurfaceParticles)
		/ gMemory->NumParticles;

}

void DiffuseGeneration::streamcompact_freeparticles()
{
    dim3 gridDim, blockDim;
    NumParticles = gMemory->NumParticles;
	calcGridDimBlockDim(NumParticles, gridDim, blockDim);

    compactationOfFreeParticles << < gridDim, blockDim>> > (
		gMemory->dIsSurfaceParticle,
		gMemory->dIsSurfaceParticleScan,
		gMemory->dFreeSurfaceParticlesIndex);
	cudaDeviceSynchronize();
}

void DiffuseGeneration::transformmatrices_freeparticles()
{
    dim3 gridDim, blockDim;
    calcGridDimBlockDim(NumFreeSurfaceParticles, gridDim, blockDim);
    calculationofNormal << < gridDim, blockDim>> > (
		gMemory->dFluidParticle,
		gMemory->dIndexRange,
		gMemory->dFreeSurfaceParticlesIndex);
	cudaDeviceSynchronize();
}

void DiffuseGeneration::calculationOfwavecrests()
{
    dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumFreeSurfaceParticles, gridDim, blockDim);
    calculationofWavecrests << < gridDim, blockDim>> > (
		gMemory->dFluidParticle,
		gMemory->dIndexRange,
		gMemory->dFreeSurfaceParticlesIndex,
        gMemory->dDiffusePotential);
    // cudaDeviceSynchronize();
}

void DiffuseGeneration::calculationOftrappedair()
{
    dim3 gridDim, blockDim;
    NumParticles = gMemory->NumParticles;

	calcGridDimBlockDim(NumParticles, gridDim, blockDim);
    calculationofTrappedairpotential << < gridDim, blockDim>> > (
		gMemory->dFluidParticle,
		gMemory->dIndexRange,
        gMemory->dDiffusePotential);
    // cudaDeviceSynchronize();
}


void DiffuseGeneration::calculationOfnumberofdiffuseparticles()
{
    dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumParticles, gridDim, blockDim);

    calculateofNumberofdiffuseparticles << < gridDim, blockDim>> > (
		gMemory->dDiffusePotential,
		gMemory->dNumDiffuseParticle,
        gMemory->dIsDiffuse);
    cudaDeviceSynchronize();
}



void DiffuseGeneration::thrustscan_diffuseparticles()
{

    GeneratedNumDiffuseParticles = ThrustExclusiveScan(
    gMemory->dNumDiffuseParticleScan,
    gMemory->dNumDiffuseParticle,
    (uint)NumParticles);


    NumIsDiffuseParticles = ThrustExclusiveScan(
    gMemory->dIsDiffuseScan,
    gMemory->dIsDiffuse,
    (uint)NumParticles);

    gMemory->GeneratedNumDiffuseParticles = GeneratedNumDiffuseParticles;
    gMemory->NumIsDiffuseParticles = NumIsDiffuseParticles;

    std::cout << "GeneratedNumDiffuseParticles =  " << GeneratedNumDiffuseParticles << std::endl;
    std::cout << "NumIsDiffuseParticles =  " << NumIsDiffuseParticles << std::endl;

    record_single.isdif =  static_cast<float>(NumIsDiffuseParticles)
		/ NumParticles;

    cudaDeviceSynchronize();
}


void DiffuseGeneration::streamcompact_diffuseparticles()
{
    dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumParticles, gridDim, blockDim);
    compactationOfDiffuseParticle << < gridDim, blockDim>> > (
		gMemory->dIsDiffuse,
		gMemory->dIsDiffuseScan,
        gMemory->dDiffuseParticlesIndex);
    cudaDeviceSynchronize();
}



void DiffuseGeneration::calculationOfdiffuseposition()
{
    std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> xunif(0, 1);	//0-1

	std::vector<float> tempRand(GeneratedNumDiffuseParticles * 3);
	for (auto &x : tempRand)
		x = xunif(gen);

	float* dtempRand;

	checkCudaErrors(cudaMalloc((void**)&dtempRand, GeneratedNumDiffuseParticles * 3 * sizeof(float)));

	checkCudaErrors(cudaMemcpy(dtempRand, static_cast<void*>(tempRand.data()),
			GeneratedNumDiffuseParticles * 3 * sizeof(float), cudaMemcpyHostToDevice));

	std::vector<float>().swap(tempRand);

    dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumIsDiffuseParticles, gridDim, blockDim);
    calculateofDiffusePosition << < gridDim, blockDim>> > (
        gMemory->dFluidParticle,
		gMemory->dNumDiffuseParticle,
		gMemory->dNumDiffuseParticleScan,
        gMemory->dDiffuseParticle,
        dtempRand,
        gMemory->dDiffusePotential,
        gMemory->dDiffuseParticlesIndex);
    cudaDeviceSynchronize();

    safeCudaFree((void**)&dtempRand);
}


void DiffuseGeneration::determinationOfdiffusetype()
{
    dim3 gridDim, blockDim;
	calcGridDimBlockDim(GeneratedNumDiffuseParticles, gridDim, blockDim);
    calculateofDiffuseType << < gridDim, blockDim>> > (
		gMemory->dFluidParticle,
        gMemory->dIndexRange,
		gMemory->dDiffuseParticle,
        gMemory->dSpatialGrid);
    cudaDeviceSynchronize();
}

void DiffuseGeneration::updatingOfDiffuse(record& record_single)
{
    if (OldNumDiffuseParticles > 0)
    {

        dim3 gridDim, blockDim;
        calcGridDimBlockDim(OldNumDiffuseParticles, gridDim, blockDim);
        memallocation_olddiffuseparticles();
        updateDiffuseParticle << < gridDim, blockDim>> > (
            gMemory->dFluidParticle,
            gMemory->dIndexRange,
            gMemory->dDiffuseParticle_old,
            gMemory->dSpatialGrid);
        cudaDeviceSynchronize();
    }
//----------------------------------------------------------------------------
}

void DiffuseGeneration::savemiddlefile()
{
    std::string basename = "Diffuse-Ship-";
    basename = basename + std::to_string(params->mDiffuseParams.trappedAirMultiplier);
	std::string path = params->mConfigParams.boundPath + std::string(basename) + ".txt";
    std::ofstream file;
	file.open(path.c_str(), std::ios::out);

    file	<< setw(10) << gMemory->NumParticles << "  "
			<< setw(10) << params->mConfigParams.frameEnd << "  "
			<< std::endl
			;

    float max_point = 0;
    float max_gpu = 0;
    record average;

    for (long i = 0; i < records.size(); i++)
    {
        average.pf += records[i].pf;
        average.po += records[i].po;
        average.gen += records[i].gen;
        average.up += records[i].up;
        average.de += records[i].de;
        average.comde += records[i].comde;
        average.isdif += average.isdif;
        average.numdif += records[i].numdif;
        average.mem_dif += records[i].mem_dif;
        if (max_point <= records[i].numdif)
        {
            max_point = records[i].numdif;
        }
        if (max_gpu <= records[i].mem_dif)
        {
            max_gpu = records[i].mem_dif;
        }
    }

    file << setw(10)<< "average" << " " << std::endl;
    file << setiosflags(ios::fixed)<<setprecision(3)
		 << setw(10)<< (average.pf + average.po + average.gen + average.up + average.de)
		 << setw(10)<< average.pf/records.size() << " "
		 << setw(10)<< average.po/records.size() << " "
		 << setw(10)<< average.gen/records.size() << " "
		 << setw(10)<< average.up/records.size() << " "
		 << setw(10)<< average.de/records.size() << " "
         << setw(10)<< average.comde/records.size() << " "
		 << setw(10)<< average.isdif/records.size() << " "
		 << setprecision(6)
		 << setw(14)<<  (average.numdif)/records.size()<< " "
         << setprecision(3)
         << setw(10)<<  average.mem_dif/records.size() << " "
         << setw(10)<< int(max_point) << " "
         << setw(10)<< max_gpu << " "
		 << std::endl;

    file	<< setw(10)<< "total" << " "
			<< setw(10)<< "pf" << " "
			<< setw(10)<< "po" << " "
			<< setw(10)<< "gen" << " "
			<< setw(10)<< "up" << " "
			<< setw(10)<< "de" << " "
            << setw(10)<< "comde" << " "
			<< setw(10)<< "isdif" << " "
			<< setw(10)<< "numdif" << " "
            << setw(10)<< "mem_dif" << " "
            << setw(10)<< "max_num" << " "
            << setw(10)<< "max_mem" << " "
			<< std::endl;

    for (long i = 0; i < records.size(); i++)
    {
		file << setiosflags(ios::fixed)<<setprecision(3)
			 << setw(10)<< records[i].pf + records[i].po + records[i].gen + records[i].up + records[i].de
			 << setw(10)<< records[i].pf << " "
			 << setw(10)<< records[i].po << " "
			 << setw(10)<< records[i].gen << " "
			 << setw(10)<< records[i].up << " "
			 << setw(10)<< records[i].de << " "
             << setw(10)<< records[i].comde << " "
			 << setw(10)<< records[i].isdif << " "
			 << setprecision(3)
			 << setw(10)<< records[i].numdif << " "
             << setw(10)<< records[i].mem_dif << " "
			 << std::endl;
	}

    // std::vector<FluidParticle> mFluidParticle;
	// mFluidParticle.resize(gMemory->NumParticles);

    // cudaMemcpy(static_cast<void*>(mFluidParticle.data()),
	// gMemory->dFluidParticle, sizeof(FluidParticle) *
	// gMemory->NumParticles, cudaMemcpyDeviceToHost);

    // std::vector<Index> mSurfaceParticlesIndex;
	// mSurfaceParticlesIndex.resize(NumFreeSurfaceParticles);

	// cudaMemcpy(static_cast<void*>(mSurfaceParticlesIndex.data()),
	// gMemory->dFreeSurfaceParticlesIndex, sizeof(Index) *
	// NumFreeSurfaceParticles, cudaMemcpyDeviceToHost);


    // std::vector<DiffusePotential> mDiffusePotential;
	// mDiffusePotential.resize(gMemory->NumParticles);

    // cudaMemcpy(static_cast<void*>(mDiffusePotential.data()),
	// gMemory->dDiffusePotential, sizeof(DiffusePotential) *
	// gMemory->NumParticles, cudaMemcpyDeviceToHost);


    // // //---------------------------free surface particles--------------------------
    // std::vector<DiffuseParticle> mSurfaceParticle;
    // mSurfaceParticle.resize(NumFreeSurfaceParticles);
    // // std::string basename = "TestFreeSur";
	// // std::string path = params->mConfigParams.boundPath + std::string(basename) + ".dat";
    // // std::ofstream file;
    // // file.open(path.c_str(), std::ios::out);

    // for (int i = 0; i < NumFreeSurfaceParticles; i++)
	// {
	// 	int index = mSurfaceParticlesIndex[i];
    //     mSurfaceParticle[i].pos = mFluidParticle[index].pos;
    //     mSurfaceParticle[i].vel = mFluidParticle[index].vel;

	// 	// file<< mFluidParticle[index].pos << " "
	// 	// 	<< mFluidParticle[index].nor << " "
	// 	// 	<< index
	// 	// 	<< std::endl;
	// }

    // fileData->saveDiffuseVTKfile(mSurfaceParticle, 3);

    // // //---------------------------particle potential-----------------------------
    // // std::vector<DiffuseParticle> mSurfaceParticle;
    // mSurfaceParticle.resize(NumParticles);

    // for (int i = 0; i < NumParticles; i++)
	// {
    //     mSurfaceParticle[i].pos = mFluidParticle[i].pos;
    //     mSurfaceParticle[i].vel = mFluidParticle[i].vel;
    //     // mSurfaceParticle[i].nor= mFluidParticle[i].nor;
    //     mSurfaceParticle[i].energy = mDiffusePotential[i].energy;
    //     mSurfaceParticle[i].Ita = mDiffusePotential[i].Ita;
    //     mSurfaceParticle[i].waveCrest = mDiffusePotential[i].waveCrest;

	// }
    // // fileData->saveDiffuseVTKfile(mSurfaceParticle, 4);

    // ConfigParams mConfigParams = params->mConfigParams;
    // int frameIndex = fileData->frameIndex;

    // std::string basename = "TestPotional";
    // std::string seqnum(mConfigParams.nzeros, '0');
    // std::string formats = std::string("%.") + std::to_string(mConfigParams.nzeros) + std::string("d");
    // std::sprintf(&seqnum[0], formats.c_str(), frameIndex);

	// std::string path = params->mConfigParams.boundPath + std::string(basename)+ seqnum +  ".dat";
    // std::ofstream file;
    // file.open(path.c_str(), std::ios::out);

    // for (int i = 0; i < NumParticles; i++)
	// {
    //     int index = i;
	// 	file << mFluidParticle[index].pos << " "
    //         //  << mFluidParticle[index].nor << " "
    //          << mSurfaceParticle[index].energy << " "
    //          << mSurfaceParticle[index].Ita << " "
    //          << mSurfaceParticle[index].waveCrest << " "
	// 		 << std::endl;
	// }


    // for (int i = 0; i < NumFreeSurfaceParticles; i++)
	// {
	// 	int index = mSurfaceParticlesIndex[i];
    //     file << mFluidParticle[index].pos << ' '
    //          << mFluidParticle[index].nor <<' '
    //          << mSurfaceParticle[index].waveCrest << std::endl;
	// }

}


void DiffuseGeneration::savefealfile()
{
    // std::string basename = "Feal-Ship-feal-";
    // basename = basename + std::to_string(params->mDiffuseParams.trappedAirMultiplier);
	// std::string path = params->mConfigParams.boundPath + std::string(basename) + ".txt";
    // std::ofstream file;
	// file.open(path.c_str(), std::ios::out);

    // file	<< setw(10) << gMemory->NumParticles << "  "
	// 		<< setw(10) << params->mConfigParams.frameEnd << "  "
	// 		<< std::endl
	// 		;

    // float max_gpu = 0;
    // record average;

    // for (long i = 0; i < records.size(); i++)
    // {
    //     average.feal += records[i].feal;
    //     average.surpar += records[i].surpar;

    //     if (max_gpu <= records[i].mem_feal)
    //     {
    //         max_gpu = records[i].mem_feal;
    //     }
    // }

    // file << setw(10)<< "average" << " " << std::endl;
    // file << setiosflags(ios::fixed)<<setprecision(3)
	// 	 << setw(10)<< average.feal/records.size() << " "
    //      << setw(10)<< average.surpar/records.size() << " "
    //      << setw(10)<< max_gpu << " "
	// 	 << std::endl;

    // file	<< setw(10)<< "time" << " "
    //         << setw(10)<< "surpar" << " "
    //         << setw(10)<< "max_mem" << " "
	// 		<< std::endl;

    // for (long i = 0; i < records.size(); i++)
    // {
	// 	file << setiosflags(ios::fixed)<<setprecision(3)
	// 		 << setw(10)<< records[i].feal
    //          << setw(10)<< records[i].surpar
	// 		 << setprecision(3)
    //          << setw(10)<< records[i].mem_feal << " "
	// 		 << std::endl;
	// }

    std::string basename = "Feal-RT-class-";
    basename = basename + std::to_string(params->mDiffuseParams.trappedAirMultiplier);
	std::string path = params->mConfigParams.boundPath + std::string(basename) + ".txt";
    std::ofstream file;
	file.open(path.c_str(), std::ios::out);

    file	<< setw(10) << gMemory->NumParticles << "  "
			<< setw(10) << params->mConfigParams.frameEnd << "  "
            << setw(10) << params->mDiffuseParams.trappedAirMultiplier << "  "
			<< std::endl
			;

    record average;
    average.our = 0;
    average.im12 = 0;
    average.mu03 = 0;
    average.total = 0;
    average.isdif = 0;
    average.de = 0;
    average.temp = 0;
    average.po = 0;
    average.gen = 0;
    average.feal = 0;
    int max_point = 0;
    float max_mem = 0;

    for (long i = 0; i < records.size(); i++)
    {
        average.our += records[i].our;
        average.mu03 += records[i].mu03;
        average.im12 += records[i].im12;
        average.total +=  records[i].total;
        average.isdif +=  records[i].isdif;
        average.de +=  records[i].de;
        average.temp +=  records[i].temp;
        average.po += records[i].po;
        average.feal += records[i].feal;
        average.gen += records[i].gen;

        if (max_point <= records[i].numdif)
        {
            max_point = records[i].numdif;
        }
        if (max_mem <= records[i].mem_dif)
        {
            max_mem = records[i].mem_dif;
        }
    }

    file << setw(14)<< "average" << " " << std::endl;
    file << setiosflags(ios::fixed)<<setprecision(6)
         << setw(14)<<
        average.total/records.size() - average.our/records.size() - average.mu03/records.size() - average.im12/records.size() - average.im12/records.size()
        + average.our/records.size() - 2*average.de/records.size()/3 - average.feal/records.size() - average.temp/records.size()
        << " "
		 << setw(14)<< average.our/records.size()<< " "
         << setw(14)<< average.mu03/records.size()<< " "
         << setw(14)<< average.im12/records.size() << " "
         << setw(14)<< average.isdif/records.size() << " "
         << setw(14)<< max_point << " "
         << setw(14)<< max_mem << " "
         << setw(14)<< average.po/records.size()<< " "
         << setw(14)<< average.feal/records.size()<< " "
         << setw(14)<<
         average.gen/records.size() - average.mu03/records.size() - 2*average.im12/records.size()
         << " "
         << setw(14)<< average.de/records.size()<< " "
		 << std::endl;

    file	<< setw(14)<< "total" << " "
            << setw(14)<< "our" << " "
            << setw(14)<< "mu03" << " "
            << setw(14)<< "im12" << " "
            << setw(14)<< "isdif" << " "
            << setw(14)<< "max_point" << " "
            << setw(14)<< "max_mem" << " "
            << setw(14)<< "poten" << " "
            << setw(14)<< "feal" << " "
            << setw(14)<< "gen" << " "
            << setw(14)<< "de" << " "
			<< std::endl;
}


void DiffuseGeneration::deleteAndappendParticles(record& record_single)
{
    std::vector<DiffuseParticle> gDiffuseParticle;
    std::vector<DiffuseParticle> oDiffuseParticle;

    std::vector<DiffuseParticle> mSpary;
    std::vector<DiffuseParticle> mFoam;
    std::vector<DiffuseParticle> mBubble;

    uint mNumTotalDiffuseParticle = 0;

    uint mNumSparyParticle = 0;
    uint mNumFoamParticle = 0;
    uint mNumBubbleParticle = 0;


    mDiffuse.clear();
    //--------------------------------------------------------------------------------------------------------

	gDiffuseParticle.resize(GeneratedNumDiffuseParticles);
	cudaMemcpy(static_cast<void*>(gDiffuseParticle.data()),
	gMemory->dDiffuseParticle, sizeof(DiffuseParticle) * GeneratedNumDiffuseParticles, cudaMemcpyDeviceToHost);

    oDiffuseParticle.resize(OldNumDiffuseParticles);
    if (OldNumDiffuseParticles)
    {
        cudaMemcpy(static_cast<void*>(oDiffuseParticle.data()),
	    gMemory->dDiffuseParticle_old, sizeof(DiffuseParticle) * OldNumDiffuseParticles, cudaMemcpyDeviceToHost);
    }

	std::copy(oDiffuseParticle.begin(), oDiffuseParticle.end(), std::back_inserter(gDiffuseParticle));

    mDiffuse.resize(GeneratedNumDiffuseParticles + OldNumDiffuseParticles);
    mSpary.resize(GeneratedNumDiffuseParticles + OldNumDiffuseParticles);
    mFoam.resize(GeneratedNumDiffuseParticles + OldNumDiffuseParticles);
    mBubble.resize(GeneratedNumDiffuseParticles + OldNumDiffuseParticles);

	int before = gDiffuseParticle.size();

	for (int i = 0; i < gDiffuseParticle.size(); ++i)
	{
		if (gDiffuseParticle[i].type == 3)
		{
			continue;
		}

		if (gDiffuseParticle[i].type == 0)
		{
			mSpary[mNumSparyParticle] = gDiffuseParticle[i];
            mNumSparyParticle++;
		}
		else if(gDiffuseParticle[i].type == 2)
		{
			mBubble[mNumBubbleParticle] = gDiffuseParticle[i];
            mNumBubbleParticle++;
		}
		else
		{
			mFoam[mNumFoamParticle] = gDiffuseParticle[i];
            mNumFoamParticle++;
		}
		mDiffuse[mNumTotalDiffuseParticle] = gDiffuseParticle[i];
		mNumTotalDiffuseParticle++;
	}

    mSpary.resize(mNumSparyParticle);
    mFoam.resize(mNumFoamParticle);
    mBubble.resize(mNumBubbleParticle);

    std::cout << "mSpary = "<< mSpary.size() << std::endl;
    std::cout << "mBubble = "<< mBubble.size() << std::endl;
    std::cout << "mFoam = "<< mFoam.size() << std::endl;

	int after = mNumTotalDiffuseParticle;
    mDiffuse.resize(mNumTotalDiffuseParticle);

    record_single.numdif = mNumTotalDiffuseParticle;

	OldNumDiffuseParticles = mNumTotalDiffuseParticle;
    gMemory->OldNumDiffuseParticles = mNumTotalDiffuseParticle;

    std::cout << "deleted: " << before - after << std::endl;
	std::cout << "generated = "<< GeneratedNumDiffuseParticles << std::endl;
    std::cout << "total = "<< mNumTotalDiffuseParticle << std::endl;

    gMemory->memallocation_olddiffuseparticles(mDiffuse);

    fileData->saveDiffuseVTKfile(mSpary, 0);
    fileData->saveDiffuseVTKfile(mFoam, 1);
    fileData->saveDiffuseVTKfile(mBubble, 2);

    std::vector<DiffuseParticle>().swap(gDiffuseParticle);
    std::vector<DiffuseParticle>().swap(oDiffuseParticle);

    std::vector<DiffuseParticle>().swap(mSpary);
    std::vector<DiffuseParticle>().swap(mFoam);
    std::vector<DiffuseParticle>().swap(mBubble);

//--------------------------------------------------------------------------------------------------------------
//     std::cout << "GeneratedNum = "<< GeneratedNumDiffuseParticles << std::endl;
//     std::cout << "OldNum = "<< OldNumDiffuseParticles << std::endl;
//     uint CombineNum = GeneratedNumDiffuseParticles + OldNumDiffuseParticles;

//     mDiffuse.resize(CombineNum);

//     cudaMemcpy(static_cast<void*>(mDiffuse.data()),
// 	gMemory->dDiffuseParticle, sizeof(DiffuseParticle) * GeneratedNumDiffuseParticles, cudaMemcpyDeviceToHost);

//     if (OldNumDiffuseParticles)
//     {
//         cudaMemcpy(static_cast<void*>(mDiffuse.data() + GeneratedNumDiffuseParticles),
// 	    gMemory->dDiffuseParticle_old, sizeof(DiffuseParticle) * OldNumDiffuseParticles, cudaMemcpyDeviceToHost);
//     }

//     std::vector<int> CombineType;
//     CombineType.resize(CombineNum);
//     for (int i = 0; i < CombineNum; i++)
//     {
//         CombineType[i] = mDiffuse[i].type;
//     }

//     int* d_KeyType;
//     cudaMalloc((void**)&d_KeyType, sizeof(int)* CombineNum);
//     cudaMemcpy(d_KeyType,
// 	    static_cast<void*>(CombineType.data()), sizeof(int) * CombineNum, cudaMemcpyHostToDevice);

//     // DiffuseParticle temp;
//     // temp.type = 4;
//     // mDiffuse.push_back(temp);//补充位

//     safeCudaFree((void**)&gMemory->dDiffuseParticle_old);
// 	cudaMallocMemset((void**)&gMemory->dDiffuseParticle_old, 0.0f, mDiffuse.size() *sizeof(DiffuseParticle));

// 	checkCudaErrors(cudaMemcpy(gMemory->dDiffuseParticle_old, static_cast<void*>(mDiffuse.data()),
// 			mDiffuse.size() * sizeof(DiffuseParticle), cudaMemcpyHostToDevice));
// //--------------------------------------------------------------------------------------------------------------
// begin = clock();


//     if(CombineNum > 1)
//         ThrustSort(gMemory->dDiffuseParticle_old, d_KeyType, CombineNum);

//     // int* d_count;
//     // cudaMallocMemset((void**)&d_count, 0, 4*sizeof(int));

//     // uint diffuseNum = mDiffuse.size();
//     // dim3 gridDim, blockDim;
// 	// calcGridDimBlockDim(diffuseNum, gridDim, blockDim);
//     // countnum<<< gridDim, blockDim >>>(gMemory->dDiffuseParticle_old, d_count, diffuseNum);
//     // cudaDeviceSynchronize();

//     // std::vector<int> count;
//     // count.resize(4);
//     // checkCudaErrors(cudaMemcpy(static_cast<void*>(count.data()), d_count, 4*sizeof(int), cudaMemcpyDeviceToHost));

//     // mNumSparyParticle = count[0];
//     // mNumFoamParticle = count[1] - count[0];
//     // mNumBubbleParticle = count[2] - count[1];

//     thrust::device_vector<int> input(CombineType.data(), CombineType.data() + CombineNum);
//   	mNumSparyParticle = thrust::count(input.begin(), input.end(), 0);
//     mNumFoamParticle = thrust::count(input.begin(), input.end(), 1);
//     mNumBubbleParticle = thrust::count(input.begin(), input.end(), 2);
//     // int mDeleteNum = thrust::count(input.begin(), input.end(), 3);

//     mNumTotalDiffuseParticle = mNumSparyParticle + mNumFoamParticle + mNumBubbleParticle ;
//     OldNumDiffuseParticles = mNumTotalDiffuseParticle;
//     gMemory->OldNumDiffuseParticles = mNumTotalDiffuseParticle;

//     record_single.numdif = mNumTotalDiffuseParticle;

//     cudaMemcpy(static_cast<void*>(mDiffuse.data()),
// 	    gMemory->dDiffuseParticle_old, sizeof(DiffuseParticle) * mNumTotalDiffuseParticle, cudaMemcpyDeviceToHost);

//     mSpary.resize(mNumSparyParticle);
//     mFoam.resize(mNumFoamParticle);
//     mBubble.resize(mNumBubbleParticle);

// #pragma omp parallel for schedule(guided)
//     for (int i = 0, j = 0; i < mNumSparyParticle; i++)
//     {
//         mSpary[j] = mDiffuse[i];
//         j++;
//     }
// #pragma omp parallel for schedule(guided)
//     for (int i = mNumSparyParticle, j = 0; i < mNumSparyParticle + mNumFoamParticle; i++)
//     {
//         mFoam[j] = mDiffuse[i];
//         j++;
//     }
// #pragma omp parallel for schedule(guided)
//     for (int i = mNumSparyParticle + mNumFoamParticle, j = 0; i < mNumTotalDiffuseParticle; i++)
//     {
//         mBubble[j] = mDiffuse[i];
//         j++;
//     }


// finish = clock();
//     record_single.de = getduration();
//     std::cout << "deleteAndappendParticles time consuming = " << record_single.de << std::endl;


//     safeCudaFree((void**)&d_KeyType);
//     // safeCudaFree((void**)&d_count);

//     // int mDeleteNum = diffuseNum - count[2];
//     int mDeleteNum = thrust::count(input.begin(), input.end(), 3);
//     std::cout << "deleted: " << mDeleteNum << std::endl;
//     std::cout << "generated = "<< GeneratedNumDiffuseParticles << std::endl;
//     std::cout << "total = "<< mNumTotalDiffuseParticle << std::endl;

//     std::cout << "mSpary = "<< mSpary.size() << std::endl;
//     std::cout << "mBubble = "<< mBubble.size() << std::endl;
//     std::cout << "mFoam = "<< mFoam.size() << std::endl;

//     // fileData->saveDiffuseVTKfile(mSpary, 0);
//     // fileData->saveDiffuseVTKfile(mFoam, 1);
//     // fileData->saveDiffuseVTKfile(mBubble, 2);

//     std::vector<DiffuseParticle>().swap(mSpary);
//     std::vector<DiffuseParticle>().swap(mFoam);
//     std::vector<DiffuseParticle>().swap(mBubble);
// --------------------------------------------------------------------------------------------------------------
}


void DiffuseGeneration::memallocation_freeparticles()
{
    gMemory->memallocation_freeparticles();
    checkCudaErrors(cudaMemcpyToSymbol(dNumParticles,
	&gMemory->NumParticles, sizeof(uint)));
    checkCudaErrors(cudaMemcpyToSymbol(dNumFreeSurfaceParticles,
	&gMemory->NumFreeSurfaceParticles, sizeof(uint)));
}

void DiffuseGeneration::memallocation_potional()
{
    gMemory->memallocation_potional();
}

void DiffuseGeneration::memallocation_diffuseparticles()
{
    gMemory->memallocation_diffuseparticles();
    checkCudaErrors(cudaMemcpyToSymbol(dGeneratedNumDiffuseParticles,
	&gMemory->GeneratedNumDiffuseParticles, sizeof(uint)));
    checkCudaErrors(cudaMemcpyToSymbol(dNumIsDiffuseParticles,
	&gMemory->NumIsDiffuseParticles, sizeof(uint)));
}

void DiffuseGeneration::memallocation_olddiffuseparticles()
{
    // gMemory->memallocation_olddiffuseparticles(mDiffuse);
    checkCudaErrors(cudaMemcpyToSymbol(dOldNumDiffuseParticles,
	&gMemory->OldNumDiffuseParticles, sizeof(uint)));
}
