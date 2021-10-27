#include "SurfaceReconstruction.h"
#include "Thrust.cuh"
clock_t start, end;

SurfaceReconstruction::SurfaceReconstruction()
{   

}

void SurfaceReconstruction::Init(GPUmemory *_gMemory, Params *_params, FileData *_fileData)
{
	params = _params;
    gMemory = _gMemory;
	fileData = _fileData;

	anisotropicMethod = new Anisotropic(gMemory, params);
	marchingCube = new MarchingCube(gMemory, params);
}


SurfaceReconstruction::~SurfaceReconstruction()
{   
	std::cout << "~SurfaceReconstruction" << std::endl;
}


void SurfaceReconstruction::Destory()
{
	delete anisotropicMethod;
	delete marchingCube;
}


void SurfaceReconstruction::runsimulation()
{
	record* record_single = new record;
	gMemory->SurfaceMemreset();

	start = clock();
	anisotropicMethod->processingOfParticles(record_single);
	anisotropicMethod->processingOfVertices(record_single);
	end = clock();
	record_single->pp = (double)(end - start) / CLOCKS_PER_SEC;

	start = clock();
	anisotropicMethod->estimationOfscalarField();
	end = clock();
	record_single->sf = (double)(end - start) / CLOCKS_PER_SEC;

	start = clock();
	marchingCube->triangulation(record_single);
	end = clock();
	record_single->tri = (double)(end - start) / CLOCKS_PER_SEC;

	// fileData->saveSurfaceParticleVTKfile();
	// fileData->saveFlagVTKfile();
	// fileData->saveSurfaceVTKfile();
	fileData->saveSurfacePLYfile();
	records.push_back(record_single);
}


void SurfaceReconstruction::saveMiddleFile()
{			
	std::string basename = "Surface-WC-2r";
	std::string path = params->mConfigParams.boundPath + std::string(basename) + ".txt";
    std::ofstream file;

	file.open(path.c_str(), std::ios::out);

	file	<< setw(10) << gMemory->NumParticles << "  "
			<< setw(10) << params->mGridParams.scresolution << "  "
			<< std::endl
			;

	record average;

	float max = 0;
	float max_mc = 0;

	average.pp = 0;
	average.sf = 0;
	average.tri = 0;
	average.surpar = 0;
	average.surver = 0;
	average.surcell = 0;
	average.trinum = 0;

	for (long i = 0; i < records.size(); i++) 
    {
		average.pp += records[i]->pp;
		average.sf += records[i]->sf;
		average.tri += records[i]->tri;

		average.surpar += records[i]->surpar;
		average.surver += records[i]->surver;
		average.surcell += records[i]->surcell;

		average.trinum += records[i]->trinum/1000;

		if (records[i]->mem >= max)
		{
			max = records[i]->mem;
		}
		
		if (records[i]->mem_mc >= max_mc)
		{
			max_mc = records[i]->mem_mc;
		}

	}
	
	average.mem = max;
	average.mem_mc = max_mc;

	file << setw(10)<< "average" << " " << std::endl;

	file << setiosflags(ios::fixed)<<setprecision(3) 
		 << setw(10)<< (average.pp + average.sf + average.tri)/records.size() 
		 << setw(10)<< average.pp/records.size() << " "
		 << setw(10)<< average.sf/records.size() << " "
		 << setw(10)<< average.tri/records.size() << " "
		 << setw(10)<< average.surpar/records.size() << " "
		 << setw(10)<< average.surver/records.size() << " "
		 << setw(10)<< average.surcell/records.size() << " "
		 << setprecision(3) 
		 << setw(10)<< average.trinum/records.size() << " "
		 << setprecision(3) 
		 << setw(10)<< average.mem << " "
		 << setprecision(3) 
		 << setw(10)<< average.mem_mc << " "
		 << std::endl;


	file	<< setw(10)<< "total" << " "
			<< setw(10)<< "pp" << " "
			<< setw(10)<< "sf" << " "
			<< setw(10)<< "tri" << " "
			<< setw(10)<< "surpar" << " "
			<< setw(10)<< "surver" << " "
			<< setw(10)<< "surcell" << " "
			<< setw(10)<< "trinum(k)" << " "
			<< setw(10)<< "mem(Gb)" << " "
			<< setw(10)<< "mem_mc(Gb)" << " "
			<< std::endl;


	for (long i = 0; i < records.size(); i++) 
    {
		file << setiosflags(ios::fixed)<<setprecision(3) 
			 << setw(10)<< records[i]->pp + records[i]->sf + records[i]->tri 
			 << setw(10)<< records[i]->pp << " "
			 << setw(10)<< records[i]->sf << " "
			 << setw(10)<< records[i]->tri << " "
			 << setw(10)<< records[i]->surpar << " "
			 << setw(10)<< records[i]->surver << " "
			 << setw(10)<< records[i]->surcell << " "
			 << setprecision(3) 
			 << setw(10)<< records[i]->trinum/1000 << " "
			 << setprecision(3) 
			 << setw(10)<< records[i]->mem << " "
			 << setw(10)<< records[i]->mem_mc << " "
			 << std::endl;
	}
    
	//---------------------------sur particles index--------------------------

	// std::string basename = "TestSurfaceIndex";
	// std::string path = params->mConfigParams.boundPath + std::string(basename) + ".dat";
    // std::ofstream file;
	// file.open(path.c_str(), std::ios::out);

	// for (int i = 0; i < NumSurfaceParticles; i++)
	// {
	// 	int index = mSurfaceParticlesIndex[i];
	// 	file<< mFluidParticle[index].pos << " "
	// 		<< mFluidParticle[index].nor << " "
	// 		<< index 
	// 		<< std::endl;
	// }
	
	//---------------------------sur particles index--------------------------

	// std::string basename = "TestInvIndex";
	// std::string path = params->mConfigParams.boundPath + std::string(basename) + ".dat";
    // std::ofstream file;
	// file.open(path.c_str(), std::ios::out);

	// for (int i = 0; i < NumInvolveParticles; i++)
	// {
	// 	int index = mInvovleParticlesIndex[i];
	// 	file      << mFluidParticle[index].pos << " "
	// 			  // << mFluidParticle[index].nor << " "
	// 			//   << index 
	// 			  << std::endl;
	// }

	// for (long i = 0; i < params->mGridParams.spSize; i++) 
    // {

	// 		if (mNumInvParticleGrid[i] == 0)
	// 		{
	// 			continue;
	// 		}
			
	// 		for (long index = mIndexRange[i].start; index < mIndexRange[i].end; index++)
	// 		{
	// 			file << mFluidParticle[index].pos << ' '
	// 				//  << mFluidParticle[index].nor << ' '
	// 				 << std::endl;
	// 		}
	// }


	//---------------------------mean particles--------------------------

	// std::string basename = "TestMean";
	// std::string path = params->mConfigParams.boundPath + std::string(basename) + ".dat";
    // std::ofstream file;
	// file.open(path.c_str(), std::ios::out);

	// for (int i = 0; i < NumSurfaceParticles; i++)
	// {
	// 	file << mMeanParticle[i].pos << std::endl;
	// }

	//---------------------------smoothed particles--------------------------

	// std::string basename = "TestSmoothed";
	// std::string path = params->mConfigParams.boundPath + std::string(basename) + ".dat";
    // std::ofstream file;
	// file.open(path.c_str(), std::ios::out);

	// for (int i = 0; i < NumSurfaceParticles; i++)
	// {
	// 	file << mSmoothedParticle[i].pos << std::endl;
	// }

	//---------------------------smoothed particles--------------------------

	// std::string basename = "TestSmoothedInv";
	// std::string path = params->mConfigParams.boundPath + std::string(basename) + ".dat";
    // std::ofstream file;
	// file.open(path.c_str(), std::ios::out);

	// for (int i = 0; i < NumInvolveParticles; i++)
	// {
	// 	file << mSmoothedInvParticle[i].pos << std::endl;
	// }

	//---------------------------extraction vertices--------------------------

	// std::string basename = "TestVer";
	// std::string path = params->mConfigParams.boundPath + std::string(basename) + ".dat";
    // std::ofstream file;
	// file.open(path.c_str(), std::ios::out);
	// for (long i = 0; i < params->mGridParams.scSize; i++) 
    // {

	// 	if (mIsSurfaceVertices[i] == 0)
	// 	{
	// 		continue;
	// 	}
	// 	float3 vPos = getVertexPos(index1DTo3D(i, params->mGridParams.scresolution),
	// 		params->mGridParams.minPos, params->mGridParams.scGridSize);

	// 	file << vPos << std::endl;
    // }

	//---------------------------vertices compaction--------------------------

	// std::string basename = "TestVerCom_4";
	// std::string path = params->mConfigParams.boundPath + std::string(basename) + ".dat";
    // std::ofstream file;
	// file.open(path.c_str(), std::ios::out);
	// for (long i = 0; i < NumSurfaceVertices; i++) 
    // {
	// 	uint Index = mSurfaceVerticeIndex[i];
	// 	float3 vPos = getVertexPos(index1DTo3D(Index, params->mGridParams.scresolution),
	// 		params->mGridParams.minPos, params->mGridParams.scGridSize);
	// 	file << vPos << ' '<< mScalarFiled[Index] <<
	// 	std::endl;
    // }
}

