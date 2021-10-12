#include "MarchingCube.h"
#include "MarchingCube_kernel.cu"

#include "MarchingCubesCPU.h"
#include "Thrust.cuh"

MarchingCube::MarchingCube(GPUmemory *_gMemory, Params *_params)
{
    params = _params;
    gMemory = _gMemory;

    gMemory->AlloTextureMem();
    BindingTexMem();
    // constantMemCopy();
}


MarchingCube::~MarchingCube()
{
    gMemory->FreeTextureMem();
    std::cout << "~~MarchingCube" << std::endl;
}


void MarchingCube::constantMemCopy()
{
    checkCudaErrors(cudaMemcpyToSymbol(dSurfaceParams, &params->mSurfaceParams, sizeof(SurfaceParams)));
	checkCudaErrors(cudaMemcpyToSymbol(dGridParams, &params->mGridParams, sizeof(GridParams)));
}


void MarchingCube::triangulation(record* _record)
{
	record_single = _record;
	constantMemCopy();
//-----------------------------------------------	
    memallocation_cubes();

	detectionOfValidCubes();

	thrustscan_cubes();

	memallocation_triangles();

	streamcompact_cubes();

	marchingcubes();

//-----------------------------------------------	
	memallocation_scalarvalue();
	
	scalarvalue();
//-----------------------------------------------
	// constantMemSurVer_Num();
	// gMemory->memAllocation_cubes_cpu();

	// NumSurfaceVertices = gMemory->NumSurfaceVertices;
	// dim3 gridDim, blockDim;
	// calcGridDimBlockDim(NumSurfaceVertices, gridDim, blockDim);
	// detectValidSurfaceCubes << <gridDim, blockDim >> > (
	// 	gMemory->dSurfaceVerticesIndex,
	// 	gMemory->dScalarFiled,
	// 	gMemory->dIsSurfaceVertices,
	// 	gMemory->dIsValidSurfaceCube);
	// cudaDeviceSynchronize();

	// generationOfSurfaceMeshUsingMCForAkinci();
}







void MarchingCube::detectionOfValidCubes()
{
	NumSurfaceVertices = gMemory->NumSurfaceVertices;
	dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumSurfaceVertices, gridDim, blockDim);
	detectValidSurfaceCubes << <gridDim, blockDim >> > (
		gMemory->dSurfaceVerticesIndex,
		gMemory->dScalarFiled,
		gMemory->dIsSurfaceVertices,
		gMemory->dCubeFlag,
		gMemory->dIsValidSurfaceCube,
		gMemory->dNumVertexCube);
	cudaDeviceSynchronize();
}

void MarchingCube::thrustscan_cubes()
{
	NumValidSurfaceCubes = ThrustExclusiveScan(
		gMemory->dIsValidSurfaceCubeScan,
		gMemory->dIsValidSurfaceCube,
		(uint)NumSurfaceVertices);

	NumSurfaceMeshVertices = ThrustExclusiveScan(
		gMemory->dNumVertexCubeScan,
		gMemory->dNumVertexCube,
		(uint)NumSurfaceVertices);

	gMemory->NumValidSurfaceCubes = NumValidSurfaceCubes;
	gMemory->NumSurfaceMeshVertices = NumSurfaceMeshVertices;

	if (NumValidSurfaceCubes <= 0)
	{
		std::cerr << "No vertex of surface mesh detected!\n";
		return;
	}

	if (NumSurfaceMeshVertices <= 0)
	{
		std::cerr << "No vertex of surface mesh detected!\n";
		return;
	}

	record_single->surcell = static_cast<float>(NumValidSurfaceCubes)
	/ (params->mGridParams.scSize);

	record_single->trinum = static_cast<float>(NumSurfaceMeshVertices)/3;
}


void MarchingCube::streamcompact_cubes()
{
	dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumSurfaceVertices, gridDim, blockDim);

	compactValidSurfaceCubes << <gridDim, blockDim >> > (
		gMemory->dValidCubesIndex,
		gMemory->dIsValidSurfaceCube,
		gMemory->dIsValidSurfaceCubeScan);
	cudaDeviceSynchronize();
}

void MarchingCube::marchingcubes()
{
	dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumValidSurfaceCubes, gridDim, blockDim);

	generateTriangles<< <gridDim, blockDim >> >(
		gMemory->dSurfaceVerticesIndex,
		gMemory->dValidCubesIndex,
		gMemory->dCubeFlag,
		gMemory->dNumVertexCubeScan,
		gMemory->dScalarFiled,
		gMemory->dVertex,
		gMemory->dNormal);
	cudaDeviceSynchronize();
}

void MarchingCube::scalarvalue()
{
	dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumSurfaceMeshVertices, gridDim, blockDim);

	estimationOfScalarValue<< <gridDim, blockDim >> >(
		gMemory->dVertex,
		gMemory->dFluidParticle,
		gMemory->dIndexRange,
		gMemory->dScalarValue);
	cudaDeviceSynchronize();
}

void MarchingCube::memallocation_cubes()
{
	constantMemSurVer_Num();
	gMemory->memAllocation_cubes();
}


void MarchingCube::memallocation_triangles()
{
	gMemory->memAllocation_triangles();
	constantMemValCube_Num();
}


void MarchingCube::memallocation_scalarvalue()
{
	gMemory->memAllocation_scalarvalues();
}

void MarchingCube::constantMemSurVer_Num()
{
	checkCudaErrors(cudaMemcpyToSymbol(dNumSurfaceVertices, 
	&gMemory->NumSurfaceVertices, sizeof(uint)));
}

void MarchingCube::constantMemValCube_Num()
{
	checkCudaErrors(cudaMemcpyToSymbol(dNumValidSurfaceCubes, 
	&gMemory->NumValidSurfaceCubes, sizeof(uint)));
	checkCudaErrors(cudaMemcpyToSymbol(dNumSurfaceMeshVertices, 
	&gMemory->NumSurfaceMeshVertices, sizeof(uint)));
}

void MarchingCube::BindingTexMem()
{
	cudaChannelFormatDesc channelDescUnsigned =
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaChannelFormatDesc channelDescSigned =
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);

	checkCudaErrors(cudaBindTexture(0, edgeTex, gMemory->dEdgeTable, channelDescUnsigned));
	checkCudaErrors(cudaBindTexture(0, edgeIndexesOfTriangleTex, gMemory->dEdgeIndicesOfTriangleTable, channelDescSigned));
	checkCudaErrors(cudaBindTexture(0, numVerticesTex, gMemory->dNumVerticesTable, channelDescUnsigned));
	checkCudaErrors(cudaBindTexture(0, vertexIndexesOfEdgeTex, gMemory->dVertexIndicesOfEdgeTable, channelDescUnsigned));  
}


void MarchingCube::generationOfSurfaceMeshUsingMCForAkinci()
{
	uint mNumValidSurfaceCubes = 0;

	//! copy valid cubes' indices from gpu.
	std::vector<uint> validVerticesIndexArray;
	std::vector<IsValid> isValidSurfaceArray;

	isValidSurfaceArray.resize(gMemory->NumSurfaceVertices, 0);
	checkCudaErrors(cudaMemcpy(static_cast<void*>(isValidSurfaceArray.data()),
		gMemory->dIsValidSurfaceCube, sizeof(IsValid) * gMemory->NumSurfaceVertices, cudaMemcpyDeviceToHost));
	
	mSurfaceVerticesIndexArray.resize(gMemory->NumSurfaceVertices, 0);
	checkCudaErrors(cudaMemcpy(static_cast<void*>(mSurfaceVerticesIndexArray.data()),
		gMemory->dSurfaceVerticesIndex, sizeof(uint) * gMemory->NumSurfaceVertices, cudaMemcpyDeviceToHost));


	for (size_t i = 0; i < isValidSurfaceArray.size(); ++i)
	{
		if (isValidSurfaceArray[i] == 1)
		{
			validVerticesIndexArray.push_back(mSurfaceVerticesIndexArray[i]);
			mValidSurfaceCubesIndexArray.push_back(i);
		}
	}

	mNumValidSurfaceCubes = validVerticesIndexArray.size();

	//! copy scalar field grid from gpu.
	std::vector<ScalarFieldGrid> scalarGrid;
	scalarGrid.resize(params->mGridParams.scSize);
	checkCudaErrors(cudaMemcpy(static_cast<void*>(scalarGrid.data()),
		gMemory->dScalarFiled, sizeof(ScalarFieldGrid) * scalarGrid.size(), cudaMemcpyDeviceToHost));

	//! perform marching cubes here.
	MarchingCubesCPU mc(&scalarGrid, params->mGridParams, params->mSurfaceParams.isoValue);
	for (size_t i = 0; i < validVerticesIndexArray.size(); ++i)
	{
		int index1D = validVerticesIndexArray[i];
		iVector3 index3D = mc.index1DTo3D(index1D);
		Triangle triangles[5];
		int triCount = 0;
		//! marching cube algorithm.
		mc.marchingCubes(index3D, triangles, triCount);
		for (size_t i = 0; i < triCount; ++i)
		{
			mVertexArray.push_back(triangles[i].vertices[0]);
			mVertexArray.push_back(triangles[i].vertices[1]);
			mVertexArray.push_back(triangles[i].vertices[2]);

			mNormalArray.push_back(triangles[i].normals[0]);
			mNormalArray.push_back(triangles[i].normals[1]);
			mNormalArray.push_back(triangles[i].normals[2]);
		}
	}

	record_single->surcell = static_cast<float>(mNumValidSurfaceCubes)
	/ (params->mGridParams.scSize);
	record_single->trinum = static_cast<float>(mVertexArray.size())/3;

	mVertexArray.clear();
	mNormalArray.clear();

	mSurfaceVerticesIndexArray.clear();
	mSurfaceParticlesIndexArray.clear();
	mValidSurfaceCubesIndexArray.clear();


	//! get triangles from device.
	// std::vector<Triangle> triangles;
	// size_t nums = mVertexArray.size();

	// std::vector<uint> mNumTri;
	// mNumTri.push_back(nums/3);

	// for (size_t index = 0; index < nums; index += 3)
	// {
	// 	Triangle tmp;
	// 	tmp.vertices[0] = fVector3(mVertexArray[index + 0].x, mVertexArray[index + 0].y, mVertexArray[index + 0].z);
	// 	tmp.vertices[1] = fVector3(mVertexArray[index + 1].x, mVertexArray[index + 1].y, mVertexArray[index + 1].z);
	// 	tmp.vertices[2] = fVector3(mVertexArray[index + 2].x, mVertexArray[index + 2].y, mVertexArray[index + 2].z);
	// 	tmp.normals[0] = fVector3(mNormalArray[index + 0].x, mNormalArray[index + 0].y, mNormalArray[index + 0].z);
	// 	tmp.normals[1] = fVector3(mNormalArray[index + 1].x, mNormalArray[index + 1].y, mNormalArray[index + 1].z);
	// 	tmp.normals[2] = fVector3(mNormalArray[index + 2].x, mNormalArray[index + 2].y, mNormalArray[index + 2].z);
	// 	triangles.push_back(tmp);
	// }

	// std::cout << "Number of vertices : " << nums << std::endl;
	// std::cout << "Number of normals  : " << nums << std::endl;
	// std::cout << "Number of triangles: " << triangles.size() << std::endl;


	// std::string basename = "Test-CPU";
	// std::string path = params->mConfigParams.boundPath + std::string(basename) + ".ply";
    // std::ofstream file;

	// file.open(path.c_str(), std::ios::out);

	// if (file)
	// {
	// 	std::cout << "Writing to " << path << "...\n";

	// 	file << "ply" << std::endl;
	// 	file << "format ascii 1.0" << std::endl;
	// 	file << "comment gpupost generated" << std::endl;
	// 	file << "element vertex " << nums << std::endl;
	// 	file << "property float x " << std::endl;
	// 	file << "property float y " << std::endl;
	// 	file << "property float z " << std::endl;
	// 	file << "element face  " << triangles.size() << std::endl;
	// 	file << "property list uchar int vertex_indices " << std::endl;
	// 	file << "end_header " << std::endl;


	// 	//! positions.
	// 	for (const auto &elem : triangles)
	// 	{
	// 		file << elem.vertices[0].x << " " << elem.vertices[0].y << " " << elem.vertices[0].z << std::endl;
	// 		file << elem.vertices[1].x << " " << elem.vertices[1].y << " " << elem.vertices[1].z << std::endl;
	// 		file << elem.vertices[2].x << " " << elem.vertices[2].y << " " << elem.vertices[2].z << std::endl;
	// 	}

	// 	//! faces.
	// 	for (size_t i = 1; i <= triangles.size() * 3; i += 3)
	// 	{
	// 		file << "3 ";
	// 		file << (i + 0) - 1 << " ";
	// 		file << (i + 1) - 1 << " ";
	// 		file << (i + 2) - 1 << " ";
	// 		file << std::endl;
	// 	}

	// 	file.close();

	// 	std::cout << "Finish writing " << path << ".\n";
	// }
	// else
	// 	std::cerr << "Failed to save the file: " << path << std::endl;
}