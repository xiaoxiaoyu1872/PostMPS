#include "myCuda.cuh"
#include "GPUmemory.h"
#include "SVD3.cuh"

#include <helper_math.h>

__constant__  SurfaceParams  dSurfaceParams;
__constant__  GridParams  dGridParams;

__constant__ uint     dNumSurfaceVertices;
__constant__ uint     dNumValidSurfaceCubes;
__constant__ uint     dNumSurfaceMeshVertices;

texture <uint, 1, cudaReadModeElementType> edgeTex;
texture <int, 1, cudaReadModeElementType> edgeIndexesOfTriangleTex;
texture <uint, 1, cudaReadModeElementType> numVerticesTex;
texture <uint, 1, cudaReadModeElementType> vertexIndexesOfEdgeTex;


__global__
void detectValidSurfaceCubes(
	Index* svIndex, 
	ScalarFieldGrid* scalarFieldGrid, 
	IsSurface* isSurfaceVertex,
	CubeFlag* cubeFlag,
	IsValid* isValidCube, 
	NumVertexCube* numVerticesCube
)
{
	uint threadId = getthreadIdGlobal();
	if (threadId >= dNumSurfaceVertices)
		return;

	uint cubeIndex1D = svIndex[threadId];

	uint3 cubeIndex3D = index1DTo3D(cubeIndex1D, dGridParams.scresolution);

	uint cornerIndex1Ds[8];

	getCornerIndex1Ds(cubeIndex3D, dGridParams.scresolution, cornerIndex1Ds);

	uint numVertices = 0;

//--------------------------------------------------------------------------
	if (isAllSfVertex(cornerIndex1Ds, isSurfaceVertex))
	{
		uint vertexFlag = getVertexFlag(cornerIndex1Ds, scalarFieldGrid, dSurfaceParams.isoValue);
		cubeFlag[threadId] = vertexFlag;
		
		numVertices = tex1Dfetch(numVerticesTex, vertexFlag);
	}

//--------------------------------------------------------------------------
	// {
	// 	uint vertexFlag = getVertexFlag(cornerIndex1Ds, scalarFieldGrid, dSurfaceParams.isoValue);
	// 	cubeFlag[threadId] = vertexFlag;
		
	// 	numVertices = tex1Dfetch(numVerticesTex, vertexFlag);
	// }

	isValidCube[threadId] = numVertices > 0 ? 1 : 0;
	numVerticesCube[threadId] = numVertices;
}


__global__
void compactValidSurfaceCubes
(
	Index* cubeIndex, 
	IsValid* validCube, 
	IsValidScan* validCubeScan
)
{
	uint threadId = getthreadIdGlobal();
	if (threadId >= dNumSurfaceVertices)
		return;

	if (validCube[threadId])
	{	
		cubeIndex[validCubeScan[threadId]] = threadId;
	}
}


__global__
void generateTriangles(
	Index* surfaceVertexIndex, 
	Index* validCubeIndex, 
	CubeFlag* cubeFlag,
	NumVertexCubeScan* numVerticesCubeScan, 
	ScalarFieldGrid* scalarFieldGrid, 
	Vertex* vertex, 
	Normal* normal)
{
	uint threadId = getthreadIdGlobal();

	if (threadId >= dNumValidSurfaceCubes)
		return;

	uint cubelocalIndex = validCubeIndex[threadId]; 

	uint cubeIndex = surfaceVertexIndex[cubelocalIndex];

	uint3 gridIndex3D = index1DTo3D(cubeIndex, dGridParams.scresolution);

	uint vertexFlag = cubeFlag[cubelocalIndex];

	uint edgeFlag = tex1Dfetch(edgeTex, vertexFlag);
	uint numVertices = tex1Dfetch(numVerticesTex, vertexFlag);

	uint3 cornerIndex3Ds[8];

	Vertex cornerPoss[8];
	Normal cornerNors[8];

	Vertex intersectPoss[12];
	Normal intersectNormals[12];

	getCornerIndex3Ds(gridIndex3D, cornerIndex3Ds);
	getCornerPositions(cornerIndex3Ds, dGridParams.scminPos, dGridParams.scGridSize, cornerPoss);
	getCornerNormals(cornerIndex3Ds, scalarFieldGrid, cornerNors, dGridParams.scresolution);

	float sign = (dSurfaceParams.isoValue < 0.0f) ? (-1.0f) : (1.0f);
#pragma unroll 12
	for (int i = 0; i < 12; i++)
	{
		if (edgeFlag & (1 << i))
		{
			uint start = tex1Dfetch(vertexIndexesOfEdgeTex, i << 1);
			uint end = tex1Dfetch(vertexIndexesOfEdgeTex, (i << 1) + 1);
			uint startIndex = index3DTo1D(cornerIndex3Ds[start], dGridParams.scresolution);
			uint endIndex = index3DTo1D(cornerIndex3Ds[end], dGridParams.scresolution);


			float startValue = scalarFieldGrid[startIndex];
			float endValue = scalarFieldGrid[endIndex];
			float lerpFac = getLerpFac(startValue, endValue, dSurfaceParams.isoValue);
			intersectPoss[i] = lerp(cornerPoss[start], cornerPoss[end], lerpFac);
			intersectNormals[i] = sign * normalize(lerp(cornerNors[start], cornerNors[end], lerpFac));
			
		}
	}

	uint numTri = numVertices / 3;
	for (uint i = 0; i < numTri; i++)
	{
#pragma unroll 3
		for (uint j = 0; j < 3; j++)
		{
			int edgeIndex = tex1Dfetch(edgeIndexesOfTriangleTex, vertexFlag * 16 + i * 3 + j);
			uint index = numVerticesCubeScan[cubelocalIndex] + i * 3 + j;
			
			vertex[index] = intersectPoss[edgeIndex];
			normal[index] = intersectNormals[edgeIndex];
		}
	}
}



__global__
void estimationOfScalarValue
(
	Vertex* vertex,
	FluidParticle* particles,
	IndexRange* particlesIndexRange,
	ScalarValue* scalarValue 
)
{
	uint threadId = getthreadIdGlobal();

	if (threadId >= dNumSurfaceMeshVertices)
		return;

	float3 vPos = vertex[threadId];

	uint3 Index3Dsp = getIndex3D(vPos, dGridParams.minPos, dGridParams.spGridSize);

	uint extent = dGridParams.spexpandExtent;

	uint3 minIndex3D = Index3Dsp -  extent;
	uint3 maxIndex3D = Index3Dsp +  extent;

	uint3 spatialGridInfo = dGridParams.spresolution;

	minIndex3D = clamp(minIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));
	maxIndex3D = clamp(maxIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));

	float wSum = 0.0f;

	float rhop = 0.0f;
	float3 vel;
	vel.x = 0, vel.y = 0, vel.z = 0;
	for (int zSp = minIndex3D.z; zSp < maxIndex3D.z; zSp++)
	{
		for (int ySp = minIndex3D.y; ySp < maxIndex3D.y; ySp++)
		{
			for (int xSp = minIndex3D.x; xSp < maxIndex3D.x; xSp++)
			{
				uint3 index3D = make_uint3(xSp, ySp, zSp);
				uint index1D = index3DTo1D(index3D, spatialGridInfo);

				IndexRange indexRange = particlesIndexRange[index1D];

				if (indexRange.start == 0xffffffff)
					continue;
				for (uint i = indexRange.start; i < indexRange.end; i++)
				{
					float3 neighborPos = particles[i].pos;
					float3 delta = vPos - neighborPos;
					float dist = sqrt(dot(delta, delta));

					float wj = wij(dist, dSurfaceParams.smoothingRadius);
				
					rhop += wj * particles[i].rhop;
					vel += wj * particles[i].vel;
					wSum += wj;
				}
			}
		}
	}

	if (wSum > 0.0f)
	{
		rhop /= wSum;
		vel  /= wSum;
		scalarValue[threadId].rhop = rhop;
		scalarValue[threadId].vel = vel;
		scalarValue[threadId].vel_ = length(vel);
	}
}


__global__
void detectValidSurfaceCubes(
	Index* svIndex, 
	ScalarFieldGrid* scalarFieldGrid, 
	IsSurface* isSurfaceVertex,
	IsValid* isValidCube
)
{
	uint threadId = getthreadIdGlobal();
	if (threadId >= dNumSurfaceVertices)
		return;

	uint cubeIndex1D = svIndex[threadId];

	uint3 cubeIndex3D = index1DTo3D(cubeIndex1D, dGridParams.scresolution);

	uint cornerIndex1Ds[8];

	getCornerIndex1Ds(cubeIndex3D, dGridParams.scresolution, cornerIndex1Ds);

	uint numVertices = 0;

	if (isAllSfVertex(cornerIndex1Ds, isSurfaceVertex))
	{
		uint vertexFlag = getVertexFlag(cornerIndex1Ds, scalarFieldGrid, dSurfaceParams.isoValue);
		
		numVertices = tex1Dfetch(numVerticesTex, vertexFlag);
	}

	isValidCube[threadId] = numVertices > 0 ? 1 : 0;
}