#include "MarchingCubesCPU.h"

void MarchingCubesCPU::marchingCubes(
	const iVector3 & index3D,
	Triangle triangles[5], 
	int & triCount)
{
	triCount = 0;

	iVector3 corners[8];
	MarchingCubesHelper::getCornerIndexes3D(index3D, corners);

	GridVertex vertices[8];
	vertices[0] = getVertexData(corners[0]);
	vertices[1] = getVertexData(corners[1]);
	vertices[2] = getVertexData(corners[2]);
	vertices[3] = getVertexData(corners[3]);
	vertices[4] = getVertexData(corners[4]);
	vertices[5] = getVertexData(corners[5]);
	vertices[6] = getVertexData(corners[6]);
	vertices[7] = getVertexData(corners[7]);

	int vertexFlag = 0;

	for (int i = 0; i < 8; i++)
	{
		if (vertices[i].value <= mIsoValue)
			vertexFlag |= 1 << i;
	}
	if (vertexFlag == 0 || vertexFlag == 255)
		return;

	unsigned int edgeFlag = MarchingCubesHelper::edgeFlags[vertexFlag];
	fVector3 intersectPoss[12];
	fVector3 intersectNormals[12];
	for (int i = 0; i < 12; i++)
	{
		if (edgeFlag & (1 << i))
		{
			int startVertex = MarchingCubesHelper::vertexIndexesOfEdge[i][0];
			int endVertex = MarchingCubesHelper::vertexIndexesOfEdge[i][1];
			float frac = fraction(vertices[startVertex].value, vertices[endVertex].value, mIsoValue);

			//! position.
			fVector3 startPos = vertices[startVertex].position;
			fVector3 endPos = vertices[endVertex].position;
			fVector3 intersectPos = lerp(startPos, endPos, frac);
			intersectPoss[i] = intersectPos;

			//! normal.
			fVector3 startNor = vertices[startVertex].normal;
			fVector3 endNor = vertices[endVertex].normal;
			intersectNormals[i] = lerp(startNor, endNor, frac);
			intersectNormals[i].normalize();
		}
	}
	for (int i = 0; i < 5; i++)
	{
		if (MarchingCubesHelper::edgeIndexesOfTriangle[vertexFlag][i * 3] < 0)
			break;
		else
		{
			for (int j = 0; j < 3; j++)
			{
				int edgeIndex = MarchingCubesHelper::edgeIndexesOfTriangle[vertexFlag][i * 3 + j];
				triangles[triCount].vertices[j] = intersectPoss[edgeIndex];
				triangles[triCount].normals[j] = intersectNormals[edgeIndex];
			}
			triCount++;
		}
	}
}

MarchingCubesCPU::GridVertex MarchingCubesCPU::getVertexData(const iVector3 & index)
{
	GridVertex ret;

	//! scalar value.
	ret.value = getValue(index);
	//! grid position.
	ret.position.x = mScalarGridInfo.minPos.x + index.x * mScalarGridInfo.scGridSize;
	ret.position.y = mScalarGridInfo.minPos.y + index.y * mScalarGridInfo.scGridSize;
	ret.position.z = mScalarGridInfo.minPos.z + index.z * mScalarGridInfo.scGridSize;
	//! normal.
	ret.normal = getNormal(index);
	ret.normal.normalize();

	return ret;
}
