#pragma once

#include "MathVector.h"

class MarchingCubesHelper
{
public:
	static const unsigned int vertexIndexesOfEdge[12][2];

	static const int cubeVerticesOffset[8][3];

	static const float cubeEdgeDirection[12][3];

	static const unsigned int edgeFlags[256];


	static const int edgeIndexesOfTriangle[256][16];

	static const unsigned int numVertices[256];

	static void getCornerIndexes3D(const iVector3& index3D, iVector3 indexes3D[8]);
};

