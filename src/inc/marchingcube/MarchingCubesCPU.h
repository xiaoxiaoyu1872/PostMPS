#pragma once

#include "Define.h"
#include "MathVector.h"
#include "MarchingCubesHelper.h"

#include <vector>

// Yang et al.
class MarchingCubesCPU final
{
public:
	MarchingCubesCPU(std::vector<ScalarFieldGrid> *grid, GridParams gridInfo, float iso)
		: mScalarGird(grid), mScalarGridInfo(gridInfo), mIsoValue(iso) {}

	void marchingCubes(
		const iVector3 &index3D,							//! input, cell id.
		Triangle triangles[5],								//! output, triangles result.
		int &triCount);										//! output, number of triangles.

	inline void getCornerIndexes3D(const iVector3& index3D, iVector3 indexes3D[8])
	{
		for (int i = 0; i < 8; i++)
		{
			const int* offset = MarchingCubesHelper::cubeVerticesOffset[i];
			iVector3 index(offset[0], offset[1], offset[2]);
			indexes3D[i] = index3D + index;
		}
	}

	inline int index3DTo1D(const iVector3 &index)
	{
		if (index.x < 0 || index.x >= mScalarGridInfo.scresolution.x ||
			index.y < 0 || index.y >= mScalarGridInfo.scresolution.y ||
			index.z < 0 || index.z >= mScalarGridInfo.scresolution.z)
			return -1;
		return (index.z * mScalarGridInfo.scresolution.y + index.y) * mScalarGridInfo.scresolution.x + index.x;
	}

	inline iVector3 index1DTo3D(int index1D)
	{
		iVector3 res;
		int xy = mScalarGridInfo.scresolution.x * mScalarGridInfo.scresolution.y;
		res.z = index1D / xy;
		int mod = index1D % xy;
		res.y = mod / mScalarGridInfo.scresolution.x;
		res.x = mod % mScalarGridInfo.scresolution.x;
		return res;
	}

	inline float getValue(const iVector3 &index3D)
	{
		if (index3D.x >= 0 && index3D.x < mScalarGridInfo.scresolution.x ||
			index3D.y >= 0 && index3D.y < mScalarGridInfo.scresolution.y ||
			index3D.z >= 0 && index3D.z < mScalarGridInfo.scresolution.z)
			return (*mScalarGird)[index3DTo1D(index3D)];
		return 0.0f;
	}

	inline const fVector3 getNormal(const iVector3& index3D)
	{
		int i = index3D.x;
		int j = index3D.y;
		int k = index3D.z;
		float gradX = getValue(iVector3(i - 1, j, k)) - getValue(iVector3(i + 1, j, k));
		float gradY = getValue(iVector3(i, j - 1, k)) - getValue(iVector3(i, j + 1, k));
		float gradZ = getValue(iVector3(i, j, k - 1)) - getValue(iVector3(i, j, k + 1));
		return fVector3(gradX, gradY, gradZ);
	}

protected:

	struct GridVertex 
	{
		float value;
		fVector3 position;
		fVector3 normal;
	};

	inline float fraction(float val0, float val1, float targetVal) const
	{
		float delta = val1 - val0;
		if (delta > -1.0e-7 && delta < 1.0e-7)
			return 0.5;
		return (targetVal - val0) / delta;
	}

	inline fVector3 lerp(const fVector3 &v0, const fVector3 &v1, float frac) const
	{
		return v0 + (v1 - v0) * frac;
	}

	GridVertex getVertexData(const iVector3 &index);


private:
	float mIsoValue;
	GridParams mScalarGridInfo;
	std::vector<ScalarFieldGrid> *mScalarGird;
};