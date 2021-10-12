#include "myCuda.cuh"
#include "GPUmemory.h"
#include "SVD3.cuh"

#include <helper_math.h>

__constant__  SimParams      dSimParams;
__constant__  SurfaceParams  dSurfaceParams;
__constant__  GridParams     dGridParams;

__constant__ uint     dNumSurfaceParticles;
__constant__ uint     dNumInvolveParticles;
__constant__ uint     dNumSurfaceVertices;

__global__
void estimationOfSurfaceParticles
(
	SpatialGrid* spatialGrid,
    NumParticleGrid* numSurParticleGrid,
    NumParticleGrid* numInvParticleGrid,
    IndexRange* particlesIndexRange
) 
{
	uint threadId = getthreadIdGlobal();

	if (threadId >= dGridParams.spSize){
		return;
	}

	if (!spatialGrid[threadId].fluid){
		return;
	}

	//detect surface grid
	bool isSurface = false;

	uint3 spatialGridRes = dGridParams.spresolution;
	uint3 Index3Dsp = index1DTo3D(threadId, dGridParams.spresolution);

	uint3 lower = make_uint3(Index3Dsp.x - 1, Index3Dsp.y - 1, Index3Dsp.z - 1);
	uint3 upper = make_uint3(Index3Dsp.x + 1, Index3Dsp.y + 1, Index3Dsp.z + 1);
	lower = clamp(lower, make_uint3(0, 0, 0), make_uint3(spatialGridRes.x - 1, spatialGridRes.y - 1, spatialGridRes.z - 1));
	upper = clamp(upper, make_uint3(0, 0, 0), make_uint3(spatialGridRes.x - 1, spatialGridRes.y - 1, spatialGridRes.z - 1));

#pragma unroll 3
	for (uint z = lower.z; z <= upper.z; ++z)
	{
#pragma unroll 3
		for (uint y = lower.y; y <= upper.y; ++y)
		{
#pragma unroll 3
			for (uint x = lower.x; x <= upper.x; ++x)
			{
				uint3 neighbor = make_uint3(x, y, z); 
				uint index = index3DTo1D(neighbor, spatialGridRes);  				
				bool flag = spatialGrid[index].fluid;
				if (!flag)
				{
					isSurface = true;
					break;
				}
			}
		}
	}
	
	if (!isSurface) 
		return;

	IndexRange indexRange = particlesIndexRange[threadId];

	uint num = indexRange.end - indexRange.start;

	numSurParticleGrid[threadId] = num;

	spatialGrid[threadId].inner = 0;
	spatialGrid[threadId].surface = 1;

	// spatialGrid[threadId].classify = 1;

	// uint extent = dGridParams.spexpandExtent;
	// lower = Index3Dsp - extent;
	// upper = Index3Dsp + extent;

	// lower = clamp(lower, make_uint3(0, 0, 0), make_uint3(spatialGridRes.x - 1, spatialGridRes.y - 1, spatialGridRes.z - 1));
	// upper = clamp(upper, make_uint3(0, 0, 0), make_uint3(spatialGridRes.x - 1, spatialGridRes.y - 1, spatialGridRes.z - 1));

	// for (uint z = lower.z; z <= upper.z; ++z)
	// {
	// 	for (uint y = lower.y; y <= upper.y; ++y)
	// 	{
	// 		for (uint x = lower.x; x <= upper.x; ++x)
	// 		{
	// 			uint3 neighbor = make_uint3(x, y, z); // 领域网格标号
	// 			uint index = index3DTo1D(neighbor, spatialGridRes);  

	// 			IndexRange indexRange = particlesIndexRange[index];	
	// 			if (indexRange.start == 0xffffffff)
	// 			{				
	// 				continue;
	// 			}
	// 			uint num = indexRange.end - indexRange.start;
	// 			numInvParticleGrid[index] = num;
	// 		}
	// 	}
	// }
}


__global__
void estimationOfInvolveParticles
(
	NumParticleGrid* numSurParticleGrid,
    NumParticleGrid* numInvParticleGrid) 
{
	uint threadId = getthreadIdGlobal();

	if (threadId >= dGridParams.spSize)
		return;

	if (numInvParticleGrid[threadId] > 0)
	{
		numInvParticleGrid[threadId] = numInvParticleGrid[threadId] - numSurParticleGrid[threadId];
	}
}


__global__
void compactationOfParticles
(
	NumParticleGrid* numParticleGrid, 
	NumParticleGridScan* numParticleGridScan,  
	IndexRange* particlesIndexRange, 
	Index* particlesIndex
) 
{
	uint threadId = getthreadIdGlobal();

	if(threadId >= dGridParams.spSize) 
        return;

    if(numParticleGrid[threadId] > 0)
	{
		uint start = numParticleGridScan[threadId];
		IndexRange range = particlesIndexRange[threadId];
		uint count = range.end - range.start;


		for (uint i = 0; i < count; ++i)
		{			
			particlesIndex[start + i] = range.start + i;
		}

	}
	
}


__global__
void calculationOfDensity(
	FluidParticle* particles,
	IndexRange* particlesIndexRange, 
	SpatialGrid* spatialGrid,
	uint numParticles
)
{
	uint threadId = getthreadIdGlobal();
	if(threadId >= numParticles){
		return;
	}

	uint particleIndex = threadId;
	float3 pos = particles[particleIndex].pos;

	uint3 cubeIndex3D = getIndex3D(pos, dGridParams.minPos, dGridParams.spGridSize);

	uint3 minIndex3D = cubeIndex3D - dGridParams.spexpandExtent; 
	uint3 maxIndex3D = cubeIndex3D + dGridParams.spexpandExtent;

	uint3 spatialGridInfo = dGridParams.spresolution;

	minIndex3D = clamp(minIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));
	maxIndex3D = clamp(maxIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));

	float wSum = 0.0f;
	float3 posMean = make_float3(0.0f, 0.0f, 0.0f);
	for (int zSp = minIndex3D.z; zSp <= maxIndex3D.z; zSp++)
	{
		for (int ySp = minIndex3D.y; ySp <= maxIndex3D.y; ySp++)
		{
			for (int xSp = minIndex3D.x; xSp <= maxIndex3D.x; xSp++)
			{
				uint3 index3D = make_uint3(xSp, ySp, zSp);
				uint index1D = index3DTo1D(index3D, spatialGridInfo);
				IndexRange indexRange = particlesIndexRange[index1D];
				if (indexRange.start == 0xffffffff)
					continue;
				for (uint i = indexRange.start; i < indexRange.end; i++)
				{
					float3 neighborPos = particles[i].pos;
					float3 delta = pos - neighborPos;
					float dist = sqrt(dot(delta, delta));

					float wj = wij_density(dist, dSurfaceParams.smoothingRadius);
					wSum += wj;
				}
			}
		}
	}

	particles[particleIndex].rhop = wSum;
}


__global__
void calculationOfSmoothed(
	FluidParticle* particles, 
	MeanPos* meanParticles, 
	SmoothedPos* smoothedParticles,
	IndexRange* particlesIndexRange, 
	Index* particlesIndex,
	SpatialGrid* spatialGrid) 
{
	uint threadId = getthreadIdGlobal();
	if (threadId >= dNumSurfaceParticles)
		return;
	
	uint particleIndex = particlesIndex[threadId];

	float3 pos = particles[particleIndex].pos;

	uint3 cubeIndex3D = getIndex3D(pos, dGridParams.minPos, dGridParams.spGridSize);

	uint3 minIndex3D = cubeIndex3D - 2 * dGridParams.spexpandExtent; // anisotropic Radius for 2 expand
	uint3 maxIndex3D = cubeIndex3D + 2 * dGridParams.spexpandExtent;

	uint3 spatialGridInfo = dGridParams.spresolution;

	minIndex3D = clamp(minIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));
	maxIndex3D = clamp(maxIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));


	float wSum = 0.0f;
	float3 posMean = make_float3(0.0f, 0.0f, 0.0f);
	for (int zSp = minIndex3D.z; zSp <= maxIndex3D.z; zSp++)
	{
		for (int ySp = minIndex3D.y; ySp <= maxIndex3D.y; ySp++)
		{
			for (int xSp = minIndex3D.x; xSp <= maxIndex3D.x; xSp++)
			{
				uint3 index3D = make_uint3(xSp, ySp, zSp);
				uint index1D = index3DTo1D(index3D, spatialGridInfo);
				IndexRange indexRange = particlesIndexRange[index1D];
				if (indexRange.start == 0xffffffff)
					continue;
				for (uint i = indexRange.start; i < indexRange.end; i++)
				{
					float3 neighborPos = particles[i].pos;
					float3 delta = pos - neighborPos;
					float dist = sqrt(dot(delta, delta));

					float wj = wij(dist, dSurfaceParams.anisotropicRadius);
					wSum += wj;
					posMean += neighborPos * wj;
				}
			}
		}
	}

	if (wSum > 0.0f)
	{
		posMean /= wSum;
		meanParticles[threadId].pos = posMean;
		// smoothedParticles[threadId].pos =
		// 	(1.0 - dSurfaceParams.lambdaForSmoothed) * pos 
		// 	+ dSurfaceParams.lambdaForSmoothed * posMean;
	}
}


__global__
void calculationOfSmoothedforInvovle(
	FluidParticle* particles, 
	SmoothedPos* smoothedParticles,
	IndexRange* particlesIndexRange, 
	Index* particlesIndex) 
{

	uint threadId = getthreadIdGlobal();
	if (threadId >= dNumInvolveParticles)
		return;
	
	uint particleIndex = particlesIndex[threadId];

	float3 pos = particles[particleIndex].pos;

	uint3 cubeIndex3D = getIndex3D(pos, dGridParams.minPos, dGridParams.spGridSize);

	uint3 minIndex3D = cubeIndex3D - 2 * dGridParams.spexpandExtent; // anisotropic Radius for 2 expand
	uint3 maxIndex3D = cubeIndex3D + 2 * dGridParams.spexpandExtent;

	uint3 spatialGridInfo = dGridParams.spresolution;

	minIndex3D = clamp(minIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));
	maxIndex3D = clamp(maxIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));


	float wSum = 0.0f;
	float3 posMean = make_float3(0.0f, 0.0f, 0.0f);
	for (int zSp = minIndex3D.z; zSp <= maxIndex3D.z; zSp++)
	{
		for (int ySp = minIndex3D.y; ySp <= maxIndex3D.y; ySp++)
		{
			for (int xSp = minIndex3D.x; xSp <= maxIndex3D.x; xSp++)
			{
				uint3 index3D = make_uint3(xSp, ySp, zSp);
				uint index1D = index3DTo1D(index3D, spatialGridInfo);
				IndexRange indexRange = particlesIndexRange[index1D];
				if (indexRange.start == 0xffffffff)
					continue;
				for (uint i = indexRange.start; i < indexRange.end; i++)
				{
					float3 neighborPos = particles[i].pos;
					float3 delta = pos - neighborPos;
					float dist = sqrt(dot(delta, delta));

					const float wj = wij(dist, dSurfaceParams.anisotropicRadius);
					wSum += wj;
					posMean += neighborPos * wj;
				}
			}
		}
	}

	if (wSum > 0.0f)
	{
		posMean /= wSum;
		smoothedParticles[threadId].pos =
			(1.0 - dSurfaceParams.lambdaForSmoothed) * pos 
			+ dSurfaceParams.lambdaForSmoothed * posMean;
	}	
}


__global__
void calculationOfTransformMatrices(
	MeanPos* meanParticles, 
	FluidParticle* particles, 
	IndexRange* particlesIndexRange, 
	Index* particlesIndex, 
	MatrixValue* svdMatrices) 
{

	uint threadId = getthreadIdGlobal();

	if (threadId >= dNumSurfaceParticles)
		return;

	uint particleIndex = particlesIndex[threadId];

	float3 pos = particles[particleIndex].pos;

	float3 posMean = meanParticles[threadId].pos;

	uint3 cubeIndex3D = getIndex3D(pos, dGridParams.minPos, dGridParams.spGridSize);

	uint extent = dGridParams.spexpandExtent;
	uint3 minIndex3D = cubeIndex3D - 2 * extent;
	uint3 maxIndex3D = cubeIndex3D + 2 * extent;

	uint3 spatialGridInfo = dGridParams.spresolution;

	minIndex3D = clamp(minIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));
	maxIndex3D = clamp(maxIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));

	MatrixValue cov;
	cov.a11 = cov.a22 = cov.a33 = dSurfaceParams.smoothingRadiusSq;
	cov.a12 = cov.a13 = cov.a21 = cov.a23 = cov.a31 = cov.a32 = 0;

	uint numNeighbors = 0;
	float wSum = 0.0f;

	for (int zSp = minIndex3D.z; zSp <= maxIndex3D.z; zSp++)
	{
		for (int ySp = minIndex3D.y; ySp <= maxIndex3D.y; ySp++)
		{
			for (int xSp = minIndex3D.x; xSp <= maxIndex3D.x; xSp++)
			{
				uint3 index3D = make_uint3(xSp, ySp, zSp);
				IndexRange indexRange = particlesIndexRange[index3DTo1D(index3D,
					spatialGridInfo)];
				if (indexRange.start == 0xffffffff)
					continue;
				for (uint i = indexRange.start; i < indexRange.end; i++)
				{
					float3 neighborPos = particles[i].pos;
					float3 u = neighborPos - pos;

					float dist = length(u);
					if (dist <= dSurfaceParams.smoothingRadius)
					{
						++numNeighbors;
					}
					float3 v = neighborPos - posMean;   // posMean

					const float wj = wij(dist, dSurfaceParams.anisotropicRadius);
					wSum += wj;
					cov.a11 += wj * v.x * v.x;
					cov.a22 += wj * v.y * v.y;
					cov.a33 += wj * v.z * v.z;
					float c_12_21 = wj * v.x * v.y;
					float c_13_31 = wj * v.x * v.z;
					float c_23_32 = wj * v.y * v.z;
					cov.a12 += c_12_21;
					cov.a21 += c_12_21;
					cov.a13 += c_13_31;
					cov.a31 += c_13_31;
					cov.a23 += c_23_32;
					cov.a32 += c_23_32;
				}
			}
		}
	}
	MatrixValue ret;

	float3 n = {0,0,0};

	if (numNeighbors < dSurfaceParams.minNumNeighbors)
	{
		ret.a11 = ret.a22 = ret.a33 = dSurfaceParams.smoothingRadiusInv;  
		ret.a12 = ret.a13 = ret.a21 = ret.a23 = ret.a31 = ret.a32 = 0;

		ret.maxvalue = 1;
		particles[particleIndex].flag = 1;
		n = {1,0,0};
	}
	else
	{
		cov.a11 /= wSum; cov.a12 /= wSum; cov.a13 /= wSum;
		cov.a21 /= wSum; cov.a22 /= wSum; cov.a23 /= wSum;
		cov.a31 /= wSum; cov.a32 /= wSum; cov.a33 /= wSum;

		MatrixValue u;
		float3 v;
		MatrixValue w;

		svd(cov.a11, cov.a12, cov.a13, cov.a21, cov.a22, cov.a23, cov.a31, cov.a32, cov.a33,
			u.a11, u.a12, u.a13, u.a21, u.a22, u.a23, u.a31, u.a32, u.a33,
			v.x, v.y, v.z,
			w.a11, w.a12, w.a13, w.a21, w.a22, w.a23, w.a31, w.a32, w.a33);

		v.x = fabsf(v.x);
		v.y = fabsf(v.y);
		v.z = fabsf(v.z);

		float maxs = fmax(v.x, fmax(v.y, v.z));
		float mins = fmin(v.x, fmin(v.y, v.z));

		if (mins <= maxs * 0.25f)
		{
			particles[particleIndex].flag = 1;
		} else 
		{
			particles[particleIndex].flag = 0;
		}

		float maxSingularVal = fmax(v.x, fmax(v.y, v.z)) / 4.0f; 		
		v.x = fmax(v.x, maxSingularVal);
		v.y = fmax(v.y, maxSingularVal);
		v.z = fmax(v.z, maxSingularVal);

		float3 invV;

		invV.x = 1.0f / v.x;
		invV.y = 1.0f / v.y;
		invV.z = 1.0f / v.z;

		float scale = powf(v.x * v.y * v.z, 1.0f / 3.0f);

		float cof = dSurfaceParams.smoothingRadiusInv * scale;

		invV.x = invV.x * cof;
		invV.y = invV.y * cof;
 		invV.z = invV.z * cof;

		ret.a11 = u.a11 * invV.x; ret.a12 = u.a21 * invV.x; ret.a13 = u.a31 * invV.x;
		ret.a21 = u.a12 * invV.y; ret.a22 = u.a22 * invV.y; ret.a23 = u.a32 * invV.y;
		ret.a31 = u.a13 * invV.z; ret.a32 = u.a23 * invV.z; ret.a33 = u.a33 * invV.z;

		matrixMul(w.a11, w.a12, w.a13, w.a21, w.a22, w.a23, w.a31, w.a32, w.a33,
			ret.a11, ret.a12, ret.a13, ret.a21, ret.a22, ret.a23, ret.a31, ret.a32, ret.a33,
			ret.a11, ret.a12, ret.a13, ret.a21, ret.a22, ret.a23, ret.a31, ret.a32, ret.a33);

		n.x = u.a13;
		n.y = u.a23;
		n.z = u.a33;

		v.x = v.x / scale;
		v.y = v.y / scale;
		v.z = v.z / scale;

		ret.maxvalue = v.x;		
	}

	if (dot(n, (pos - posMean)) < 0.f)
	{
		n = -n;
	}

	svdMatrices[threadId] = ret;
	particles[particleIndex].nor = n;
}


__global__
void estimationOfSurfaceVertices(
	FluidParticle* particles,
	SmoothedPos* smoothedParticles,
	IndexRange* particlesIndexRange,
	Index* particlesIndex, 
	IsSurface* isSurfaceVertex,
	MatrixValue* svdMatrices,
	SpatialGrid* spatialGrid,
	ScalarFieldGrid* scalarFieldGrid) 
{
	uint threadId = getthreadIdGlobal();

	if (threadId >= dNumSurfaceParticles)
		return;

	uint particleIndex = particlesIndex[threadId];
	float3 pos = particles[particleIndex].pos;
	
	bool flag = particles[particleIndex].flag;
	float3 normal =  particles[particleIndex].nor;

	// MatrixValue gMat = svdMatrices[threadId];
	// float maxvalue = gMat.maxvalue;

	int3 index3D = getIndex3D_int(pos, dGridParams.scminPos, dGridParams.scGridSize); 	

	float maxdistance = dSurfaceParams.smoothingRadius; // maxdistance = 2 * maxdistance for robust

	uint extent = ceil(maxdistance / dGridParams.scGridSize); 

	float3 aa = pos - maxdistance;
	float3 bb = pos + maxdistance;

	int3 minIndex3D = index3D - extent;
	int3 maxIndex3D = index3D + extent;

	minIndex3D = clamp(minIndex3D, make_int3(0, 0, 0), make_int3(dGridParams.scresolution.x - 1,
		dGridParams.scresolution.y - 1, dGridParams.scresolution.z - 1));
	maxIndex3D = clamp(maxIndex3D, make_int3(0, 0, 0), make_int3(dGridParams.scresolution.x - 1,
		dGridParams.scresolution.y - 1, dGridParams.scresolution.z - 1));

	for (int zSp = minIndex3D.z; zSp <= maxIndex3D.z; ++zSp)
	{
		for (int ySp = minIndex3D.y; ySp <= maxIndex3D.y; ++ySp)
		{
			for (int xSp = minIndex3D.x; xSp <= maxIndex3D.x; ++xSp)
			{
				int3 Index3Dsc = make_int3(xSp, ySp, zSp);
				float3 vPos = getVertexPos(Index3Dsc, dGridParams.scminPos, dGridParams.scGridSize);
#ifdef USING_GRID
				int3 Index3Dsp = getIndex3D_int(vPos, dGridParams.minPos, dGridParams.spGridSize);
				int index1Dsp = index3DTo1D(Index3Dsp, dGridParams.spresolution);
				if (!spatialGrid[index1Dsp].inner)
				{
					if(!(vPos < aa || vPos > bb))
					{
						int Index1Dsc = index3DTo1D(Index3Dsc, dGridParams.scresolution);
						isSurfaceVertex[Index1Dsc] = 1;
	#ifdef LOWRES
						Index3Dsc = Index3Dsc + make_int3(1, 0, 0);
						Index1Dsc = index3DTo1D(Index3Dsc, dGridParams.scresolution);
						isSurfaceVertex[Index1Dsc] = 1;
						Index3Dsc = Index3Dsc + make_int3(-1, 0, 0);
						Index1Dsc = index3DTo1D(Index3Dsc, dGridParams.scresolution);
						isSurfaceVertex[Index1Dsc] = 1;
						Index3Dsc = Index3Dsc + make_int3(0, 1, 0);
						Index1Dsc = index3DTo1D(Index3Dsc, dGridParams.scresolution);
						isSurfaceVertex[Index1Dsc] = 1;
						Index3Dsc = Index3Dsc + make_int3(0, -1, 0);
						Index1Dsc = index3DTo1D(Index3Dsc, dGridParams.scresolution);
						isSurfaceVertex[Index1Dsc] = 1;
						Index3Dsc = Index3Dsc + make_int3(0, 0, 1);
						Index1Dsc = index3DTo1D(Index3Dsc, dGridParams.scresolution);
						isSurfaceVertex[Index1Dsc] = 1;
						Index3Dsc = Index3Dsc + make_int3(0, 0, -1);
						Index1Dsc = index3DTo1D(Index3Dsc, dGridParams.scresolution);
						isSurfaceVertex[Index1Dsc] = 1;
	#endif
					}
				}
#else
				if(!(vPos < aa || vPos > bb))
				{
					if (flag || dot(normal, (vPos - pos)) > 0)
					{
						int Index1Dsc = index3DTo1D(Index3Dsc, dGridParams.scresolution);
						isSurfaceVertex[Index1Dsc] = 1;
	#ifdef LOWRES
						Index3Dsc = Index3Dsc + make_int3(1, 0, 0);
						Index1Dsc = index3DTo1D(Index3Dsc, dGridParams.scresolution);
						isSurfaceVertex[Index1Dsc] = 1;
						Index3Dsc = Index3Dsc + make_int3(-1, 0, 0);
						Index1Dsc = index3DTo1D(Index3Dsc, dGridParams.scresolution);
						isSurfaceVertex[Index1Dsc] = 1;
						Index3Dsc = Index3Dsc + make_int3(0, 1, 0);
						Index1Dsc = index3DTo1D(Index3Dsc, dGridParams.scresolution);
						isSurfaceVertex[Index1Dsc] = 1;
						Index3Dsc = Index3Dsc + make_int3(0, -1, 0);
						Index1Dsc = index3DTo1D(Index3Dsc, dGridParams.scresolution);
						isSurfaceVertex[Index1Dsc] = 1;
						Index3Dsc = Index3Dsc + make_int3(0, 0, 1);
						Index1Dsc = index3DTo1D(Index3Dsc, dGridParams.scresolution);
						isSurfaceVertex[Index1Dsc] = 1;
						Index3Dsc = Index3Dsc + make_int3(0, 0, -1);
						Index1Dsc = index3DTo1D(Index3Dsc, dGridParams.scresolution);
						isSurfaceVertex[Index1Dsc] = 1;
	#endif
					}
				}
#endif
			}
		}
	}
}


__global__
void compactationOfSurfaceVertices(
	IsSurface* isSurfaceVertex, 
	IsSurfaceScan* isSurfaceVertexScan, 
	Index* verticesIndex) 
{
	uint threadId = getthreadIdGlobal();
	if (threadId >= dGridParams.scSize)
		return;
	
	if (isSurfaceVertex[threadId])
	{
		verticesIndex[isSurfaceVertexScan[threadId]] = threadId;
	}
}


__global__
void computationOfScalarFieldGrid(
	FluidParticle* particles, 
	SmoothedPos* smoothedSurfaceParticle,
	SmoothedPos* smoothedInvolveParticle,
	IndexRange* particlesIndexRange, 
	Index* surfaceParticlesIndex,
	Index* involveParticlesIndex,
	Index* verticesIndex,
	NumParticleGrid* numSurParticleGrid, 
	NumParticleGridScan* numSurParticleGridScan, 
	NumParticleGrid* numInvParticleGrid, 
	NumParticleGridScan* numInvParticleGridScan, 
	MatrixValue* svdMatrices, 
	ScalarFieldGrid* scalarFieldGrid)
{
	uint threadId = getthreadIdGlobal();
	if (threadId >= dNumSurfaceVertices) 		
		return;

	uint svIndex = verticesIndex[threadId];

	uint3 Index3Dsc = index1DTo3D(svIndex, dGridParams.scresolution);

	float3 vPos = getVertexPos(Index3Dsc, dGridParams.scminPos, dGridParams.scGridSize);

	uint3 Index3Dsp;

	Index3Dsp = getIndex3D(vPos, dGridParams.minPos, dGridParams.spGridSize);

	uint extent = dGridParams.spexpandExtent;

	uint3 minIndex3D = Index3Dsp - 2 * extent;
	uint3 maxIndex3D = Index3Dsp + 2 * extent;

	uint3 spatialGridInfo = dGridParams.spresolution;

	minIndex3D = clamp(minIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));
	maxIndex3D = clamp(maxIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));

	float sum = 0.0f;
	for (int zSp = minIndex3D.z; zSp < maxIndex3D.z; zSp++)
	{
		for (int ySp = minIndex3D.y; ySp < maxIndex3D.y; ySp++)
		{
			for (int xSp = minIndex3D.x; xSp < maxIndex3D.x; xSp++)
			{
				uint3 index3D = make_uint3(xSp, ySp, zSp);
				uint index1D = index3DTo1D(index3D, spatialGridInfo);

				if (numSurParticleGrid[index1D] > 0)
				{

					uint start = numSurParticleGridScan[index1D];

					IndexRange range = particlesIndexRange[index1D]; 					
					uint count = range.end - range.start;
					for (uint i = 0; i < count; ++i)
					{
						uint id = start + i;
						uint spindex = surfaceParticlesIndex[id]; 
						MatrixValue gMat = svdMatrices[id];

						// float3 neighborPos = smoothedSurfaceParticle[id].pos;						
						float3 neighborPos = particles[spindex].pos;
						if(particles[spindex].rhop == 0){
							continue;
						}
						sum += dSimParams.mass / particles[spindex].rhop
							* (anisotropicW(neighborPos - vPos, gMat, 
							determinant(gMat), gMat.maxvalue * dSurfaceParams.smoothingRadius));

					}
				}

//----------------------------if involve----------------------------------------------
				// else if (numInvParticleGrid[index1D] > 0)
				// {
				// 	uint start_involve = numInvParticleGridScan[index1D];

				// 	IndexRange range = particlesIndexRange[index1D];
				// 	uint count = range.end - range.start;

				// 	for (uint i = 0; i < count; ++i)
				// 	{
				// 		uint id = start_involve + i;
				// 		uint spindex = involveParticlesIndex[id]; 

				// 		// float3 neighborPos = smoothedInvolveParticle[id].pos;
				// 		float3 neighborPos = particles[spindex].pos;
					
				// 		sum += dSimParams.mass / particles[spindex].rhop * 
				// 		(anisotropicWOpt(neighborPos - vPos, dSurfaceParams.smoothingRadius));
				// 	}
				// }
//------------------------------------------------------------------------------------
				else{
					IndexRange range = particlesIndexRange[index1D];
					uint count = range.end - range.start;

					for (uint i = 0; i < count; ++i)
					{
						uint spindex  = range.start + i;

						float3 neighborPos = particles[spindex].pos;
						if(particles[spindex].rhop == 0){
							continue;
						}
						sum += dSimParams.mass / particles[spindex].rhop * 
						(anisotropicWOpt(neighborPos - vPos, dSurfaceParams.smoothingRadius));
					}
				}

				if (sum > 2 * dSurfaceParams.isoValue)
					break;
			}
		}
	}
	scalarFieldGrid[svIndex] = sum;   
}


