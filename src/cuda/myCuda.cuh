#ifndef _MYCUDA_CUH
#define _MYCUDA_CUH

#include "Define.h"
#include "CudaUtils.h"
#include "Params.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_functions.h>


inline __device__
uint getthreadIdGlobal()
{
	uint blockId = blockIdx.z*gridDim.y*gridDim.x + blockIdx.y*gridDim.x + blockIdx.x;
	uint threadId = threadIdx.z*blockDim.y*blockDim.x + threadIdx.y*blockDim.x + threadIdx.x
		+ blockId * blockDim.x*blockDim.y*blockDim.z;
	return threadId;
}

inline __host__ __device__ 
bool operator>(float3 a, float3 b)
{
    if (a.x > b.x || a.y > b.y || a.z > b.z)
    {
        return true;
    }
    
    return false;
}

inline __host__ __device__ 
bool operator<(float3 a, float3 b)
{
    // bool outofbound;

    if (a.x < b.x || a.y < b.y || a.z < b.z)
    {
        return true;
    }
    
    return false;
}

inline __host__ __device__ 
int3 getIndex3D_int(float3 pos, float3 gridMinPos, float cellSize)
{
	return make_int3((pos - gridMinPos) / cellSize);
}

inline __host__ __device__ 
uint3 getIndex3D(float3 pos, float3 gridMinPos, float cellSize)
{
	return make_uint3((pos - gridMinPos) / cellSize);
}

inline __host__ __device__ 
uint3 index1DTo3D(uint index1, uint3 res)
{
	uint z = index1 / (res.x*res.y);
	uint m = index1 % (res.x*res.y);
	uint y = m / res.x;
	uint x = m % res.x;
	return make_uint3(x, y, z);
}

inline __host__ __device__ 
uint index3DTo1D(uint3 index3, uint3 res)
{
	return index3.z*res.x*res.y + index3.y*res.x + index3.x;
}

inline __host__ __device__ 
int index3DTo1D(int3 index3, uint3 res)
{
	return index3.z*res.x*res.y + index3.y*res.x + index3.x;
}


inline __host__ __device__
float wij(float distance, float r)
{
	if (distance < r)
	{
		float s = distance / r;
		return 1.0f - s * s * s;
	}
	else
		return 0.0f;
}

__device__ __forceinline__
void matrixMul(
	float a11, float a12, float a13,
	float a21, float a22, float a23,
	float a31, float a32, float a33,
	float b11, float b12, float b13,
	float b21, float b22, float b23,
	float b31, float b32, float b33,
	float &c11, float &c12, float &c13,
	float &c21, float &c22, float &c23,
	float &c31, float &c32, float &c33)
{
	c11 = a11 * b11 + a12 * b21 + a13 * b31;
	c21 = a21 * b11 + a22 * b21 + a23 * b31;
	c31 = a31 * b11 + a32 * b21 + a33 * b31;

	c12 = a11 * b12 + a12 * b22 + a13 * b32;
	c22 = a21 * b12 + a22 * b22 + a23 * b32;
	c32 = a31 * b12 + a32 * b22 + a33 * b32;

	c13 = a11 * b13 + a12 * b23 + a13 * b33;
	c23 = a21 * b13 + a22 * b23 + a23 * b33;
	c33 = a31 * b13 + a32 * b23 + a33 * b33;
}

inline __host__ __device__ 
float3 getVertexPos(uint3 index3D, float3 gridMinPos, float cellSize)
{
	return make_float3(index3D)*cellSize + gridMinPos;
}

inline __host__ __device__ 
float3 getVertexPos(int3 index3D, float3 gridMinPos, float cellSize)
{
	return make_float3(index3D)*cellSize + gridMinPos;
}

inline __host__ __device__
bool anisotropicWdistance(float3 r, MatrixValue g)
{
	float3 target;
	target.x = r.x * g.a11 + r.y * g.a12 + r.z * g.a13;
	target.y = r.x * g.a21 + r.y * g.a22 + r.z * g.a23;
	target.z = r.x * g.a31 + r.y * g.a32 + r.z * g.a33;

	float dist = length(target);
	// float distSq = dist * dist;

	if (dist >= 1.0)
		return false;

	return true;
}


inline __host__ __device__
float anisotropicW(float3 r, MatrixValue g, float det, float h)
{
	if (length(r) >= h)
		return 0.0f;

	float3 target;
	target.x = r.x * g.a11 + r.y * g.a12 + r.z * g.a13;
	target.y = r.x * g.a21 + r.y * g.a22 + r.z * g.a23;
	target.z = r.x * g.a31 + r.y * g.a32 + r.z * g.a33;

	float dist = length(target);
	float distSq = dist * dist;

	if (distSq >= 1.0)
		return 0.0f;

	float x = 1 - distSq;
	return SIGMA * det * x * x * x;
}

inline __host__ __device__
float anisotropicWOpt(float3 r, float h)
{
	if (length(r) >= h)
		return 0.0f;

	float InvR = 1/h;
	float det = InvR*InvR*InvR;

	r = r * InvR;

	float dist = length(r);
	float distSq = dist * dist;

	float x = 1 - distSq;

	float sum = SIGMA * det * x * x * x;
	
	return sum;
}

inline __device__  __host__
float determinant(const MatrixValue& mat)
{
	return
		mat.a11 * mat.a22 * mat.a33 +
		mat.a12 * mat.a23 * mat.a31 +
		mat.a13 * mat.a21 * mat.a32 -
		mat.a11 * mat.a23 * mat.a32 -
		mat.a12 * mat.a21 * mat.a33 -
		mat.a13 * mat.a22 * mat.a31;
}

inline __host__ __device__ 
void getCornerIndex1Ds(uint3 curIndex3D, uint3 resolution, uint* cornerIndex1Ds)
{
	cornerIndex1Ds[0] = index3DTo1D(curIndex3D + make_uint3(0, 0, 0), resolution);
	cornerIndex1Ds[1] = index3DTo1D(curIndex3D + make_uint3(0, 0, 1), resolution);
	cornerIndex1Ds[2] = index3DTo1D(curIndex3D + make_uint3(0, 1, 1), resolution);
	cornerIndex1Ds[3] = index3DTo1D(curIndex3D + make_uint3(0, 1, 0), resolution);
	cornerIndex1Ds[4] = index3DTo1D(curIndex3D + make_uint3(1, 0, 0), resolution);
	cornerIndex1Ds[5] = index3DTo1D(curIndex3D + make_uint3(1, 0, 1), resolution);
	cornerIndex1Ds[6] = index3DTo1D(curIndex3D + make_uint3(1, 1, 1), resolution);
	cornerIndex1Ds[7] = index3DTo1D(curIndex3D + make_uint3(1, 1, 0), resolution);
}

inline __host__ __device__ 
uint getVertexFlag(uint3 curIndex3D, ScalarFieldGrid* scalarFieldGrid, float isoValue, uint3 resolution)
{
	uint cornerIndex[8];
	cornerIndex[0] = index3DTo1D(curIndex3D + make_uint3(0, 0, 0), resolution);
	cornerIndex[1] = index3DTo1D(curIndex3D + make_uint3(0, 0, 1), resolution);
	cornerIndex[2] = index3DTo1D(curIndex3D + make_uint3(0, 1, 1), resolution);
	cornerIndex[3] = index3DTo1D(curIndex3D + make_uint3(0, 1, 0), resolution);
	cornerIndex[4] = index3DTo1D(curIndex3D + make_uint3(1, 0, 0), resolution);
	cornerIndex[5] = index3DTo1D(curIndex3D + make_uint3(1, 0, 1), resolution);
	cornerIndex[6] = index3DTo1D(curIndex3D + make_uint3(1, 1, 1), resolution);
	cornerIndex[7] = index3DTo1D(curIndex3D + make_uint3(1, 1, 0), resolution);

	uint vertexFlag = 0;		

	for (size_t i = 0; i < 8; i++)
		if (scalarFieldGrid[cornerIndex[i]] <= isoValue)		
			vertexFlag |= 1 << i;
	return vertexFlag;
}

inline __host__ __device__ 
bool isAllSfVertex(uint* corIndex1Ds, IsSurface* isSfGrid)
{
	for (int i = 0; i < 8; i++)
		if (!isSfGrid[corIndex1Ds[i]])
			return false;
	return true;
}

inline __host__ __device__ 
uint getVertexFlag(uint cornerIndex[8], 
ScalarFieldGrid* scalarFieldGrid, float isoValue)
{
	uint vertexFlag = 0;		
	for (size_t i = 0; i < 8; i++)
		if (scalarFieldGrid[cornerIndex[i]] <= isoValue)	
			vertexFlag |= 1 << i;
	return vertexFlag;
}

inline __host__ __device__ 
uint isValid(int3 index3D, uint3 resolution)
{
	return (index3D.x >= 0 && (uint)index3D.x < resolution.x &&
		index3D.y >= 0 && (uint)index3D.y < resolution.y &&
		index3D.z >= 0 && (uint)index3D.z < resolution.z);
}


inline __host__ __device__ 
float getValue(int3 index3D, ScalarFieldGrid* scalarFieldGrid, uint3 resolution)
{
	if (isValid(index3D, resolution))
		return scalarFieldGrid[index3DTo1D(make_uint3(index3D), resolution)];
	return 0.f;
}


inline __host__ __device__
float3 getVertexNorm(uint3 index3D, ScalarFieldGrid* scalarFieldGrid, uint3 resolution)
{
	int i = index3D.x;
	int j = index3D.y;
	int k = index3D.z;
	float3 n;
	n.x = getValue(make_int3(i - 1, j, k), scalarFieldGrid, resolution) - getValue(make_int3(i + 1, j, k), scalarFieldGrid, resolution);
	n.y = getValue(make_int3(i, j - 1, k), scalarFieldGrid, resolution) - getValue(make_int3(i, j + 1, k), scalarFieldGrid, resolution);
	n.z = getValue(make_int3(i, j, k - 1), scalarFieldGrid, resolution) - getValue(make_int3(i, j, k + 1), scalarFieldGrid, resolution);
	n = normalize(n);
	return n;
}


inline __host__ __device__ 
void getCornerIndex3Ds(uint3 curIndex3D, uint3* cornerIndex3Ds)
{
	cornerIndex3Ds[0] = curIndex3D + make_uint3(0, 0, 0);
	cornerIndex3Ds[1] = curIndex3D + make_uint3(0, 0, 1);
	cornerIndex3Ds[2] = curIndex3D + make_uint3(0, 1, 1);
	cornerIndex3Ds[3] = curIndex3D + make_uint3(0, 1, 0);
	cornerIndex3Ds[4] = curIndex3D + make_uint3(1, 0, 0);
	cornerIndex3Ds[5] = curIndex3D + make_uint3(1, 0, 1);
	cornerIndex3Ds[6] = curIndex3D + make_uint3(1, 1, 1);
	cornerIndex3Ds[7] = curIndex3D + make_uint3(1, 1, 0);
}


inline __host__ __device__ 
void getCornerPositions(uint3* cornerIndex3Ds, float3 gridMinPos, float cellSize, float3* cornerPoss)
{
	cornerPoss[0] = getVertexPos(cornerIndex3Ds[0], gridMinPos, cellSize);
	cornerPoss[1] = getVertexPos(cornerIndex3Ds[1], gridMinPos, cellSize);
	cornerPoss[2] = getVertexPos(cornerIndex3Ds[2], gridMinPos, cellSize);
	cornerPoss[3] = getVertexPos(cornerIndex3Ds[3], gridMinPos, cellSize);
	cornerPoss[4] = getVertexPos(cornerIndex3Ds[4], gridMinPos, cellSize);
	cornerPoss[5] = getVertexPos(cornerIndex3Ds[5], gridMinPos, cellSize);
	cornerPoss[6] = getVertexPos(cornerIndex3Ds[6], gridMinPos, cellSize);
	cornerPoss[7] = getVertexPos(cornerIndex3Ds[7], gridMinPos, cellSize);
}

inline __host__ __device__ 
void getCornerNormals(uint3* cornerIndex3Ds, ScalarFieldGrid* scalarFieldGrid, float3* cornerNormals, uint3 resolution)
{
	cornerNormals[0] = getVertexNorm(cornerIndex3Ds[0], scalarFieldGrid, resolution);
	cornerNormals[1] = getVertexNorm(cornerIndex3Ds[1], scalarFieldGrid, resolution);
	cornerNormals[2] = getVertexNorm(cornerIndex3Ds[2], scalarFieldGrid, resolution);
	cornerNormals[3] = getVertexNorm(cornerIndex3Ds[3], scalarFieldGrid, resolution);
	cornerNormals[4] = getVertexNorm(cornerIndex3Ds[4], scalarFieldGrid, resolution);
	cornerNormals[5] = getVertexNorm(cornerIndex3Ds[5], scalarFieldGrid, resolution);
	cornerNormals[6] = getVertexNorm(cornerIndex3Ds[6], scalarFieldGrid, resolution);
	cornerNormals[7] = getVertexNorm(cornerIndex3Ds[7], scalarFieldGrid, resolution);
}

inline  __device__ 
float getLerpFac(float val0, float val1, float targetVal)
{
	float delta = val1 - val0;
	if (delta > -EPSILON_ && delta < EPSILON_)
		return 0.5f;
	return (targetVal - val0) / delta;
}

inline __device__  __host__
float Weend(float3 xij, float h) {
  float rval = 0;
  float mxij = length(xij);
  if (mxij <= h && mxij > 0)
    rval = 1. - (mxij / h);
  return rval;
}


inline __device__  __host__
float curvature2p(float3 xi, float3 xj, float3 ni, float3 nj, float h) 
{
	 float e1 = 1 - dot(ni, nj);
	 float e2 = Weend(xi - xj, h);
	 return e1 * e2;
}

inline __device__  __host__
float crests2p(float3 xi, float3 xj, float3 ni, float3 nj, float h) 
{
	float3 xji = normalize(xj - xi);

    // float3 nni = normalize(ni);
    // float3 nnj = normalize(nj);
	float3 nni = ni;
	float3 nnj = nj;

    float kij = 0;

    // if (dot(xji, nni) < EPSILON_ || flag)
	if (dot(xji, nni) < -EPSILON_)
    {
  		kij = curvature2p(xi, xj, nni, nnj, h);
    }
 
    return kij;
}

inline __device__  __host__
float phi(float I, float tmin, float tmax) {
  return (fmin(I, tmax) - fmin(I, tmin)) / (tmax - tmin);
}

inline __device__  __host__
float solveEq(float px, float py, float pz, float vx,
              float vy, float vz, float x, float y) {
  return ((-(x - px) * vx - (y - py) * vy) / vz) + pz;
}

inline __device__  __host__
float Wwendland(float3 xij, float h) {
  float rval = 0;
  float mxij = length(xij), q = mxij / h;

  if (q >= 0. && q <= 2) {
    float ad = 21. / (16. * PI * h * h * h), e1 = (1. - (q / 2.0));
    rval = ad * e1 * e1 * e1 * e1 * (2 * q + 1.);
  }
  return rval;
}

inline __host__ __device__
float wij_density(float distance, float r)
{
	if (distance < r)
	{
		float r3 = pow(r, 3);
		float s = 1.0 - pow(distance / r, 2);
		// return 315.0 / (SIGMA * r3)* s * s * s;
		return SIGMA / r3 * s * s * s;
	}
	else
		return 0.0f;
}

#endif