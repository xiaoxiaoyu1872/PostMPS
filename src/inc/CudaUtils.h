#pragma once

#include <iostream>
#include <helper_cuda.h>
#include <helper_math.h>
#include <vector_types.h>
#include <cuda_runtime.h>

#include "MathVector.h"

#define numThreads 256

inline float3 make_float3(const fVector3& a)
{
	return make_float3(a.x, a.y, a.z);
}

inline uint3 make_uint3(const iVector3& a)
{
	return make_uint3(a.x, a.y, a.z);
}

inline __host__ __device__ uint3 make_uint3(float3 a)
{
	return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

inline __host__ void checkMemUsed()
{
	size_t avail, total;
	cudaMemGetInfo( &avail, &total );
	float used = total - avail;

	std::cout << "================="<< std::endl;
    std::cout << "Device memory used: " << float(used/1024/1024/1024) <<"GB"<< std::endl;
    std::cout << "Total memory : " << float(total/1024/1024/1024) <<"GB"<<std::endl;
    std::cout << "================="<< std::endl;
}

inline void safeCudaFree(void** ptr)
{
	if (ptr == 0 || *ptr == 0)
		return;
	cudaFree(*ptr);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) 
	{
		std::cout << "getLastCudaError() CUDA error : "
			<< " safeCudaFree : " << "(" << static_cast<int>(err) << ") "
			<< cudaGetErrorString(err) << ".\n";
		std::cout << "getLastCudaError() CUDA error : "
			<< " safeCudaFree : " <<".\n";
	}
	*ptr = NULL;
}

inline void cudaMallocMemcpy(void** dst, void* src, size_t size)
{
	cudaMalloc(dst, size);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		std::cout << "getLastCudaError() CUDA error : "
			<< " cudaMallocMemcpy(cudaMalloc) : " << "(" << static_cast<int>(err) << ") "
			<< cudaGetErrorString(err) << ".\n";
		std::cout << "getLastCudaError() CUDA error : "
			<< " cudaMallocMemcpy(cudaMalloc) : " << ".\n";
	}

	cudaMemcpy(*dst, src, size, cudaMemcpyHostToDevice);
	err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		std::cout << "getLastCudaError() CUDA error : "
			<< " cudaMallocMemcpy(cudaMemcpy) : " << "(" << static_cast<int>(err) << ") "
			<< cudaGetErrorString(err) << ".\n";
		std::cout << "getLastCudaError() CUDA error : "
				<< " cudaMallocMemcpy(cudaMemcpy) : " << ".\n";
	}
}

inline void cudaMallocMemset(void** dst, int value, size_t size)
{
	cudaMalloc(dst, size);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		std::cout << "getLastCudaError() CUDA error : "
			<< " cudaMallocMemset(cudaMalloc) : " << "(" << static_cast<int>(err) << ") "
			<< cudaGetErrorString(err) << ".\n";
		std::cout << "getLastCudaError() CUDA error : "
			<< " cudaMallocMemset(cudaMalloc) : " << ".\n";
	}

	cudaMemset(*dst, value, size);
	err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		std::cout << "getLastCudaError() CUDA error : "
			<< " cudaMallocMemset(cudaMemset) : " << "(" << static_cast<int>(err) << ") "
			<< cudaGetErrorString(err) << ".\n";
		std::cout << "getLastCudaError() CUDA error : "
			<< " cudaMallocMemset(cudaMemset) : " << ".\n";
	}
}

inline bool calcGridDimBlockDim(unsigned long long threadSize, dim3& gridDim_, dim3& blockDim_)
{
	const static unsigned long long MaxThreadSizeSupport = (long long)65535 * 65535 * 128;//~5.4*10^11
	if (threadSize == 0 || threadSize > MaxThreadSizeSupport)
		return false;

	blockDim_ = make_uint3(numThreads, 1, 1);

	uint blocksNum = (uint)(threadSize / numThreads);
	if (threadSize % numThreads != 0)
		blocksNum++;

	gridDim_ = make_uint3(blocksNum, 1, 1);
	
	if (blocksNum > 65535)
	{
		gridDim_.x = 65535;
		gridDim_.y = blocksNum / 65535;
		if (blocksNum % 65535 != 0)
			gridDim_.y++;
	}
	return true;
}
