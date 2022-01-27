
/*
 * cuCompactor.h
 *
 *  Created on: 21/mag/2015
 *      Author: knotman
 */

#ifndef CUCOMPACTOR_H_
#define CUCOMPACTOR_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include "cuda_error_check.cu"
#include <stdlib.h>
#include <sys/time.h>

template <class T>
void print_d(T *d_array, int len, int print_len, char *str) {
	T *array = new T[len]();
	CUDASAFECALL(cudaMemcpy(array, d_array, len * sizeof(T), cudaMemcpyDeviceToHost));
	printf("%s", str);
	assert(print_len <= len);
	// T max = 0;
	// for(int i = 0; i < len; i++) {
	// 	max = array[i] > max ? array[i] : max;
	// }
	// printf("max is %d", max);
	for(int i = 0; i < print_len; i++) {
		// if(array[i] >= 2708)
		// 	printf("%dth is %d\n", i, array[i]);
		printf("%d ", array[i]);
	}
	// printf("%d ", array[len - 2]);
	// printf("%d", array[len - 1]);
	printf("\n");
	delete [] array;
}

void exclusive_scan(int *part_count, int *part_pointer, int num_parts) {
	thrust::device_ptr<int> thrustPtr_Count(part_count);
	thrust::device_ptr<int> thrustPtr_Pointer(part_pointer);

	thrust::exclusive_scan(thrustPtr_Count, thrustPtr_Count + num_parts + 1, thrustPtr_Pointer);
	// print_d(part_pointer, num_parts + 1, num_parts + 1, "part_pointer");
}



namespace cuCompactor_index
{

#define warpSize (32)
#define FULL_MASK 0xffffffff

	__host__ __device__ int divup(int x, int y)
	{
		return x / y + (x % y ? 1 : 0);
	}

	__device__ __inline__ int pow2i(int e)
	{
		return 1 << e;
	}

	template <typename T, typename Predicate>
	__global__ void computeBlockCounts(T *d_input, int length, int *d_BlockCounts, Predicate predicate)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < length)
		{
			int pred = predicate(idx);
			int BC = __syncthreads_count(pred);

			if (threadIdx.x == 0)
			{
				d_BlockCounts[blockIdx.x] = BC; // BC will contain the number of valid elements in all threads of this thread block
			}
		}
	}

	template <typename T, typename Predicate>
	__global__ void compactK(T *d_input, int length, T *d_output, int *d_BlocksOffset, Predicate predicate)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		extern __shared__ int warpTotals[];
		if (idx < length)
		{
			int pred = predicate(idx);
			int w_i = threadIdx.x / warpSize; //warp index
			int w_l = idx % warpSize;		  //thread index within a warp

			// compute exclusive prefix sum based on predicate validity to get output offset for thread in warp
			int t_m = FULL_MASK >> (warpSize - w_l); //thread mask
#if (CUDART_VERSION < 9000)
			int b = __ballot(pred) & t_m; //ballot result = number whose ith bit is one if the ith's thread pred is true masked up to the current index in warp
#else
			int b = __ballot_sync(FULL_MASK, pred) & t_m;
#endif
			int t_u = __popc(b); // popc count the number of bit one. simply count the number predicated true BEFORE MY INDEX

			// last thread in warp computes total valid counts for the warp
			if (w_l == warpSize - 1)
			{
				warpTotals[w_i] = t_u + pred;
			}

			// need all warps in thread block to fill in warpTotals before proceeding
			__syncthreads();

			// first numWarps threads in first warp compute exclusive prefix sum to get output offset for each warp in thread block
			int numWarps = blockDim.x / warpSize;
			unsigned int numWarpsMask = FULL_MASK >> (warpSize - numWarps);
			if (w_i == 0 && w_l < numWarps)
			{
				int w_i_u = 0;
				for (int j = 0; j <= 5; j++)
				{ // must include j=5 in loop in case any elements of warpTotals are identically equal to 32
#if (CUDART_VERSION < 9000)
					int b_j = __ballot(warpTotals[w_l] & pow2i(j)); //# of the ones in the j'th digit of the warp offsets
#else
					int b_j = __ballot_sync(numWarpsMask, warpTotals[w_l] & pow2i(j));
#endif
					w_i_u += (__popc(b_j & t_m)) << j;
					//printf("indice %i t_m=%i,j=%i,b_j=%i,w_i_u=%i\n",w_l,t_m,j,b_j,w_i_u);
				}
				warpTotals[w_l] = w_i_u;
			}

			// need all warps in thread block to wait until prefix sum is calculated in warpTotals
			__syncthreads();

			// if valid element, place the element in proper destination address based on thread offset in warp, warp offset in block, and block offset in grid
			if (pred)
			{
				d_output[t_u + warpTotals[w_i] + d_BlocksOffset[blockIdx.x]] = d_input[idx];
			}
		}
	}

	template <class T>
	__global__ void printArray_GPU(T *hd_data, int size, int newline)
	{
		int w = 0;
		for (int i = 0; i < size; i++)
		{
			if (i % newline == 0)
			{
				printf("\n%i -> ", w);
				w++;
			}
			printf("%i ", hd_data[i]);
		}
		printf("\n");
	}

	template <typename T, typename Predicate>
	int compact(T *d_input, T *d_output, int length, Predicate predicate, int blockSize)
	{
		int numBlocks = divup(length, blockSize);
		int *d_BlocksCount;
		int *d_BlocksOffset;
		CUDASAFECALL(cudaMalloc(&d_BlocksCount, sizeof(int) * numBlocks));
		CUDASAFECALL(cudaMalloc(&d_BlocksOffset, sizeof(int) * numBlocks));
		thrust::device_ptr<int> thrustPrt_bCount(d_BlocksCount);
		thrust::device_ptr<int> thrustPrt_bOffset(d_BlocksOffset);

		//phase 1: count number of valid elements in each thread block
		computeBlockCounts<<<numBlocks, blockSize>>>(d_input, length, d_BlocksCount, predicate);
		CUDASAFECALL(cudaDeviceSynchronize());
		//phase 2: compute exclusive prefix sum of valid block counts to get output offset for each thread block in grid
		thrust::exclusive_scan(thrustPrt_bCount, thrustPrt_bCount + numBlocks, thrustPrt_bOffset);
		//phase 3: compute output offset for each thread in warp and each warp in thread block, then output valid elements
		compactK<<<numBlocks, blockSize, sizeof(int) * (blockSize / warpSize)>>>(d_input, length, d_output, d_BlocksOffset, predicate);
		CUDASAFECALL(cudaDeviceSynchronize());

		// determine number of elements in the compacted list
		int compact_length = thrustPrt_bOffset[numBlocks - 1] + thrustPrt_bCount[numBlocks - 1];

		cudaFree(d_BlocksCount);
		cudaFree(d_BlocksOffset);

		return compact_length;
	}

} /* namespace cuCompactor */

namespace cuCompactor_val
{

#define warpSize (32)
#define FULL_MASK 0xffffffff

	__host__ __device__ int divup(int x, int y)
	{
		return x / y + (x % y ? 1 : 0);
	}

	__device__ __inline__ int pow2i(int e)
	{
		return 1 << e;
	}

	template <typename T, typename Predicate>
	__global__ void computeBlockCounts(T *d_input, int length, int *d_BlockCounts, Predicate predicate)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < length)
		{
			int pred = predicate(d_input[idx]);
			int BC = __syncthreads_count(pred);

			if (threadIdx.x == 0)
			{
				d_BlockCounts[blockIdx.x] = BC; // BC will contain the number of valid elements in all threads of this thread block
			}
		}
	}

	template <typename T, typename Predicate>
	__global__ void compactK(T *d_input, int length, T *d_output, int *d_BlocksOffset, Predicate predicate)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		extern __shared__ int warpTotals[];
		if (idx < length)
		{
			int pred = predicate(d_input[idx]);
			int w_i = threadIdx.x / warpSize; //warp index
			int w_l = idx % warpSize;		  //thread index within a warp

			// compute exclusive prefix sum based on predicate validity to get output offset for thread in warp
			int t_m = FULL_MASK >> (warpSize - w_l); //thread mask
#if (CUDART_VERSION < 9000)
			int b = __ballot(pred) & t_m; //ballot result = number whose ith bit is one if the ith's thread pred is true masked up to the current index in warp
#else
			int b = __ballot_sync(FULL_MASK, pred) & t_m;
#endif
			int t_u = __popc(b); // popc count the number of bit one. simply count the number predicated true BEFORE MY INDEX

			// last thread in warp computes total valid counts for the warp
			if (w_l == warpSize - 1)
			{
				warpTotals[w_i] = t_u + pred;
			}

			// need all warps in thread block to fill in warpTotals before proceeding
			__syncthreads();

			// first numWarps threads in first warp compute exclusive prefix sum to get output offset for each warp in thread block
			int numWarps = blockDim.x / warpSize;
			unsigned int numWarpsMask = FULL_MASK >> (warpSize - numWarps);
			if (w_i == 0 && w_l < numWarps)
			{
				int w_i_u = 0;
				for (int j = 0; j <= 5; j++)
				{ // must include j=5 in loop in case any elements of warpTotals are identically equal to 32
#if (CUDART_VERSION < 9000)
					int b_j = __ballot(warpTotals[w_l] & pow2i(j)); //# of the ones in the j'th digit of the warp offsets
#else
					int b_j = __ballot_sync(numWarpsMask, warpTotals[w_l] & pow2i(j));
#endif
					w_i_u += (__popc(b_j & t_m)) << j;
					//printf("indice %i t_m=%i,j=%i,b_j=%i,w_i_u=%i\n",w_l,t_m,j,b_j,w_i_u);
				}
				warpTotals[w_l] = w_i_u;
			}

			// need all warps in thread block to wait until prefix sum is calculated in warpTotals
			__syncthreads();

			// if valid element, place the element in proper destination address based on thread offset in warp, warp offset in block, and block offset in grid
			if (pred)
			{
				d_output[t_u + warpTotals[w_i] + d_BlocksOffset[blockIdx.x]] = d_input[idx];
			}
		}
	}

	template <class T>
	__global__ void printArray_GPU(T *hd_data, int size, int newline)
	{
		int w = 0;
		for (int i = 0; i < size; i++)
		{
			if (i % newline == 0)
			{
				printf("\n%i -> ", w);
				w++;
			}
			printf("%i ", hd_data[i]);
		}
		printf("\n");
	}

	template <typename T, typename Predicate>
	int compact(T *d_input, T *d_output, int length, Predicate predicate, int blockSize)
	{
		int numBlocks = divup(length, blockSize);
		int *d_BlocksCount;
		int *d_BlocksOffset;
		CUDASAFECALL(cudaMalloc(&d_BlocksCount, sizeof(int) * numBlocks));
		CUDASAFECALL(cudaMalloc(&d_BlocksOffset, sizeof(int) * numBlocks));
		thrust::device_ptr<int> thrustPrt_bCount(d_BlocksCount);
		thrust::device_ptr<int> thrustPrt_bOffset(d_BlocksOffset);

		//phase 1: count number of valid elements in each thread block
		computeBlockCounts<<<numBlocks, blockSize>>>(d_input, length, d_BlocksCount, predicate);
		CUDASAFECALL(cudaDeviceSynchronize());
		//phase 2: compute exclusive prefix sum of valid block counts to get output offset for each thread block in grid
		thrust::exclusive_scan(thrustPrt_bCount, thrustPrt_bCount + numBlocks, thrustPrt_bOffset);
		//phase 3: compute output offset for each thread in warp and each warp in thread block, then output valid elements
		compactK<<<numBlocks, blockSize, sizeof(int) * (blockSize / warpSize)>>>(d_input, length, d_output, d_BlocksOffset, predicate);
		CUDASAFECALL(cudaDeviceSynchronize());

		// determine number of elements in the compacted list
		int compact_length = thrustPrt_bOffset[numBlocks - 1] + thrustPrt_bCount[numBlocks - 1];

		cudaFree(d_BlocksCount);
		cudaFree(d_BlocksOffset);

		return compact_length;
	}

} /* namespace cuCompactor */

namespace cuCompactor_mask
{

#define warpSize (32)
#define FULL_MASK 0xffffffff

	__host__ __device__ int divup(int x, int y)
	{
		return x / y + (x % y ? 1 : 0);
	}

	__device__ __inline__ int pow2i(int e)
	{
		return 1 << e;
	}

	template <typename T>
	__global__ void computeBlockCounts(T *d_input, int length, int *d_BlockCounts)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < length)
		{
			int pred = d_input[idx] != -1;
			int BC = __syncthreads_count(pred);

			if (threadIdx.x == 0)
			{
				d_BlockCounts[blockIdx.x] = BC; // BC will contain the number of valid elements in all threads of this thread block
			}
		}
	}

	template <typename T, typename S>
	__global__ void compactK(T *d_input_id, S *d_input_edge, int length, T *d_output_id, S *d_output_edge, int *d_BlocksOffset)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		extern __shared__ int warpTotals[];
		if (idx < length)
		{
			int pred = d_input_id[idx] != -1;
			int w_i = threadIdx.x / warpSize; //warp index
			int w_l = idx % warpSize;		  //thread index within a warp

			// compute exclusive prefix sum based on predicate validity to get output offset for thread in warp
			int t_m = FULL_MASK >> (warpSize - w_l); //thread mask
#if (CUDART_VERSION < 9000)
			int b = __ballot(pred) & t_m; //ballot result = number whose ith bit is one if the ith's thread pred is true masked up to the current index in warp
#else
			int b = __ballot_sync(FULL_MASK, pred) & t_m;
#endif
			int t_u = __popc(b); // popc count the number of bit one. simply count the number predicated true BEFORE MY INDEX

			// last thread in warp computes total valid counts for the warp
			if (w_l == warpSize - 1)
			{
				warpTotals[w_i] = t_u + pred;
			}

			// need all warps in thread block to fill in warpTotals before proceeding
			__syncthreads();

			// first numWarps threads in first warp compute exclusive prefix sum to get output offset for each warp in thread block
			int numWarps = blockDim.x / warpSize;
			unsigned int numWarpsMask = FULL_MASK >> (warpSize - numWarps);
			if (w_i == 0 && w_l < numWarps)
			{
				int w_i_u = 0;
				for (int j = 0; j <= 5; j++)
				{ // must include j=5 in loop in case any elements of warpTotals are identically equal to 32
#if (CUDART_VERSION < 9000)
					int b_j = __ballot(warpTotals[w_l] & pow2i(j)); //# of the ones in the j'th digit of the warp offsets
#else
					int b_j = __ballot_sync(numWarpsMask, warpTotals[w_l] & pow2i(j));
#endif
					w_i_u += (__popc(b_j & t_m)) << j;
					//printf("indice %i t_m=%i,j=%i,b_j=%i,w_i_u=%i\n",w_l,t_m,j,b_j,w_i_u);
				}
				warpTotals[w_l] = w_i_u;
			}

			// need all warps in thread block to wait until prefix sum is calculated in warpTotals
			__syncthreads();

			// if valid element, place the element in proper destination address based on thread offset in warp, warp offset in block, and block offset in grid
			if (pred)
			{
				d_output_id[t_u + warpTotals[w_i] + d_BlocksOffset[blockIdx.x]] = d_input_id[idx];
				d_output_edge[t_u + warpTotals[w_i] + d_BlocksOffset[blockIdx.x]] = d_input_edge[idx];
			}
		}
	}

	template <class T>
	__global__ void printArray_GPU(T *hd_data, int size, int newline)
	{
		int w = 0;
		for (int i = 0; i < size; i++)
		{
			if (i % newline == 0)
			{
				printf("\n%i -> ", w);
				w++;
			}
			printf("%i ", hd_data[i]);
		}
		printf("\n");
	}

	template <typename T, typename S>
	int compact(T *d_input_id, T *d_output_id, S *d_input_edge, S *d_output_edge, int length, int blockSize)
	{
		int numBlocks = divup(length, blockSize);
		int *d_BlocksCount;
		int *d_BlocksOffset;
		CUDASAFECALL(cudaMalloc(&d_BlocksCount, sizeof(int) * numBlocks));
		CUDASAFECALL(cudaMalloc(&d_BlocksOffset, sizeof(int) * numBlocks));
		thrust::device_ptr<int> thrustPrt_bCount(d_BlocksCount);
		thrust::device_ptr<int> thrustPrt_bOffset(d_BlocksOffset);

		//phase 1: count number of valid elements in each thread block
		computeBlockCounts<<<numBlocks, blockSize>>>(d_input_id, length, d_BlocksCount);
		CUDASAFECALL(cudaDeviceSynchronize());
		//phase 2: compute exclusive prefix sum of valid block counts to get output offset for each thread block in grid
		thrust::exclusive_scan(thrustPrt_bCount, thrustPrt_bCount + numBlocks, thrustPrt_bOffset);
		//phase 3: compute output offset for each thread in warp and each warp in thread block, then output valid elements
		compactK<<<numBlocks, blockSize, sizeof(int) * (blockSize / warpSize)>>>(d_input_id, d_input_edge, length, d_output_id, d_output_edge, d_BlocksOffset);
		CUDASAFECALL(cudaDeviceSynchronize());

		// determine number of elements in the compacted list
		int compact_length = thrustPrt_bOffset[numBlocks - 1] + thrustPrt_bCount[numBlocks - 1];

		cudaFree(d_BlocksCount);
		cudaFree(d_BlocksOffset);

		return compact_length;
	}

} /* namespace cuCompactor */

namespace cuCompactor_part
{

#define warpSize (32)
#define FULL_MASK 0xffffffff
#define startimer(timer) (timer -= get_time())
#define stoptimer(timer) (timer += get_time())

	__host__ __device__ int divup(int x, int y)
	{
		return x / y + (x % y ? 1 : 0);
	}

	__device__ __inline__ int pow2i(int e)
	{
		return 1 << e;
	}

	inline double get_time()
	{
		struct timeval tv;
		gettimeofday(&tv, NULL);
		return tv.tv_sec + (tv.tv_usec / 1e6);
	}

	template <typename T>
	__global__ void computeBlockCounts(T *d_input, int length, int *d_BlockCounts, int partSize)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < length)
		{
			int part_num = (d_input[idx] + partSize - 1) / partSize;
			int BC = 0;
			for (int i = 0; i < 10; i++)
			{ // may be more than 10
				int pred = (part_num >> i) & 1;
				int count = __syncthreads_count(pred);
				BC += count << i;
			}

			if (threadIdx.x == 0)
			{
				d_BlockCounts[blockIdx.x] = BC; // BC will contain the number of valid elements in all threads of this thread block
			}
		}
	}

	template <typename T>
	__global__ void compactK(T *d_input, int length, T *d_output, int *d_BlocksOffset, int partSize)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		extern __shared__ int warpTotals[];
		if (idx < length)
		{
			int w_i = threadIdx.x / warpSize; //warp index
			int w_l = idx % warpSize;		  //thread index within a warp
			int t_m = FULL_MASK >> (warpSize - w_l);
			// compute exclusive prefix sum based on predicate validity to get output offset for thread in warp
			int part_num = (d_input[idx] + partSize - 1) / partSize;
			int t_u = 0;
			for (int i = 0; i < 10; i++)
			{
				int pred = (part_num >> i) & 1;
				int b = __ballot_sync(FULL_MASK, pred) & t_m;
				t_u += __popc(b) << i;
			}

			// last thread in warp computes total valid counts for the warp
			if (w_l == warpSize - 1)
			{
				warpTotals[w_i] = t_u + part_num;
			}

			// need all warps in thread block to fill in warpTotals before proceeding
			__syncthreads();

			// first numWarps threads in first warp compute exclusive prefix sum to get output offset for each warp in thread block
			int numWarps = blockDim.x / warpSize;
			unsigned int numWarpsMask = FULL_MASK >> (warpSize - numWarps);
			if (w_i == 0 && w_l < numWarps)
			{
				int w_i_u = 0;
				for (int j = 0; j <= 10; j++)
				{ // must include j=5 in loop in case any elements of warpTotals are identically equal to 32
#if (CUDART_VERSION < 9000)
					int b_j = __ballot(warpTotals[w_l] & pow2i(j)); //# of the ones in the j'th digit of the warp offsets
#else
					int b_j = __ballot_sync(numWarpsMask, warpTotals[w_l] & pow2i(j));
#endif
					w_i_u += (__popc(b_j & t_m)) << j;
					//printf("indice %i t_m=%i,j=%i,b_j=%i,w_i_u=%i\n",w_l,t_m,j,b_j,w_i_u);
				}
				warpTotals[w_l] = w_i_u;
			}

			// need all warps in thread block to wait until prefix sum is calculated in warpTotals
			__syncthreads();

			// if valid element, place the element in proper destination address based on thread offset in warp, warp offset in block, and block offset in grid
			if (d_input[idx] > 0)
			{
				for (int i = 0; i < part_num; i++)
				{
					d_output[t_u + i + warpTotals[w_i] + d_BlocksOffset[blockIdx.x]] = i + 1 == part_num ? d_input[idx] - i * partSize : partSize;
				}
			}
		}
	}

	template <class T>
	__global__ void printArray_GPU(T *hd_data, int size, int newline)
	{
		int w = 0;
		for (int i = 0; i < size; i++)
		{
			if (i % newline == 0)
			{
				printf("\n%i -> ", w);
				w++;
			}
			printf("%i ", hd_data[i]);
		}
		printf("\n");
	}

	template <typename T>
	int compact(T *d_input, T *d_output, int length, int blockSize, int partSize)
	{
		int numBlocks = divup(length, blockSize);
		int *d_BlocksCount;
		int *d_BlocksOffset;
		CUDASAFECALL(cudaMalloc(&d_BlocksCount, sizeof(int) * numBlocks));
		CUDASAFECALL(cudaMalloc(&d_BlocksOffset, sizeof(int) * numBlocks));
		thrust::device_ptr<int> thrustPrt_bCount(d_BlocksCount);
		thrust::device_ptr<int> thrustPrt_bOffset(d_BlocksOffset);

		//phase 1: count number of valid elements in each thread block
		computeBlockCounts<<<numBlocks, blockSize>>>(d_input, length, d_BlocksCount, partSize);
		CUDASAFECALL(cudaDeviceSynchronize());
		//phase 2: compute exclusive prefix sum of valid block counts to get output offset for each thread block in grid
		thrust::exclusive_scan(thrustPrt_bCount, thrustPrt_bCount + numBlocks, thrustPrt_bOffset);
		//phase 3: compute output offset for each thread in warp and each warp in thread block, then output valid elements
		compactK<<<numBlocks, blockSize, sizeof(int) * (blockSize / warpSize)>>>(d_input, length, d_output, d_BlocksOffset, partSize);
		CUDASAFECALL(cudaDeviceSynchronize());

		// determine number of elements in the compacted list
		int compact_length = thrustPrt_bOffset[numBlocks - 1] + thrustPrt_bCount[numBlocks - 1];

		cudaFree(d_BlocksCount);
		cudaFree(d_BlocksOffset);

		return compact_length;
	}

} /* namespace cuCompactor */

namespace cuCompactor_count {
	
#define warpSize (32)
#define FULL_MASK 0xffffffff

	__host__ __device__ int divup(int x, int y)
	{
		return x / y + (x % y ? 1 : 0);
	}

	__device__ __inline__ int pow2i(int e)
	{
		return 1 << e;
	}

	template <typename T>
	__global__ void computeBlockCounts(T *d_input, int length, int *d_BlockCounts)
	{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < length)
		{
			int pred = d_input[idx] != 0;
			int BC = __syncthreads_count(pred);

			if (threadIdx.x == 0)
			{
				d_BlockCounts[blockIdx.x] = BC; // BC will contain the number of valid elements in all threads of this thread block
			}
		}
	}

	template <typename T>
	int compact(T *d_input, int length, int blockSize)
	{
		int numBlocks = divup(length, blockSize);
		int *d_BlocksCount;
		int *d_BlocksOffset;
		CUDASAFECALL(cudaMalloc(&d_BlocksCount, sizeof(int) * numBlocks));
		CUDASAFECALL(cudaMalloc(&d_BlocksOffset, sizeof(int) * numBlocks));
		thrust::device_ptr<int> thrustPrt_bCount(d_BlocksCount);
		thrust::device_ptr<int> thrustPrt_bOffset(d_BlocksOffset);

		//phase 1: count number of valid elements in each thread block
		computeBlockCounts<<<numBlocks, blockSize>>>(d_input, length, d_BlocksCount);
		CUDASAFECALL(cudaDeviceSynchronize());
		//phase 2: compute exclusive prefix sum of valid block counts to get output offset for each thread block in grid
		thrust::exclusive_scan(thrustPrt_bCount, thrustPrt_bCount + numBlocks, thrustPrt_bOffset);

		// determine number of elements in the compacted list
		int compact_length = thrustPrt_bOffset[numBlocks - 1] + thrustPrt_bCount[numBlocks - 1];

		cudaFree(d_BlocksCount);
		cudaFree(d_BlocksOffset);

		return compact_length;
	}

}

int compact_mask(int *d_input_id, int *d_output_id, int *d_input_edge, int *d_output_edge, int length, int blockSize) {
	int re = cuCompactor_mask::compact(d_input_id, d_output_id, d_input_edge, d_output_edge, length, blockSize); 
	// print_d(d_output_edge, re, re, "edgelist: ");
	return re;
}

int compact_part(int *d_input, int *d_output, int length, int blockSize, int partSize) {
	int re = cuCompactor_part::compact(d_input, d_output, length, blockSize, partSize);
	// print_d(d_output, re, re, "partCount: ");
	return re;
}

int compact_count(int *d_input, int length, int blockSize) {
	return cuCompactor_count::compact(d_input, length, blockSize);
}

#endif /* CUCOMPACTOR_H_ */
