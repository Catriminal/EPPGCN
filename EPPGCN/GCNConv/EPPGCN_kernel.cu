#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

#define WARP_SIZE 32
// #define WARP_SIZE dimWorker

__global__ void warmup(){}

__device__ inline 
void atomicAdd_F(float* address, float value)
{
  float old = value;  
  while ((old = atomicExch(address, atomicExch(address, 0.0f)+old))!=0.0f);
}



template <typename scalar_t>
__global__ void spmm_forward_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_pointers, 
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> column_index,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> degrees,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part2Node,
    const int num_nodes, 
    const int dim,
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock
);


template <typename T>
__global__ void mask_forward_cuda_kernel(
    torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> ids,
    torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> partPointer,
    torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> edgeList,
    torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> src_mask,
    torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> ngh_mask,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> backEdgeMask,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> node_degs,
    const int num_parts,
    const bool isLastLayer
);

template <typename scalar_t>
__global__ void spmm_backward_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> column_index,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> degrees,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part2Node,
    const int num_nodes, 
    const int dim,
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock
);

template <typename scalar_t>
__global__ void ours_backward_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> ids,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> partPointer,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edgeList,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> degrees,
    const int num_parts,
    const int dim,  // multiple of 4
    const int partSize
);

////////////////////////////////////////////
//
// Basic Scatter-And-Gather kernel.
//
////////////////////////////////////////////

float for_sum_time = 0.0;
float for_com_time = 0.0;
float layer_back_sum_times[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
float back_sum_time = 0.0;
float back_com_time = 0.0;
float mask_time = 0.0;

void print_time() {
    // printf("kernel time: %.6f\n", (for_sum_time + for_com_time + back_sum_time + back_com_time)/ 1000);
    printf("for_agg_time: %.6f\n", for_sum_time / 1000);
    // printf("for_com_time: %.6f\n", for_com_time / 1000);
    printf("back_agg_time: %.6f\n", back_sum_time / 1000);
    // printf("back_com_time: %.6f\n", back_com_time / 1000);
    for(int i = 0; i < 6 && layer_back_sum_times[i] != 0.0; i++) {
        printf("l%d_back_agg_time: %.6f\n", i + 1, layer_back_sum_times[i] / 1000);
    }
    printf("mask_time: %.6f\n", mask_time / 1000);
}

void clear_time() {
    for_sum_time = 0.0;
    for_com_time = 0.0;
    back_sum_time = 0.0;
    back_com_time = 0.0;
    for(int i = 0; i < 6; i++) {
        layer_back_sum_times[i] = 0.0;
    }
}

////////////////////////////////////////////
//
// Foward Pass (GCN)  node update --> neighbor aggregation
//
////////////////////////////////////////////
std::vector<torch::Tensor> spmm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock
) 
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time = 0.0;

    cudaEventRecord(start, 0);
    auto tmp = torch::mm(input, weight);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    for_com_time += time;

    // auto output = torch::zeros_like(tmp);
    auto output = torch::zeros({input.size(0), weight.size(1)}, torch::kCUDA);
    const int dim = output.size(1);
    const int num_nodes = output.size(0);
    const int num_parts = part2Node.size(0);

    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (num_parts * WARP_SIZE + block  - 1) / block; 
    int shared_memory = partSize*warpPerBlock*sizeof(int)+warpPerBlock*dim*sizeof(float);

    // printf("grid: %d, block: %d\n", grid, block);
    // printf("dim: %d, num_nodes: %d, num_parts: %d\n", dim, num_nodes, num_parts);
    // printf("input: (%d, %d)\n", tmp.size(0), tmp.size(1));
    // printf("dimWorker: %d\n", dimWorker);
    // printf("shared_memory: %d\n", tmp.size(0), tmp.size(1));


    cudaEventRecord(start, 0);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "spmm_cuda_forward", ([&] {
                                spmm_forward_cuda_kernel<scalar_t><<<grid, block, shared_memory>>>(
                                    output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    tmp.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    row_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(), 
                                    column_index.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    degrees.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                                    part_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(), 
                                    part2Node.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    num_nodes, 
                                    dim,
                                    num_parts,
                                    partSize,
                                    dimWorker,
                                    warpPerBlock
                                );
                            }));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    for_sum_time += time;
                                 
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    return {output};
}

template <typename scalar_t>
__global__ void spmm_forward_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_pointers, 
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> column_index,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> degrees,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part2Node,
    const int num_nodes, 
    const int dim,
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock
) {

    int tid =  blockIdx.x * blockDim.x + threadIdx.x;  // global thread-id
    int warpId = tid / WARP_SIZE;                             // global warp-id
    int block_warpId = threadIdx.x / WARP_SIZE;               // block warp-id
    int laneid = threadIdx.x % WARP_SIZE;                     // warp thread-id -- laneid

    extern __shared__ int part_meta[];                                      // part information.
    int *partial_ids = part_meta;                                           // caching ids
    float *partial_results = (float*)&part_meta[partSize*warpPerBlock];     // caching partial results.

    if (warpId < num_parts){

        int srcId = part2Node[warpId];              // aggregated source node
        int partBeg = part_pointers[warpId];        // partitioning pointer start
        int partEnd = part_pointers[warpId + 1];    // part pointer end
        float src_norm = degrees[srcId];            // norm of the source node

        // Cache the part neighbors by all threads from a warp.
        const int pindex_base = block_warpId * partSize;
        #pragma unroll
        for (int nidx = partBeg + laneid; nidx < partEnd; nidx += WARP_SIZE){
            // if(column_index[nidx] >= num_nodes || column_index[nidx] < 0) printf("column_index: %d\n", column_index[nidx]);
            partial_ids[pindex_base + nidx - partBeg] = column_index[nidx];
        }
        
        // #pragma unroll
        // for (int nidx = partBeg; nidx < partEnd; nidx++){
        // //     if(column_index[nidx] >= num_nodes || column_index[nidx] < 0) printf("column_index: %d\n", column_index[nidx]);
        //     partial_ids[nidx - partBeg] = column_index[nidx];
        // }
        
        __syncwarp();

        // if (laneid == 0)
        // for (int nIdx = laneid; nIdx < partEnd - partBeg; nIdx++){
            // int nid = partial_ids[pindex_base + nIdx];
            // int nid = partial_ids[nIdx];
            // printf("verify nid - 111111: %d\n", nid);
            // if(nid >= num_nodes || nid < 0) printf("verify nid: %d\n", nid);
        // }

        // Neighbor aggregation within each part
        const int presult_base = block_warpId * dim;
        for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++)
        {
            int nid = partial_ids[pindex_base + nIdx];
            // int nid = partial_ids[nIdx];
            // if (laneid == 0)
            //     printf("verify nid - 222222: %d\n", nid);
            float degree_norm_inv = __fmaf_rn(src_norm, degrees[nid], 0);

            // Initialize shared memory for partial results
            if (nIdx == 0)
                if (laneid < dimWorker)
                #pragma unroll
                for (int d = laneid; d < dim; d += dimWorker){
                    partial_results[presult_base + d] = 0.0f;
                }
            
            if (laneid < dimWorker)
            #pragma unroll
            for (int d = laneid; d < dim; d += dimWorker){
                // if(nid >= num_nodes || nid < 0) printf("aggregation: %d\n", nid);
                partial_results[presult_base + d] += __fmaf_rn(degree_norm_inv, input[nid][d], 0);
                // partial_results[presult_base + d] += input[nid][d];
            }
        }

        // output the result to global memory from the shared memory
        if (laneid < dimWorker)
        #pragma unroll
        for (int d = laneid; d < dim; d += dimWorker){
            atomicAdd_F((float*)&output[srcId][d], partial_results[presult_base + d]);
        }
    }
}

void mask_forward_cuda(
    torch::Tensor id,
    torch::Tensor partPointer,
    torch::Tensor edgeList,
    torch::Tensor src_mask,
    torch::Tensor ngh_mask,
    torch::Tensor backEdgeMask,
    torch::Tensor node_degs,
    int num_layers,
    int layer,
    int blockx, 
    int blocky
) {

    const int num_parts = id.size(0);
    dim3 blocksize(blockx, blocky);
    const int grid = num_parts / blocksize.y + 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time = 0.0;
    
    cudaEventRecord(start, 0);
    AT_DISPATCH_ALL_TYPES(id.type(), "mask_forward_cuda__", ([&] {
                        mask_forward_cuda_kernel<int><<<grid, blocksize>>>(
                            id.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                            partPointer.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                            edgeList.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                            src_mask.packed_accessor32<bool,1,torch::RestrictPtrTraits>(),
                            ngh_mask.packed_accessor32<bool,1,torch::RestrictPtrTraits>(),
                            backEdgeMask.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                            node_degs.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                            num_parts,
                            layer == num_layers
                        );
                    }));
    

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    mask_time += time;

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

template <typename T>
__global__ void mask_forward_cuda_kernel(
    torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> ids,
    torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> partPointer,
    torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> edgeList,
    torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> src_mask,
    torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> ngh_mask,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> backEdgeMask,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> node_degs,
    const int num_parts,
    const bool isLastLayer
) {
    int partId = blockDim.y * blockIdx.x + threadIdx.y;

	if(partId < num_parts)
	{
        T id = ids[partId];
        T partStart = partPointer[partId];

        T size = partPointer[partId + 1] - partPointer[partId];
        int mask_active_count = 0;
		for(int i = 0; i < size; i++)
		{	
			T end_id = edgeList[partStart + i];

            if(!isLastLayer && src_mask[id]) {
                ngh_mask[id] = true;
                ngh_mask[end_id] = true;
            }

            if(src_mask[end_id]) {
                backEdgeMask[partStart + i] = id;
            }
            mask_active_count += threadIdx.x == 0 && src_mask[end_id];
		}
        atomicAdd(&(node_degs[id]), mask_active_count);
	}
}

////////////////////////////////////////////
// 
// backward pass (GCN)
//
////////////////////////////////////////////
std::vector<torch::Tensor> spmm_backward_cuda(
    torch::Tensor d_output,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock
) {

    auto d_input_prime = torch::zeros_like(d_output);

    const int dim = d_input_prime.size(1);
    const int num_nodes = d_input_prime.size(0);
    const int num_parts = part2Node.size(0);

    const int block = warpPerBlock*WARP_SIZE;
    const int grid = (num_parts*WARP_SIZE + block - 1) / block; 
    // const int shared_memory = warpPerBlock * partSize * sizeof(int) + warpPerBlock * dim * sizeof(float);
    int shared_memory = partSize*warpPerBlock*sizeof(int)+warpPerBlock*dim*sizeof(float);

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    AT_DISPATCH_FLOATING_TYPES(d_output.type(), "spmm_cuda_backward", ([&] {
                                spmm_backward_cuda_kernel<scalar_t><<<grid, block, shared_memory>>>(
                                    d_input_prime.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    d_output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    row_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    column_index.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    degrees.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                                    part_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(), 
                                    part2Node.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    num_nodes, 
                                    dim,
                                    num_parts,
                                    partSize,
                                    dimWorker,
                                    warpPerBlock
                                );
                            }));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);

    back_sum_time += time;

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaEventRecord(start, 0);
    auto d_input = torch::mm(d_input_prime, W.transpose(0,1));
    auto d_weight = torch::mm(X.transpose(0,1), d_input_prime);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    back_com_time += time;
    return {d_input, d_weight};
}

template <typename scalar_t>
__global__ void spmm_backward_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> column_index,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> degrees,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part2Node,
    const int num_nodes, 
    const int dim,
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock
) {

    int tid =  blockIdx.x * blockDim.x + threadIdx.x;
    int warpId =  tid / WARP_SIZE;
    int block_warpId = threadIdx.x / WARP_SIZE;
    int laneid = threadIdx.x % WARP_SIZE;

    extern __shared__ int part_meta[];                                      // part information.
    int *partial_ids = part_meta;                                           // caching ids
    float *partial_results = (float*)&part_meta[partSize*warpPerBlock];     // caching partial results.

    if (warpId < num_parts){

        const int srcId = part2Node[warpId];
        const int partBeg = part_pointers[warpId];
        const int partEnd = part_pointers[warpId + 1];
        float src_norm = degrees[srcId];

        const int pindex_base = block_warpId * partSize;
        #pragma unroll
        for (int nid = partBeg + laneid; nid < partEnd; nid += WARP_SIZE){
            partial_ids[pindex_base + nid - partBeg] = column_index[nid];
        }
        
        // #pragma unroll
        // for (int nidx = partBeg; nidx < partEnd; nidx++){
        // //     if(column_index[nidx] >= num_nodes || column_index[nidx] < 0) printf("column_index: %d\n", column_index[nidx]);
        //     partial_ids[nidx - partBeg] = column_index[nidx];
        // }

        __syncwarp();

        const int presult_base = block_warpId * dim;
        for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++)
        {
            int nid = partial_ids[pindex_base + nIdx];
            // int nid = partial_ids[nIdx];
            float degree_norm =  __fmaf_rn(src_norm, degrees[nid], 0);

            if (nIdx == 0)
                if (laneid < dimWorker)
                #pragma unroll
                for (int d = laneid; d < dim; d += dimWorker){
                    partial_results[presult_base + d] = 0;
                }
            
            if (laneid < dimWorker)
            #pragma unroll
            for (int d = laneid; d < dim; d += dimWorker){
                partial_results[presult_base + d] += __fmaf_rn(degree_norm, d_output[nid][d], 0);
            }
        }

        if (laneid < dimWorker)
        #pragma unroll
        for (int d = laneid; d < dim; d += dimWorker){
            atomicAdd_F((float*)&d_input[srcId][d], partial_results[presult_base + d]);
        }
    }
}

std::vector<torch::Tensor> ours_backward_cuda(
    torch::Tensor d_output,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor id,
    torch::Tensor partPointer,
    torch::Tensor edgeList,
    torch::Tensor degrees,
    int partSize, 
    int numParts,
    int layer,
    int blockx, 
    int blocky
) {

    auto d_input_prime = torch::zeros_like(d_output);

    const int dim = d_input_prime.size(1);
    const int num_nodes = d_input_prime.size(0);

    dim3 blocksize(blockx, blocky);
    const int grid = numParts / blocksize.y + 1;
    size_t shared_mem_size = dim * blocksize.y * sizeof(float);

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // std::cout << "before backward kernel " << numParts << " " << dim << " " << partSize << " " << blockx << " " << blocky << " " << num_nodes << std::endl;
    AT_DISPATCH_FLOATING_TYPES(d_output.type(), "ours_backward_cuda", ([&] {
                                ours_backward_cuda_kernel<scalar_t><<<grid, blocksize, shared_mem_size>>>(
                                    d_input_prime.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    d_output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    id.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    partPointer.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    edgeList.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    degrees.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                                    numParts,
                                    dim,
                                    partSize
                                );
                            }));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    back_sum_time += time;
    layer_back_sum_times[layer - 1] += time;
    // std::cout << "after backward kernel" << std::endl;
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // auto mask = torch::randint(0, X.sizes()[0], int(X.sizes()[0]*0.1), torch::dtype(torch::kLong)).to(X.device()); \\using mask in combination
    // std::cout<<"info"<<std::endl;
    // std::cout<<X.sizes()[0]<<"  "<<mask.sizes()[0]<<std::endl;
    // std::cout<<mask.device()<<std::endl;
    // std::cout<<mask.device().type()<<std::endl;

    cudaEventRecord(start, 0);
    auto d_input = torch::mm(d_input_prime, W.transpose(0,1));
    auto d_weight = torch::mm(X.transpose(0,1), d_input_prime);
    // auto d_weight = torch::mm(X.index({mask}).transpose(0,1), d_input_prime.index({mask})); \\using mask in combination
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    back_com_time += time;
    // std::cout << "after backward ready return" << std::endl;
    return {d_input, d_weight};
}

template <typename scalar_t>
__global__ void ours_backward_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> ids,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> partPointer,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edgeList,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> degrees,
    const int num_parts,
    const int dim,
    const int partSize
) {
    int partId = blockDim.y * blockIdx.x + threadIdx.y;

	if(partId < num_parts)
	{
		int partStart = partPointer[partId];
        // if(threadIdx.x == 0) {
        //     printf("%d %d %d\n", partPointer[partId + 1], partPointer[partId], partId);
        // }
		int id = ids[partStart];
		
		extern __shared__ float feats[];
		
		for(int j = threadIdx.x; j < dim; j += blockDim.x)
			feats[threadIdx.y * dim + j] = 0;

		int size = partPointer[partId + 1] - partPointer[partId];
		for(int i = 0; i < size; i++)
		{	
			int offset = partStart + i;
			int end_id = edgeList[offset];
            float src_norm = degrees[id];
            float weight =  __fmaf_rn(src_norm, degrees[end_id], 0);
            
            for(int j = threadIdx.x; j < dim; j += blockDim.x) {
                feats[threadIdx.y * dim + j] += d_output[end_id][j] * weight;
            }
		}

		for(int j = threadIdx.x; j < dim; j += blockDim.x) {
            atomicAdd(&(d_input[id][j]), feats[threadIdx.y * dim + j]);
		}
	}
}