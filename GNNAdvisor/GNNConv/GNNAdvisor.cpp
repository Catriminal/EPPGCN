#include <torch/extension.h>
#include <vector>
#include <string>
using namespace std;
torch::Tensor SAG_cuda(
    torch::Tensor input,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock
);


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
);

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
  );

std::vector<torch::Tensor> ours_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor id,
    torch::Tensor partPointer,
    torch::Tensor edgeList,
    torch::Tensor degrees,
    int partSize, 
    int blockx, 
    int blocky
);

void mask_forward_cuda(
    torch::Tensor id,
    torch::Tensor partPointer,
    torch::Tensor edgeList,
    torch::Tensor src_mask,
    torch::Tensor ngh_mask,
    torch::Tensor backEdgeMask,
    torch::Tensor node_degs,
    int layer,
    int blockx, 
    int blocky
);

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
);

void exclusive_scan(int *part_count, int *part_pointer, int num_parts);

int compact_mask(int *d_input_id, int *d_output_id, int *d_input_edge, int *d_output_edge, int length, int blockSize);
int compact_part(int *d_input, int *d_output, int length, int blockSize, int partSize);
int compact_count(int *d_input, int length, int blockSize);

void print_time();
void clear_time();


std::vector<torch::Tensor> spmm_forward_cuda_gin(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    float epsilon,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock
  );

std::vector<torch::Tensor> spmm_backward_cuda_gin(
    torch::Tensor d_output,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    float epsilon,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock
  );

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor SAG(
    torch::Tensor input,
    torch::Tensor row_pointers,
    torch::Tensor column_index, 
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock) 
{
  CHECK_INPUT(input);
  CHECK_INPUT(row_pointers);
  CHECK_INPUT(column_index);
  CHECK_INPUT(degrees);
  CHECK_INPUT(part_pointers);
  CHECK_INPUT(part2Node);

  return SAG_cuda(input, row_pointers, column_index, 
              degrees, part_pointers, part2Node, 
              partSize, dimWorker, warpPerBlock);
}


std::vector<torch::Tensor> spmm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor row_pointers,
    torch::Tensor column_index, 
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock) 
{
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(row_pointers);
  CHECK_INPUT(column_index);
  CHECK_INPUT(degrees);
  CHECK_INPUT(part_pointers);
  CHECK_INPUT(part2Node);

  return spmm_forward_cuda(input, weight, row_pointers, column_index, 
                            degrees, part_pointers, part2Node, 
                            partSize, dimWorker, warpPerBlock);
}

std::vector<torch::Tensor> spmm_backward(
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

  CHECK_INPUT(d_output);
  CHECK_INPUT(X);
  CHECK_INPUT(W);
  CHECK_INPUT(row_pointers);
  CHECK_INPUT(column_index);
  CHECK_INPUT(degrees);
  CHECK_INPUT(part_pointers);
  CHECK_INPUT(part2Node);

  return spmm_backward_cuda(d_output, X, W, row_pointers, column_index, 
                            degrees, part_pointers, part2Node,
                            partSize, dimWorker, warpPerBlock);
}

std::vector<torch::Tensor> ours_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor id,
    torch::Tensor partPointer,
    torch::Tensor edgeList,
    torch::Tensor degrees,
    int partSize, 
    int blockx, 
    int blocky
  ) {
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(id);
  CHECK_INPUT(partPointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(degrees);

  return ours_forward_cuda(input, weight, id, partPointer, 
                            edgeList, degrees, partSize, blockx, blocky);
}

void mask_forward(
    torch::Tensor id,
    torch::Tensor partPointer,
    torch::Tensor edgeList,
    torch::Tensor src_mask,
    torch::Tensor ngh_mask,
    torch::Tensor backEdgeMask,
    torch::Tensor node_degs,
    int layer,
    int blockx, 
    int blocky
  ) {
  CHECK_INPUT(id);
  CHECK_INPUT(partPointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(src_mask);
  CHECK_INPUT(ngh_mask);
  CHECK_INPUT(backEdgeMask);
  CHECK_INPUT(node_degs);

  mask_forward_cuda(id, partPointer, edgeList, src_mask, ngh_mask, 
                            backEdgeMask, node_degs, layer, blockx, blocky);
}

std::vector<torch::Tensor> ours_backward(
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

  CHECK_INPUT(d_output);
  CHECK_INPUT(X);
  CHECK_INPUT(W);
  CHECK_INPUT(id);
  CHECK_INPUT(partPointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(degrees);

  // std::cout << "before ours_backward_cuda." << std::endl;
  return ours_backward_cuda(d_output, X, W, id, partPointer, 
                            edgeList, degrees, partSize,
                            numParts, layer, blockx, blocky);
}

////////////////////////////////
// spmm forward GIN
///////////////////////////////
std::vector<torch::Tensor> spmm_forward_gin(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor row_pointers,
    torch::Tensor column_index, 
    float epsilon,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock
  ) {
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(row_pointers);
  CHECK_INPUT(column_index);
  CHECK_INPUT(part_pointers);
  CHECK_INPUT(part2Node);

  return spmm_forward_cuda_gin(input, weight, row_pointers, column_index, 
                              epsilon, part_pointers, part2Node, 
                              partSize, dimWorker, warpPerBlock);
}

////////////////////////////////
// spmm backward GIN
///////////////////////////////
std::vector<torch::Tensor> spmm_backward_gin(
    torch::Tensor d_output,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    float epsilon,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock
  ) {
  CHECK_INPUT(d_output);
  CHECK_INPUT(X);
  CHECK_INPUT(W);
  CHECK_INPUT(row_pointers);
  CHECK_INPUT(column_index);
  CHECK_INPUT(part_pointers);
  CHECK_INPUT(part2Node);

  return spmm_backward_cuda_gin(d_output, X, W, row_pointers, column_index, 
                            epsilon, part_pointers, part2Node,
                            partSize, dimWorker, warpPerBlock);
}


std::vector<torch::Tensor> build_part(
    int partSize, 
    torch::Tensor indptr
  ) {

  auto indptr_acc = indptr.accessor<int, 1>();
  int num_nodes = indptr.size(0) - 1;
  int degree, thisNumParts, numParts = 0;

	for(int i = 0; i < num_nodes; i++)
	{
    degree = indptr_acc[i + 1] - indptr_acc[i];
	  if(degree % partSize == 0)
			thisNumParts = degree / partSize;
    else
			thisNumParts = degree / partSize + 1;
    numParts += thisNumParts;
	}

  auto partPtr = torch::zeros(numParts + 1);
  auto part2Node = torch::zeros(numParts);
	
  int part_counter = 0;
	for(int i = 0; i < num_nodes; i++)
	{
    int degree = indptr_acc[i + 1] - indptr_acc[i];
    if(degree % partSize == 0)
			thisNumParts = degree / partSize ;
    else
			thisNumParts = degree / partSize + 1;

    for (int pid = 0; pid < thisNumParts; pid++){
      int partBeg = indptr_acc[i] + pid * partSize;
      int partEnd = partBeg + partSize < indptr_acc[i  + 1]? partBeg + partSize: indptr_acc[i + 1];
      partPtr[part_counter] = partBeg;
      part2Node[part_counter++] = i;
      if (i == num_nodes - 1 &&  partEnd == indptr_acc[i + 1])
        partPtr[part_counter] = partEnd;
    }
	}
  return {partPtr, part2Node};
}

std::vector<torch::Tensor> build_back_part(
    torch::Tensor edge_mask,
    torch::Tensor column_index,
    torch::Tensor node_deg,
    int back_part_size
  ) {
  int num_nodes = node_deg.size(0);
  int num_edges = edge_mask.size(0);

  auto id = torch::zeros_like(edge_mask);
  auto edge_list = torch::zeros_like(column_index);
  // std::cout << "mask before" << std::endl;
  int valid_len = compact_mask(edge_mask.data_ptr<int>(), id.data_ptr<int>(), column_index.data_ptr<int>(), edge_list.data_ptr<int>(), num_edges, 1024);
  // std::cout << "mask after " << valid_len << std::endl;
  if(back_part_size == 0) {
    // if(model_input == "") {
      int valid_node = compact_count(node_deg.data_ptr<int>(), num_nodes, 1024);
      int valid_degree = valid_len / valid_node;
      if(valid_degree < 4) {
        back_part_size = 4;
      } else if(valid_degree >= 4 && valid_degree < 16) {
        back_part_size = 8;
      } else if(valid_degree >= 16 && valid_degree < 64) {
        back_part_size = 16;
      } else if(valid_degree >= 64 && valid_degree < 256) {
        back_part_size = 32;
      } else if(valid_degree >= 256 && valid_degree < 512) {
        back_part_size = 64;
      }
    // } else {
      // string cmd = "/home/yc/param_opt/param_pred.py --dataset " + model_input;
      // FILE *fp;
      // if(fp = popen(cmd.data(), "r")) {
      //   char line[50];
      //   if(fgets(line, sizeof(line) - 1, fp) != NULL) {
      //     back_part_size = atoi(line);
      //   }
      // }
      // pclose(fp);
    // }
  }

  auto part_count = torch::zeros_like(column_index);
  auto part_pointer = torch::zeros_like(column_index);
  // std::cout << "part before" << std::endl;
  int num_parts = compact_part(node_deg.data_ptr<int>(), part_count.data_ptr<int>(), num_nodes, 1024, back_part_size);
  // std::cout << "part after " << num_parts << std::endl;
  exclusive_scan(part_count.data_ptr<int>(), part_pointer.data_ptr<int>(), num_parts);
  // std::cout << "scan after" << std::endl;

  auto info = torch::zeros(2);
  info[0] = back_part_size;
  info[1] = num_parts;
  // std::cout << "info after" << std::endl;
  return {id, edge_list, part_pointer, info};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("SAG", &SAG, "GNNAdvisor base Scatter-and-Gather Kernel (CUDA)");

  m.def("forward", &spmm_forward, "GNNAdvisor forward (CUDA)");
  m.def("backward", &spmm_backward, "GNNAdvisor backward (CUDA)");
  m.def("ours_forward", &ours_forward, "ours forward (CUDA)");
  m.def("mask_forward", &mask_forward, "mask forward (CUDA)");
  m.def("ours_backward", &ours_backward, "ours backward (CUDA)");
  m.def("print_time", &print_time, "print time");
  m.def("clear_time", &clear_time, "clear time");

  m.def("forward_gin", &spmm_forward_gin, "GNNAdvisor forward GIN (CUDA)");
  m.def("backward_gin", &spmm_backward_gin, "GNNAdvisor forward GIN (CUDA)");

  m.def("build_part", &build_part, "GNNAdvisor backward (CPU)");
  m.def("build_back_part", &build_back_part, "ours build back part (CUDA)");
  }