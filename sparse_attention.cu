#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void sparse_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int batch_size, int num_heads, int seq_len, int head_dim) {

    // Calculate thread and block indices
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int seq_id = threadIdx.x;

    // Shared memory allocation for attention scores
    extern __shared__ float shared_scores[];

    // Ensure thread does not exceed sequence length
    if (seq_id >= seq_len) return;

    // Index calculations for Q, K, and V tensors
    const int q_idx = ((batch_id * num_heads + head_id) * seq_len + seq_id) * head_dim;

    // Compute attention scores for the sequence
    for (int i = 0; i < seq_len; ++i) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += Q[q_idx + d] * K[q_idx + d];
        }
        shared_scores[seq_id * seq_len + i] = score;
    }

    // Synchronize threads to ensure all scores are computed
    __syncthreads();

    // Compute the output by weighting values (V) by attention scores
    for (int i = 0; i < head_dim; ++i) {
        float weighted_sum = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            weighted_sum += shared_scores[seq_id * seq_len + j] * V[q_idx + i];
        }
        output[q_idx + i] = weighted_sum;
    }
}

torch::Tensor sparse_attention(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V) {

    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);

    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);

    auto output = torch::zeros_like(Q);

    const dim3 blocks(batch_size, num_heads);
    const dim3 threads(seq_len);
    size_t shared_mem_size = seq_len * seq_len * sizeof(float);

    sparse_attention_kernel<<<blocks, threads, shared_mem_size>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len,
        head_dim);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_attention", &sparse_attention, "Sparse Attention (CUDA)");
}
