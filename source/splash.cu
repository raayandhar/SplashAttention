#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_M 32
#define BLOCK_N 32 // should be a multiple of BLOCK_M
#define D_MAX 128

__device__ inline float load_or(const float* ptr, int idx, int max_idx, float other) {
    if (idx < 0 || idx >= max_idx) {
        return other;
    }
    return ptr[idx];
}

/*
Sparse flash attention kernel:

This implements the hash-based Sparse Causal Flash Attention (SCFA) from the 2023 paper by Pagliardini et al:
https://arxiv.org/pdf/2306.01160
*/
__global__
void sparse_attention_forward_kernel(
    // Q, K, V
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    // Q_idx, K_idx
    const int* __restrict__ Q_idx,
    const int* __restrict__ K_idx,
    // for storing partial sums
    float* __restrict__ L,
    float* __restrict__ M,
    // final output
    float* __restrict__ Out,
    // metadata
    const float softmax_scale,
    int B,
    int H,
    int NQ,
    int NK,
    int d
)
{

    /* Block-wise layout:
    A 2D grid of blocks, where each block column on the y-dimension handles one batch & head (B*H columns, each column handles data shape [seq_len, dim]).
    Each block within its column handles BLOCK_M queries. For queries, our data shape is [BLOCK_M, dim].
    Each thread within a block handles a single query.

    Q_idx and K_idx refer to the Q and K original sequence indices.
    */
    int block_m_id = blockIdx.x; 
    int bh_id = blockIdx.y; 
    int thread_id = threadIdx.x; 

    int b = bh_id / H;  // batch index
    int h = bh_id % H;  // head index

    size_t bh_offset_qkv = (size_t)b * H * NQ * d + (size_t)h * NQ * d;
    size_t bh_offset_LM = (size_t)(b * H + h) * NQ;

    int start_m = block_m_id * BLOCK_M;
    int q_i   = start_m + thread_id;  // local index for the queries

    size_t q_offset = bh_offset_qkv + (size_t)q_i * d;

    // ======= Load query vector & L and M values into thread registers =======
    if (d > D_MAX) {
        printf("Error: Kernel not engineered for d > %d\n", D_MAX);
        return;
    }
    float q_val[D_MAX]; 
    #pragma unroll
    for (int dim = 0; dim < d; dim++) {
        q_val[dim] = 0.0f;
    }

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    // "masked" loads
    if (q_i < NQ) {
        #pragma unroll
        for (int dim = 0; dim < d; dim++) {
            q_val[dim] = Q[q_offset + dim];
        }
        l_prev = L[bh_offset_LM + q_i]; 
        m_prev = M[bh_offset_LM + q_i]; 
    }

    // ======= Load query sequence indices into shared memory =======
    // Each thread has one query index, and we synchronize between threads in the block
    // to get the max query index in the block.

    int Qi = -1; // Qi is the sequence index for the current query
    if (q_i < NQ) {
        size_t bh_offset_idxQ = (size_t)b * H * NQ + (size_t)h * NQ;
        Qi = Q_idx[bh_offset_idxQ + q_i];
    }

    __shared__ int qi_sh[BLOCK_M];
    if (thread_id < BLOCK_M) {
        qi_sh[thread_id] = (q_i < NQ) ? Qi : -1; 
    }
    __syncthreads();
    // TODO: profile - if slow, replace with block-wide reduction
    // right now thread 0 does a linear scan
    if (thread_id == 0) {
        int block_max = -1;
        for (int i = 0; i < BLOCK_M; i++) {
            block_max = max(block_max, qi_sh[i]);
        }
        qi_sh[0] = block_max;
    }
    __syncthreads();
    int block_max_Qi = qi_sh[0];
    int end_n_blocks = 0; // accumulator for the number of blocks we need to process (e.g blocks that fall below the causal mask)
    // Each thread processes NK / BLOCK_M indices of K
    for (int start_n = 0; start_n < NK; start_n += BLOCK_N) {
        // Within this loop, we're processing a single block of K. Each thread
        // processes BLOCK_N / BLOCK_M keys.
        float local_min = INFINITY;
        for (int x = thread_id; x < BLOCK_N; x += blockDim.x) {
            // Get smallest K index among the keys this thread is handling
            int kn = start_n + x;
            int val = (kn < NK)
                ? K_idx[(b * H + h) * NK + kn]
                : INFINITY; 
            if (val < local_min) {
                local_min = (float)val;
            }
        }
        // reduce among threads
        __shared__ float min_sh[BLOCK_M];
        min_sh[thread_id] = local_min;
        __syncthreads();

        // block-wide reduce min. TODO: replace with actual block-wide reduction
        if (thread_id == 0) {
            float block_min = INFINITY;
            for (int i = 0; i < BLOCK_M; i++) {
                if (min_sh[i] < block_min) {
                    block_min = min_sh[i];
                }
            }
            min_sh[0] = block_min;
        }
        __syncthreads();
        float block_min_ki = min_sh[0];

        // If the min_ki <= max_Qi, we need that block - some query can attend to a prior key
        // If block_min_ki == INFINITY, that means it's out of range
        if (block_min_ki <= block_max_Qi && block_min_ki < INFINITY) {
            end_n_blocks += 1;
        } else {
            // no more blocks needed
        }
    }

    // ======= Load K and V into shared memory =======
    // Each thread loads BLOCK_N / BLOCK_M keys and values
    // We load the keys and values into shared memory, and then use them in the dot product
    // with the query vector.

    // ======= Initialize accumulator =======
    float acc[D_MAX];
    #pragma unroll
    for (int dim = 0; dim < d; dim++) {
        acc[dim] = 0.0f;
    }
    int n_blocks_done = 0;
    for (int start_n = 0; start_n < NK && n_blocks_done < end_n_blocks; start_n += BLOCK_N) {
        float local_min = INFINITY;
        // In this block, we process BLOCK_N / BLOCK_M keys per thread.

        // Find min key index in block
        for (int x = thread_id; x < BLOCK_N; x += blockDim.x) {
            int kn = start_n + x;
            int val = (kn < NK)
                ? K_idx[(b * H + h) * NK + kn]
                : INFINITY; 
            if (val < local_min) {
                local_min = (float)val;
            }
        }
        __shared__ float min_sh2[BLOCK_M];
        min_sh2[thread_id] = local_min;
        __syncthreads();
        if (thread_id == 0) {
            float block_min_ki = INFINITY;
            for (int i = 0; i < BLOCK_M; i++) {
                block_min_ki = fminf(block_min_ki, min_sh2[i]);
            }
            min_sh2[0] = block_min_ki;
        }
        __syncthreads();
        float block_min_ki = min_sh2[0];
        // check if we skip
        if (block_min_ki > block_max_Qi || block_min_ki >= INFINITY) {
            // skip
            continue;
        }
        n_blocks_done += 1;

        // ====== Compute dot products for loaded BLOCK_M queries * BLOCK_N keys =======
        float qk[BLOCK_N];
        #pragma unroll
        for (int i = 0; i < BLOCK_N; i++) {
            qk[i] = -INFINITY;  
        }

        // Load keys & values into shared mem - threads individually hold queries
        __shared__ float k_sh[BLOCK_N * D_MAX];
        __shared__ float v_sh[BLOCK_N * D_MAX];
        __shared__ int   ki_sh[BLOCK_N];

        for (int x = thread_id; x < BLOCK_N; x += blockDim.x) {
            int kn = start_n + x;
            if (kn < NK) {
                ki_sh[x] = K_idx[(b * H + h) * NK + kn];
            } else {
                ki_sh[x] = INFINITY;
            }
        }

        for (int x = thread_id; x < BLOCK_N * d; x += blockDim.x) {
            int block_k = x / d;  // 0..BLOCK_N-1
            int dim_k   = x % d;
            int kn = start_n + block_k;
            float kval = 0.0f;
            float vval = 0.0f;
            if (kn < NK) {
                size_t offset_k = bh_offset_qkv + (size_t)kn * d;
                kval = K[offset_k + dim_k];
                vval = V[offset_k + dim_k];
            }
            k_sh[x] = kval;
            v_sh[x] = vval;
        }
        __syncthreads();

        if (q_i < NQ) {
            for (int j = 0; j < BLOCK_N; j++) {
                // check if j < NK
                int kn = start_n + j;
                if (kn >= NK) {
                    break; // out of range
                }
                // causal mask: if Qi < Ki => -inf
                if (Qi >= ki_sh[j]) {
                    // do dot
                    float sum = 0.f;
                    #pragma unroll
                    for (int dim_k = 0; dim_k < d; dim_k++) {
                        sum += q_val[dim_k] * k_sh[j * d + dim_k];
                    }
                    sum *= softmax_scale;
                    qk[j] = sum;
                } else {
                    qk[j] = -INFINITY;
                }
            }
        }
        __syncthreads();

        float local_max = -INFINITY;
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j++) {
            float val = qk[j];
            if (val > local_max) {
                local_max = val;
            }
        }
        float m_curr = fmaxf(local_max, m_prev);

        // compute p = exp(qk - m_curr), scaled old terms
        float sum_p = 0.f;
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j++) {
            if (qk[j] > -INFINITY) {
                float val = expf(qk[j] - m_curr);
                qk[j] = val;  // reuse
                sum_p += val;
            } else {
                qk[j] = 0.f;
            }
        }
        // combine with old l
        float l_curr = sum_p + l_prev * expf(m_prev - m_curr);

        if (q_i < NQ && l_curr > 0.f) {
            float scale_old = (l_prev == 0.f) ? 0.f : (l_prev * expf(m_prev - m_curr)) / l_curr;
            float scale_new = 1.f / l_curr;
            #pragma unroll
            for (int dim_k = 0; dim_k < d; dim_k++) {
                acc[dim_k] = acc[dim_k] * scale_old; 
            }
            // Now accumulate p * V
            for (int j = 0; j < BLOCK_N; j++) {
                float p_ij = qk[j] * scale_new;
                // add dot with v_sh
                #pragma unroll
                for (int dim_k = 0; dim_k < d; dim_k++) {
                    acc[dim_k] += p_ij * v_sh[j * d + dim_k];
                }
            }
            // update l_prev, m_prev
            l_prev = l_curr;
            m_prev = m_curr;
        }
        __syncthreads();
    } // end loop over blocks in K

    if (q_i < NQ) {
        L[bh_offset_LM + q_i] = l_prev;
        M[bh_offset_LM + q_i] = m_prev;
        size_t out_offset = bh_offset_qkv + (size_t)q_i * d;
        for (int dim_k = 0; dim_k < d; dim_k++) {
            Out[out_offset + dim_k] = acc[dim_k];
        }
    }
}


torch::Tensor forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor Q_idx,
    torch::Tensor K_idx,
    float sm_scale
) {
    // Expect shapes [B,H,NQ,d], [B,H,NK,d], ...
    int B = Q.size(0);
    int H = Q.size(1);
    int NQ = Q.size(2);
    int d  = Q.size(3);
    int NK = K.size(2);

    auto Out = torch::zeros_like(Q);
    auto L = torch::zeros({B, H, NQ}, Q.options().dtype(torch::kFloat32));
    auto M = torch::full({B, H, NQ}, -INFINITY, Q.options().dtype(torch::kFloat32));

    dim3 block_dim(BLOCK_M);
    dim3 grid_dim(
        (NQ + BLOCK_M - 1) / BLOCK_M, // # blocks covering all queries
        B * H                         // batch*head
    );

    sparse_attention_forward_kernel<<<grid_dim, block_dim>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        Q_idx.data_ptr<int>(),
        K_idx.data_ptr<int>(),
        L.data_ptr<float>(),
        M.data_ptr<float>(),
        Out.data_ptr<float>(),
        sm_scale,
        B, H, NQ, NK, d
    );
    return Out;
}
