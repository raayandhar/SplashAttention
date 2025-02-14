import torch
import time
import sparse_attention  # Your compiled extension

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define dimensions
batch_size = 64
seq_len = 512
head_dim = 64
num_heads = 8

# Generate random inputs
Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)
K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)
V = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)

# Define masks for sparse attention
sparse_mask = torch.randint(0, 2, (seq_len, seq_len), device=device, dtype=torch.bool)

# Benchmark regular attention
def regular_attention(Q, K, V):
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_weights = torch.softmax(attn_scores, dim=-1)
    return torch.matmul(attn_weights, V)

# Benchmark sparse attention
def sparse_attention_benchmark(Q, K, V, mask):
    return sparse_attention.forward(Q, K, V, mask)

# Timing regular attention
start = time.time()
for _ in range(10):
    output = regular_attention(Q, K, V)
torch.cuda.synchronize()  # Ensure all CUDA operations finish
end = time.time()
print(f"Regular Attention Time: {end - start:.4f}s")

# Timing sparse attention
start = time.time()
for _ in range(10):
    output = sparse_attention_benchmark(Q, K, V, sparse_mask)
torch.cuda.synchronize()
end = time.time()
print(f"Sparse Attention Time: {end - start:.4f}s")
