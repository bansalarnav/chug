#!/usr/bin/env python3
"""Check the FlashInfer BF16 attention build before loading a large model."""
import torch
import flashinfer

workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device='cuda')
wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, 'NHD', backend='fa2')
indptr = torch.tensor([0, 4], dtype=torch.int32, device='cuda')
wrapper.plan(indptr, indptr, num_qo_heads=16, num_kv_heads=2,
    head_dim_qk=128, causal=True, q_data_type=torch.bfloat16,
    kv_data_type=torch.bfloat16)
q = torch.randn(4, 16, 128, dtype=torch.bfloat16, device='cuda')
k = torch.randn(4, 2, 128, dtype=torch.bfloat16, device='cuda')
v = torch.randn_like(k)
output = wrapper.run(q, k, v)
torch.cuda.synchronize()
assert output.shape == q.shape
assert torch.isfinite(output).all().item()
print('FlashInfer BF16 ragged prefill compiled, loaded and executed successfully.', flush=True)
