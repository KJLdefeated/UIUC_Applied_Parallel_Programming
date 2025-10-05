# GPU Kernel Optimization

**Goal:** Push GPU convolution performance well beyond the milestone requirement by **stacking fusion, Tensor-Core GEMM, memory-access optimizations, and mixed-precision**.

---

## Key Improvement

* **Total op-time dropped from ~200 ms → ~65 ms** for batch = 10 000 (≈ 3× speed-up vs baseline).
* Delivered **one fused kernel** with **all major optimizations** integrated and validated using Nsight-Systems / Nsight-Compute.

---

## My Contributions

### 1. Kernel Fusion (Backbone)

Merged **unroll → GEMM → permute** into a single CUDA kernel.

* Removed intermediate buffers and redundant global memory traffic.
* Gave the **biggest single jump**, cutting kernel-side op-time by >50% vs baseline.

### 2. Tensor-Core GEMM

Adopted **CUDA WMMA API** for the fused kernel’s GEMM stage.

* FP16 inputs + FP32 accumulation; warp-level tiling + shared-memory staging.
* Boosted compute throughput while keeping accuracy.

### 3. Memory-Access and Control-Flow Optimizations

* **`__restrict__` pointers:** guided compiler for better alias-free optimizations → ≈ 2–3 ms gain on GEMM part.
* **Loop Unrolling:** manually unrolled inner accumulation loops (for non-WMMA fallback path) → ≈ 4–5 ms gain by trimming loop-control overhead.
* **Pinned-memory + dual-stream overlap:** hid a good share of host-device copy latency.

### 4. Mixed-Precision FP16

* Converted intermediate matrices (unrolled buffer & GEMM output) to FP16 **inside the fused pipeline** (single conversion step).
* Reduced memory bandwidth by ≈ 2× for those stages; overcame previous float↔half conversion overhead.
* Yielded an extra ≈ 3 ms net gain vs all-float fused baseline.

### 5. Profiling-Driven Tuning

* Iterated with Nsight-Compute to balance register pressure, occupancy, and tile sizes.
* Identified final bottlenecks as L2-bandwidth-limited rather than compute-bound.

---

## Final Performance (Batch = 10 000)

| Variant                          | Op-Time 1   | Op-Time 2   | Sum of Op-Times |
| -------------------------------- | ----------- | ----------- | --------------- |
| Baseline (matrix-unroll)         | 95.6 ms     | 72.9 ms     | ≈ 200 ms        |
| + Fusion + Tensor-Cores          | 42.3 ms     | 32.2 ms     | **74.5 ms**     |
| + `__restrict__` + Unroll + FP16 | **36.8 ms** | **28.5 ms** | ** 65.3 ms**   |

---

## Project Layout

```
project/
 ├─ src/layer/custom/m3-forward.cu      # Final all-in-one fused implementation
 └─ m3/
     ├─ req_0/    # Streams
     ├─ req_1/    # Tensor Cores
     ├─ req_2/    # Fusion
     ├─ op_1/     # __restrict__
     ├─ op_2/     # Loop Unrolling
     └─ op_5/     # FP16 experiment
```
