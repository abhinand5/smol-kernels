# Unit 03: Matrix Transpose

> Zero FLOPs, and still one of the most instructive kernels you will write. Transpose is where coalescing stops being a footnote and becomes the entire performance story — and where you meet shared memory and the bank conflict.

Transpose does no arithmetic:

```text
B[j, i] = A[i, j]
```

Yet a naive version runs at a fraction of memcpy bandwidth, and fixing it requires **shared memory tiling**, an understanding of **memory coalescing** in both directions, and the **bank conflict** — the first time the *internal* structure of on-chip memory bites you.

You implement, twice:

1. CUDA C++ transpose (naive, then tiled + padded shared memory)
2. Triton transpose (tiled pointer arithmetic)

Baseline: `x.t().contiguous()` / `torch.transpose` materialized.

Numbers from the [hardware spec sheet](reference-hardware-spec-sheet.md).

---

## 3.1 Why Transpose Is Hard When It Does Nothing

Intensity is `0 FLOPs / 8 bytes = 0`. Pure memory movement — it should hit memcpy bandwidth. The problem: **you cannot read and write both coalesced with the naive mapping.**

A warp accesses 32 consecutive threads. Coalescing wants those 32 threads to touch 32 *consecutive* addresses (one cache line). For a transpose:

```text
read  A[i, j]  with j = threadIdx.x  -> consecutive in memory  ✅ coalesced
write B[j, i]  with j = threadIdx.x  -> stride = N rows apart   ❌ scattered
```

The read is coalesced; the write is a strided scatter (each thread writes to a different cache line). The scattered side throttles the whole kernel — often to **10–20% of memcpy bandwidth**. Swapping which side is coalesced just moves the problem. You cannot win with a direct global-to-global mapping.

---

## 3.2 The Fix: Stage Through Shared Memory

Shared memory is on-chip, per-block, and — crucially — **fast for non-contiguous access** (no cache-line penalty like global memory). The tiled algorithm:

```text
1. A block owns a TILE×TILE tile of A.
2. Read the tile from global memory COALESCED into a __shared__ buffer.
3. __syncthreads()  — wait until the whole tile is loaded.
4. Write the tile to B, reading from shared memory with swapped indices,
   so the GLOBAL write is also COALESCED.
```

The transpose happens *inside* shared memory (cheap), so **both** global accesses — the read and the write — stay coalesced. This is the canonical use of shared memory: **rearrange data on-chip to make global traffic coalesced.**

```text
__shared__ float tile[TILE][TILE];

// coalesced global read -> shared
tile[threadIdx.y][threadIdx.x] = A[ (blockY+threadIdx.y) * N + (blockX+threadIdx.x) ];
__syncthreads();

// coalesced global write <- shared (note swapped block coords AND swapped tile indices)
B[ (blockX+threadIdx.y) * M + (blockY+threadIdx.x) ] = tile[threadIdx.x][threadIdx.y];
```

(Index details are yours to derive — that is the exercise.)

---

## 3.3 The Bank Conflict

Shared memory is split into **32 banks**, one per warp lane. A warp can read 32 words in one transaction **only if they fall in 32 different banks**. Bank = `address mod 32` (in 4-byte words).

Now look at `tile[threadIdx.x][threadIdx.y]` in the write step. Threads in a warp vary `threadIdx.x` while reading down a *column* of the tile:

```text
tile[0][k], tile[1][k], tile[2][k], ...   addresses differ by TILE (e.g. 32) words
```

If `TILE == 32`, every one of those addresses is in **the same bank** → a **32-way bank conflict** → the access serializes into 32 transactions. Your beautiful tiled kernel crawls.

**The fix is one character of padding:**

```text
__shared__ float tile[TILE][TILE + 1];   // pad the inner dimension by 1
```

Padding by 1 shifts each row by one word, so a column walk now hits 32 *different* banks. The conflict vanishes. This single `+ 1` is one of the most famous lines in CUDA.

> The lesson generalizes: shared memory is fast *only* if your access pattern spreads across banks. Column-major access into a power-of-two-width tile is the classic trap; padding is the classic fix.

---

## 3.4 Triton's Version

Triton does not expose `__shared__` or banks directly — the compiler manages on-chip storage. But the *shape* of your tiling still decides coalescing. You work with 2D tiles:

```python
# row offsets and column offsets, then a 2D block of pointers
row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
a_ptrs = a_ptr + row[:, None] * stride_am + col[None, :] * stride_an
# load the tile, store it to B with swapped strides
```

The transpose is expressed by **swapping the strides** when you store. `tl.load` / `tl.store` of a 2D block, plus masks for ragged edges, plus `tl.make_block_ptr` if you want the structured API. The compiler handles the shared-memory staging — but tile shape (`BLOCK_M × BLOCK_N`) still determines whether the generated loads/stores coalesce, so you still benchmark shapes.

---

## 3.5 CUDA Exercise Spec

Write three kernels and benchmark all three:

1. **Naive** — direct global-to-global transpose. Establish the slow baseline.
2. **Tiled** — shared-memory staging, *without* padding. Observe the bank-conflict slowdown.
3. **Tiled + padded** — `[TILE][TILE+1]`. Observe the fix.

Each must:

- use a 2D grid and 2D blocks
- handle non-square and non-tile-divisible matrices (ragged edges) with bounds checks
- `__syncthreads()` between the shared load and the shared store

Tile sizes to test: `16×16`, `32×32`. Matrix sizes: at least `4096×4096` and one non-square, non-divisible case (e.g. `4097×2049`).

> Reading the three numbers in sequence — naive → tiled-unpadded → tiled-padded — *is* the deliverable. You should be able to explain each jump.

---

## 3.6 Triton Exercise Spec

Write a Triton transpose that:

- accepts `a_ptr`, `b_ptr`, the dims `M`, `N`, the strides, and `BLOCK_M`, `BLOCK_N` constexprs
- builds 2D masked offsets
- loads an `A` tile and stores it transposed into `B`
- handles ragged edges

Benchmark tile shapes (e.g. `32×32`, `64×64`, `32×64`) and compare to the CUDA padded version and the PyTorch baseline.

---

## 3.7 Benchmark Requirements

```text
Matrices: 4096 x 4096 float32  (64 MB each, 128 MB traffic)
plus one non-square non-divisible case
```

Report per implementation:

| Field | Meaning |
|---|---|
| implementation | CUDA naive / tiled / tiled+pad / Triton / PyTorch |
| tile size | e.g. 32×32 |
| median time | µs |
| effective bandwidth | `2 * M * N * 4 / s / 1e9` (read + write) |
| efficiency | vs 264 GB/s and vs your Unit 0 memcpy |
| correctness | exact match vs `A.t()` |

---

## 3.8 What Good Looks Like

- **Naive:** badly memory-throttled — often 10–25% of memcpy bandwidth. This is the point.
- **Tiled, unpadded:** better, but bank conflicts cap it well below copy bandwidth.
- **Tiled + padded:** approaches memcpy bandwidth (this is a pure copy with a relabeling, so copy bandwidth *is* the ceiling).
- You can articulate why each step helped — coalescing for the first jump, bank conflicts for the second.

---

## 3.9 Common Mistakes

### Forgetting `__syncthreads()`

Reading the shared tile before every thread has finished writing it = garbage. Both a correctness and a classic race-condition bug.

### Padding the wrong dimension (or not at all)

Pad the **inner** dimension (`[TILE][TILE+1]`). Padding the outer one does nothing for bank conflicts.

### Assuming square / divisible matrices

Real transposes hit ragged edges. Bounds-check both dimensions independently.

### Counting bytes wrong

Transpose moves `2 * M * N * 4` bytes (read once, write once). Same as a copy.

### Believing Triton has no coalescing concerns

The compiler stages shared memory for you, but your tile shape still dictates global-access patterns. Bad shapes still run slow.

---

## 3.10 Files

```text
src/unit03/
  transpose_triton.py   # student: tiled Triton transpose
  transpose_cuda.cu     # student: naive + tiled + tiled-padded
  benchmark_transpose.py
```

---

## 3.11 Check Your Understanding

1. Transpose does zero FLOPs — why is it not trivially memcpy-fast?
2. In a naive transpose, which access (read or write) is uncoalesced, and why?
3. How does staging through shared memory make *both* global accesses coalesced?
4. What is a shared-memory bank, and how is a word's bank computed?
5. Why does a `[32][32]` shared tile cause a 32-way bank conflict on column access?
6. Why does `[32][33]` (pad by 1) fix it?
7. Why is `__syncthreads()` required between the shared load and store?
8. What is the theoretical bandwidth ceiling for a perfect transpose, and why?
9. In Triton, what role does tile shape play even though you never name banks?
10. How would you transpose a non-tile-divisible matrix correctly?

---

**Next:** Unit 04 — Softmax: parallel reductions, numerical stability, and the online trick that becomes FlashAttention
