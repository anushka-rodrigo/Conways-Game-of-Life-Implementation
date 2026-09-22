# Conway's Game of Life — CPU vs CUDA

## Overview

This project implements [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) — a cellular automaton where every cell in a grid is alive or dead and evolves generation by generation under a fixed neighbour-count rule — in two ways, and benchmarks them against each other:

- A **sequential CPU implementation** in C++, using a flat, cache-friendly 2D array.
- A **parallel CUDA implementation**, mapping each grid cell to an independent GPU thread.

Both versions run the identical 1024×1024 simulation, for 100 generations, from the identical seeded random starting grid, using the identical standard B3/S23 rule set — so the only variable is *how* the work is executed, not *what* is computed. The goal is to quantify, in a statistically reliable way, how much faster the problem can be solved by mapping cells to independent GPU threads, and to explain the specific HPC techniques responsible.

**Headline result:** the CUDA implementation is **~440× faster** than the sequential CPU baseline (mean 6.41 ms vs. 2819.36 ms for a full 100-generation run), while also being far more consistent run-to-run.

Full methodology, code walkthroughs, and analysis are in [`report/report.pdf`](report/report.pdf).

## Repository structure

```
.
├── 2D_array_immplementation/   # Sequential CPU implementation
│   └── Conways_Game_of_Life_cpp_implementation.ipynb
├── CUDA_implementation/        # Parallel CUDA GPU implementation
│   └── Conways_Game_of_Life_CUDA_implementation.ipynb
├── presentation/                # Slide deck + speaker script for the project presentation
│   ├── presentation.pptx
└── report/                      # Full written report
    └── CCS3032-HPC_Assignment-FC221019.pdf
```

| Folder | Contents |
|---|---|
| `2D_array_immplementation/` | Jupyter notebook (`.ipynb`) that writes, compiles (`g++ -O2 -std=c++17`), and runs the sequential CPU version (`game_of_life.cpp`) on Google Colab, plus frame export and correctness checks. |
| `CUDA_implementation/` | Jupyter notebook that writes, compiles (`nvcc -O3`), and runs the CUDA GPU version (`game_of_life_cuda.cu`) on Google Colab, plus frame export and correctness checks. |
| `presentation/` | The presentation deck and a matching slide-by-slide speaker script used to present this project. |
| `report/` | The full project report: problem definition, experimental setup, implementation details, benchmarking methodology, results, and conclusion. |

## How it works

### The rule (standard Conway B3/S23)
- A live cell with **fewer than 2** or **more than 3** live neighbours **dies** (under/overpopulation).
- A live cell with **2 or 3** live neighbours **survives**.
- A dead cell with **exactly 3** live neighbours **becomes alive** (reproduction).

### CPU implementation (`2D_array_immplementation/`)
- Grid stored as a flat `std::vector<int>` (`idx(x, y) = y*WIDTH + x`) instead of nested arrays — more cache-friendly for large grids.
- `countAliveNeighbors()` scans the 8-cell Moore neighbourhood with boundary checks at grid edges.
- Double-buffered generations (`current` / `next`), swapped in O(1) each step — avoids read-after-write hazards within a generation.
- Compiled with `g++ -O2 -std=c++17` for auto-vectorization and loop unrolling — the only optimization available on a single core.

### CUDA implementation (`CUDA_implementation/`)
- One GPU thread per cell — the entire ~1.05M-cell grid is evaluated in parallel every generation via a single kernel launch.
- 16×16 thread blocks (256 threads/block), a clean multiple of the 32-thread warp size.
- The grid never leaves device memory during the timed loop — `d_current` / `d_next` are swapped by pointer after every kernel launch, eliminating host↔device transfer from the timed path.
- Every CUDA API call wrapped in a `CUDA_CHECK` macro so errors never fail silently.
- Compiled with `nvcc -O3`.

### Benchmarking methodology
1. One **untimed warm-up pass** (primes memory allocation, CPU caches, and the GPU context/driver).
2. **Five timed repeats**, each resetting the grid to the identical initial state, measured with `std::chrono::high_resolution_clock` (GPU timing bracketed by `cudaDeviceSynchronize()`).
3. **Mean and range (min–max)** reported across the five repeats — not a single run.
4. Frame-saving for GIF visualization is handled as a **separate, untimed pass** (`saveFrames = true`), kept out of the timed runs so disk I/O never distorts the comparison.

### Correctness verification
Alive-cell counts extracted from saved frames at matching generations were compared between the CPU and GPU runs and found identical — confirming both implementations compute the same simulation, not just a faster one.

## Environment

Both notebooks were developed and run on **Google Colab**:

| | CPU | GPU |
|---|---|---|
| Hardware | Intel(R) Xeon(R) @ 2.20GHz (1 socket, 1 core, 2 threads/core) | NVIDIA Tesla T4 (compute capability 7.5, 15,360 MiB) |
| Compiler | `g++ -O2 -std=c++17` | `nvcc -O3` |

## Running the notebooks

1. Open `2D_array_immplementation/Conways_Game_of_Life_cpp_implementation.ipynb` and `CUDA_implementation/Conways_Game_of_Life_CUDA_implementation.ipynb` in Google Colab.
2. For the CUDA notebook, ensure the runtime type is set to a **GPU** (Runtime → Change runtime type → T4 GPU, or any available CUDA-capable GPU).
3. Run all cells top to bottom in each notebook. Each notebook:
   - Writes the C++/CUDA source via `%%writefile`.
   - Compiles it with the flags above.
   - Runs the simulation (warm-up + 5 timed repeats) and prints timing results.
   - Optionally saves `.pgm` frames and an animated `.gif` of the simulation (set `saveFrames = true` in the source to enable — this is off by default so it doesn't affect timing).

## Results summary

| Metric | CPU (2D array) | CUDA (GPU) |
|---|---|---|
| Mean total time (100 generations) | 2819.36 ms | 6.41154 ms |
| Range (min–max) | 2404.47 – 3389.84 ms | 6.40737 – 6.41658 ms |
| Mean time / generation | 28.19 ms | 0.0641 ms |
| **Speedup** | | **≈ 440×** |

The CPU showed noticeable run-to-run variance (~41% spread), most plausibly from contention on Colab's shared, non-dedicated CPU infrastructure rather than the algorithm itself. The GPU was highly consistent (within a 0.01 ms band) once warmed up, reflecting isolated execution on dedicated streaming multiprocessors independent of host OS scheduling.

See [`report/report.pdf`](report/report.pdf) for the full analysis, complete source listings, and discussion.

## Key techniques behind the speedup

1. **Thread-per-cell parallelism** — ~1.05M independent GPU threads replace ~1.05M sequential loop iterations.
2. **Warp-aligned block sizing** — 16×16 = 256 threads/block, a clean multiple of the 32-thread warp.
3. **Zero host↔device transfer in the timed loop** — buffers swapped by pointer entirely on-device between generations.

## License

Academic coursework submitted for CCS3032 — High Performance Computing, University of Sri Jayewardenepura. Provided as-is for educational reference.