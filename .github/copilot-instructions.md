# GitHub Copilot Instructions

## Overview

This repository contains Assignment 1 for CUDA Programming with the MiniTorch framework. The goal is to implement high-performance CUDA kernels for tensor operations (`map`, `zip`, `reduce`) in `src/combine.cu` and integrate them into Python via `minitorch/cuda_kernel_ops.py`.

## Core Components

- `minitorch/`: Core Python framework (autodiff, operators, tensor, tensor_ops).
- `src/combine.cu`: CUDA kernels (`mapKernel`, `zipKernel`, `reduceKernel`, plus matrix multiply support).
- `minitorch/cuda_kernel_ops.py`: Python–CUDA bridge using `ctypes`/`PyCUDA` (see `fn_map` & `lib.tensorMap`).
- `tests/`: Hypothesis-based test suite for unary, binary, reduce operations and matrix multiplication.

## Development Workflow

1. Environment setup:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt -r requirements.extra.txt
   pip install -Ue .
   ```
2. Compile CUDA kernels:
   ```bash
   ./compile_cuda.sh
   # or:
   nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
   ```
3. Run style checks:
   ```bash
   ./style.sh
   ```
4. Execute tests:
   ```bash
   pytest -l -v
   ```

## Patterns & Conventions

- CUDA assignment regions are marked in `src/combine.cu` with `BEGIN ASSIGN2_*` / `END ASSIGN2_*`.
- Function IDs mapped in `fn_map` (e.g., `operators.add: 1`) before calling kernels.
- Use `shape_broadcast` to align tensor shapes in `tensorZip` and similar ops.
- All tensor backends use contiguous C arrays (`.zeros()`, `.contiguous()`).
- Tests employ Hypothesis strategies (`data.draw(tensors(...))`, `small_floats`, etc.) to generate random tensors.
- No cuda code can can be run locally. All tests are done through Google Colab, i.e. not in the current environment.

## Integration Points

- Shared object loaded in Python:
  ```python
  lib = ctypes.CDLL("minitorch/cuda_kernels/combine.so")
  ```
- Define `argtypes` and `restype` on `lib.tensorMap`, `lib.tensorZip`, `lib.tensorReduce` before invocation.
- Kernel invocation pattern:
  ```python
  lib.tensorMap(
      out._tensor._storage,
      out._tensor._shape.astype(np.int32),
      out._tensor._strides.astype(np.int32),
      out.size,
      a._tensor._storage,
      a._tensor._shape.astype(np.int32),
      a._tensor._strides.astype(np.int32),
      a.size,
      len(a.shape),
      fn_id
  )
  ```

## External Dependencies

- PyCUDA for GPU interfacing (`pycuda.driver`, `pycuda.compiler`, `pycuda.gpuarray`).
- Numba (`numba.cuda.is_available()`) used to detect GPU backend in tests.
- Hypothesis & PyTest for testing.

## Tips & References

- Inspect `hw1/reduce.jpg` and `hw1/simple_parallel.png` for data flow illustrations.
- Validate strides computation using `to_index`/`index_to_position` helpers in `tensor_data.py`.
- For debugging, adjust `THREADS_PER_BLOCK` in `minitorch/cuda_kernel_ops.py`.

---

_Is any section unclear or missing details? Let me know to iterate further._