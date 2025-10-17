# MiniTorch CUDA Operations

A high-performance CUDA implementation of tensor operations for the MiniTorch framework. This project provides GPU-accelerated implementations of fundamental tensor operations (map, zip, reduce, and matrix multiplication) using CUDA C++ kernels integrated with Python.

## Features

- **Map Operations**: Element-wise unary operations on tensors (e.g., sigmoid, ReLU, negation, logarithm)
- **Zip Operations**: Element-wise binary operations on tensor pairs (e.g., addition, multiplication, comparison)
- **Reduce Operations**: Aggregation along tensor dimensions (e.g., sum, product, max)
- **Matrix Multiplication**: Optimized GPU-accelerated matrix multiplication
- **Automatic Differentiation**: Integration with MiniTorch's autodiff framework
- **Flexible Broadcasting**: Support for broadcasting operations across different tensor shapes

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (NVIDIA)
- CUDA Toolkit (12.0+)
- NVCC compiler

**Note**: For GPU access, you can use:
- Google Colab (free, T4 GPU)
- AWS with GPU instances
- Local machine with NVIDIA GPU

## Installation

### 1. Set up a virtual environment (recommended)

Using venv:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Or using conda:
```bash
conda create -n minitorch-cuda python=3.9
conda activate minitorch-cuda
```

### 2. Clone and install the package

```bash
git clone https://github.com/ppaleja/llmsys_f25_hw1.git
cd llmsys_f25_hw1
python -m pip install -r requirements.txt
python -m pip install -r requirements.extra.txt
python -m pip install -Ue .
```

### 3. Verify installation

```bash
python -c "import minitorch; print('Success: minitorch is installed correctly');"
```

### 4. Compile CUDA kernels

Create the directory for compiled kernels and compile:
```bash
mkdir -p minitorch/cuda_kernels
nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
```

**Note**: If using a system like PSC, you may need to load the CUDA module first:
```bash
module load cuda/12.4.0
```

## Project Structure

```
minitorch/                  # Core MiniTorch framework
    cuda_kernel_ops.py      # Python-CUDA integration layer
    tensor.py               # Tensor data structure
    tensor_ops.py           # Tensor operation interfaces
    autodiff.py             # Automatic differentiation
    operators.py            # Mathematical operators
src/
    combine.cu              # CUDA kernel implementations
tests/                      # Test suite
    test_tensor_general.py  # Tensor operation tests
```

## Usage

### Setup Backend

First, create a CUDA backend for tensor operations:
```python
import minitorch
from minitorch import Tensor, TensorBackend
from minitorch.cuda_kernel_ops import CudaKernelOps

# Create CUDA backend
cuda_backend = TensorBackend(CudaKernelOps)
```

### Map Operations

Map operations apply a unary function element-wise to a tensor. The operation produces a new tensor with the same shape as the input.

**Example**: Applying ReLU activation
```python
# Create a tensor
x = minitorch.tensor([1.0, -2.0, 3.0, -4.0], backend=cuda_backend)

# Apply ReLU (max(0, x))
y = x.relu()  # Result: [1.0, 0.0, 3.0, 0.0]
```

**Supported unary operations**:
- `relu()` - ReLU activation
- `sigmoid()` - Sigmoid activation  
- `log()` - Natural logarithm
- `exp()` - Exponential
- `neg()` - Negation
- And more...

### Zip Operations

Zip operations apply a binary function to corresponding elements from two input tensors. The tensors must have the same shape or be broadcastable.

**Example**: Element-wise addition
```python
# Create two tensors
a = minitorch.tensor([1.0, 2.0, 3.0], backend=cuda_backend)
b = minitorch.tensor([4.0, 5.0, 6.0], backend=cuda_backend)

# Element-wise addition
c = a + b  # Result: [5.0, 7.0, 9.0]
```

**Supported binary operations**:
- `+` - Addition
- `*` - Multiplication
- `<` - Less than comparison
- `==` - Equality comparison
- `max()` - Element-wise maximum
- And more...

### Reduce Operations

Reduce operations aggregate elements along a specified dimension using a binary function, producing a tensor with reduced dimensionality.

**Example**: Sum along dimension
```python
# Create a 2D tensor
x = minitorch.tensor([[1.0, 2.0, 3.0], 
                      [4.0, 5.0, 6.0]], backend=cuda_backend)

# Sum along dimension 1 (columns)
y = x.sum(1)  # Result: [6.0, 15.0]

# Sum along dimension 0 (rows)
z = x.sum(0)  # Result: [5.0, 7.0, 9.0]
```

**Supported reduction operations**:
- `sum()` - Sum reduction
- `mean()` - Mean/average
- `max()` - Maximum value
- Custom reductions with arbitrary binary functions

The implementation uses efficient parallel reduction with GPU thread blocks.

### Matrix Multiplication

High-performance GPU-accelerated matrix multiplication, one of the most critical operations in deep learning.

**Example**: Matrix multiplication
```python
# Create two matrices
A = minitorch.tensor([[1.0, 2.0], 
                      [3.0, 4.0]], backend=cuda_backend)
B = minitorch.tensor([[5.0, 6.0], 
                      [7.0, 8.0]], backend=cuda_backend)

# Matrix multiplication
C = A @ B  # Result: [[19.0, 22.0], [43.0, 50.0]]
```

The matrix multiplication kernel uses optimized GPU parallelization strategies:
- Simple parallelization: Each thread computes one output element
- Optional shared memory tiling for improved memory bandwidth utilization

## Testing

The project includes a comprehensive test suite using pytest and Hypothesis for property-based testing.

### Run all tests
```bash
python -m pytest -l -v
```

### Run specific test categories
```bash
# Test map operations (unary functions)
python -m pytest -l -v -k "cuda_one_args"

# Test zip operations (binary functions)
python -m pytest -l -v -k "cuda_two_args"

# Test reduce operations
python -m pytest -l -v -k "cuda_reduce"

# Test matrix multiplication
python -m pytest -l -v -k "cuda_matmul"

# Run all CUDA tests
python -m pytest -l -v -k "cuda"
```


## Development

### Recompiling CUDA Kernels

After making changes to `src/combine.cu`, recompile the kernels:
```bash
nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
```

### Debugging Tips

1. **Memory Access Errors**: Check thread bounds, stride calculations, and use `__syncthreads()` appropriately for shared memory
2. **CUDA Kernel Debugging**: Use `printf` statements in kernels (affects performance), start with simple test cases
3. **Google Colab**: Restart the runtime if encountering persistent issues after recompiling

## Technical Details

### Supported Operations

The CUDA kernels support the following functions (mapped via function IDs):
- Arithmetic: `add`, `mul`, `neg`, `inv`
- Comparison: `lt`, `eq`, `is_close`, `max`
- Activation: `sigmoid`, `relu`, `tanh`
- Math: `log`, `exp`, `pow`
- Derivatives: `relu_back`, `log_back`, `inv_back`

### Implementation Notes

- **Thread Configuration**: Uses 32 threads per block by default (configurable via `THREADS_PER_BLOCK`)
- **Memory Layout**: Stride-based indexing for flexible multidimensional tensor representation
  - For a 2D tensor: `A[i,j] = Memory[i * strides[0] + j * strides[1]]`
  - Example: Shape (2, 4), Strides (4, 1) → A[1,2] = Memory[1 × 4 + 2 × 1] = Memory[6]
- **Broadcasting**: Automatic shape broadcasting for compatible operations
- **Error Handling**: Bounds checking in kernels to prevent memory access violations

## Demo

See `Project_Demo.ipynb` for a comprehensive demonstration of all features, including:
- Environment setup
- Basic tensor operations
- Map, zip, and reduce examples
- Matrix multiplication examples
- Integration with MiniTorch's autodiff

## Acknowledgments

This project is based on the MiniTorch educational framework and implements CUDA acceleration for tensor operations. The implementation follows parallel computing best practices from "Programming Massively Parallel Processors" (4th Edition).

Additional resources:
- [CUDA Reduction Techniques](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [MiniTorch Framework](https://minitorch.github.io/)

## License

See LICENSE file for details.
