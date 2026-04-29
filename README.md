# Onnx4Deeploy

[![CI](https://github.com/pulp-platform/Onnx4Deeploy/workflows/CI/badge.svg)](https://github.com/pulp-platform/Onnx4Deeploy/actions)
[![Tests](https://img.shields.io/badge/tests-91%20passed-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](.github/CONTRIBUTING.md)

**A comprehensive framework for ONNX model generation, optimization, and deployment for Deeploy.**

Onnx4Deeploy provides a unified interface for exporting PyTorch models to ONNX format with specialized optimizations for training and inference on Deeploy hardware accelerators.

---

## ✨ Features

### 🎯 Core Capabilities
- **Unified Model Export**: Single API for inference, full training, and single-step training-as-inference debug mode
- **27 Operator Tests**: Comprehensive test coverage for all supported ONNX operators
- **3 Pre-built Models**: CCT, EpiDeNet, and MI-BMInet ready to use
- **Training Graph Optimization**: Specialized optimizations for on-device training
- **Type-safe API**: Full type annotations and documentation

### 🔧 Optimization Suite
- GEMM conversion and fusion
- Gradient node optimization
- Graph cleaning and simplification
- Shape operation optimization
- Node naming and annotation utilities

### 🧪 Testing Framework
- Pytest-based test suite
- ONNX Runtime validation
- Baseline comparison testing

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.0+
- ONNX 1.14+
- ONNX Runtime 1.19+

### Install from source

```bash
git clone https://github.com/pulp-platform/Onnx4Deeploy.git
cd Onnx4Deeploy
pip install -e .
```

### Verify installation

```bash
python -c "import onnx4deeploy; print(onnx4deeploy.__version__)"
```

---

## 🚀 Quick Start

Onnx4Deeploy provides two main features: **Operator-level** generation and **Model-level** export.

### 🎯 Command Line Tool (Recommended)

Use the unified CLI tool `Onnx4Deeploy.py`:

```bash
# Generate operator tests
python Onnx4Deeploy.py -operator Relu -o ./onnx

# Generate model inference graph
python Onnx4Deeploy.py -model CCT -mode infer -o ./onnx

# Generate model training graph
python Onnx4Deeploy.py -model CCT -mode train -o ./onnx

# Generate single-step training-as-inference test fixture
# (per-tensor gradient verification — see "Single-step debug mode" below)
python Onnx4Deeploy.py -model CCT -mode train_single_step -o ./onnx

# List available options
python Onnx4Deeploy.py --list-models
python Onnx4Deeploy.py --list-operators
python Onnx4Deeploy.py --examples
```

**Available Arguments:**
- `-operator NAME`: Generate operator test (e.g., Relu, Add, Gemm)
- `-model NAME`: Generate model ONNX (e.g., CCT, EpiDeNet, MIBMInet)
- `-mode {infer,train,train_single_step}`: Model export mode (default: infer). `train_single_step` produces an inference-runner-compatible single-step training fixture — see *Single-step debug mode* below.
- `-o PATH`: Output directory path
- `--list-models`: List all available models
- `--list-operators`: List all available operators
- `--examples`: Show usage examples

---

## 🔍 Single-step debug mode (`train_single_step`)

Standard `train` mode runs **N optimizer steps** and only compares the final loss
or weight values against a reference. When a model diverges (e.g. MobileNetV1
step-2 loss off by 1.7 %), the symptom is a single scalar — you cannot tell
which gradient is wrong.

`train_single_step` rewires the same training graph so the **inference**
runner (`deeployRunner_*.py`) can drive it for **per-tensor** gradient
verification:

| | `train` | `train_single_step` |
|---|---|---|
| Optimizer steps | N (default 4) | 1 (forward + backward only) |
| `lazy_reset_grad` | runtime input | **constant initializer = `True`** (each `InPlaceAccumulator` output = pure batch dW, no historical accum) |
| `inputs.npz` | `arr_0000…` + per-batch data + meta | **single named tensor** (the data input) — labels, params, and grad-accumulation buffers are baked into the deployed `network.onnx` as initializers |
| `outputs.npz` | SGD-updated params + per-step losses | **`loss` + raw `<param>_grad.accumulation.out` per parameter** (PyTorch-autograd reference) |
| Driver | `deeployTrainingRunner_*.py` | `deeployRunner_*.py` (untiled) or `deeployRunner_tiled_*.py` |
| Failure tells you | "step k loss off by X" | "Output K (= `<layer>_grad.accumulation.out`) diff = X at index Y" |

### Generate

```bash
# Direct from PyTorch model — needs onnxruntime-training installed
python Onnx4Deeploy.py -model SimpleMLP -mode train_single_step -o ./onnx/simplemlp_single

# Post-process an existing `train` artifact dir (also works on vendored
# Deeploy fixtures that ship only network.onnx + inputs.npz + outputs.npz —
# falls back to PyTorch fresh weights when network_infer.onnx is absent)
python scripts/make_single_step.py --model MobileNetV1 \
    /path/to/mobilenetv1_train  /path/to/mobilenetv1_single_step
```

### Run via the inference runner

```bash
cd $DEEPLOY/DeeployTest
# untiled
python deeployRunner_siracusa.py -t /path/to/<model>_single_step --cores=8 -vv
# tiled (use the same --l1 / --defaultMemLevel as the original train test)
python deeployRunner_tiled_siracusa.py -t /path/to/<model>_single_step \
    --cores=8 --l1 128000 --defaultMemLevel L3 -vv
```

The runner prints `Errors: K out of N` plus per-element `Expected / Actual /
Diff at Index … in Output …` lines, letting you bisect which Conv/BN backward
gradient diverges in the integrated execution.

### Required Deeploy companion change

Deeploy's stock `PULPInPlaceAccumulatorV2TilingReadyBindings` uses the **tiled**
template, which writes only `accum_buffer` and skips `data_out` (so the graph
output that the inference runner reads gets garbage). Switch the binding to
the non-tiled template for `train_single_step` to work:

```python
# Deeploy/Targets/PULPOpen/Tiler.py:201
PULPInPlaceAccumulatorV2TilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = PULPInPlaceAccumulatorV2Bindings,  # was: PULPInPlaceAccumulatorV2TiledBindings
    tileConstraint = InPlaceAccumulatorV2TileConstraint())
```

The non-tiled template additionally writes `data_out` (an extra in-cluster
copy, no DMA egress) and is regression-clean against the standard tiled
training tests.

### What single-step does *not* catch

A single forward+backward exercises step-0 grads only. Bugs that need optimizer
state, BN running statistics, or multi-step gradient accumulation history
(e.g. drift introduced after gamma is updated, or `mm_add` race conditions
that emerge only after several schedule rounds) will not surface here. Use
`train_single_step` to confirm per-layer kernel correctness in isolation;
fall back to multi-step `train` mode for end-to-end validation.

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](.github/CONTRIBUTING.md) for details.

### Quick Development Setup

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black .
isort .
```

---

## 📄 License
All licenses used in this repository are listed under the `LICENSES` folder. Unless specified otherwise in the respective file headers, all code checked into this repository is made available under a permissive license.
- Most software sources and tool scripts are licensed under the [MIT license](https://opensource.org/licenses/mit).
- Markdown, JSON, text files, pictures, PDFs, are licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0) license (CC BY 4.0).

To extract license information for all files, you can use the [reuse tool](https://reuse.software/) and by running `reuse spdx` in the root directory of this repository.


---

## 🙏 Acknowledgments

- Built with [ONNX](https://onnx.ai/)
- Tested with [ONNX Runtime](https://onnxruntime.ai/)
- Optimized for [Deeploy](https://deeploy.ml/) hardware

---

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/pulp-platform/Onnx4Deeploy/issues)
- **Documentation**: [docs/](docs/)
- **Progress**: [REFACTORING_STATUS.md](REFACTORING_STATUS.md)
