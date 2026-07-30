#!/usr/bin/env bash

# Install the Ascend NPU environment used by the reasoning and agent examples.
#
# Run this script inside an activated Python virtual environment on an Ascend
# host with a compatible CANN installation. It installs CPU PyTorch plus
# torch_npu, then builds the NPU-enabled SGLang stack from source.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SGLANG_VERSION="${SGLANG_VERSION:-0.5.2}"
SGLANG_DIR="${SGLANG_DIR:-$PWD/sglang}"
SGL_KERNEL_NPU_DIR="${SGL_KERNEL_NPU_DIR:-$PWD/sgl-kernel-npu}"
SGL_KERNEL_NPU_REF="${SGL_KERNEL_NPU_REF:-20251214}"
TORCH_WHEEL_URL="${TORCH_WHEEL_URL:-https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl}"
TORCH_NPU_WHEEL_URL="${TORCH_NPU_WHEEL_URL:-https://gitcode.com/Ascend/pytorch/releases/download/v7.2.0-pytorch2.6.0/torch_npu-2.6.0.post3-cp311-cp311-manylinux_2_28_aarch64.whl}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
TEST_PYPI_INDEX_URL="${TEST_PYPI_INDEX_URL:-https://test.pypi.org/simple}"

command -v git >/dev/null || { echo "git is required." >&2; exit 1; }
"${PYTHON_BIN}" -m pip --version >/dev/null

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install \
    "numpy>=1.19.2,<2" "decorator>=4.4.0" "sympy>=1.5.1" \
    cffi attrs cython pyyaml scipy requests psutil absl-py pybind11 \
    "omegaconf==2.4.0.dev4" "hydra-core==1.4.0.dev1" einops nvitop \
    "setuptools>=69.5.1,<75.9" ninja tensorboard "swanlab>=0.6.11"

"${PYTHON_BIN}" -m pip install "${TORCH_WHEEL_URL}" "${TORCH_NPU_WHEEL_URL}"
"${PYTHON_BIN}" -m pip install "torchvision==0.21.0" \
    --index-url https://download.pytorch.org/whl/cpu
"${PYTHON_BIN}" -m pip install -i "${TEST_PYPI_INDEX_URL}" "triton-ascend<3.2.0rc"

if [[ ! -d "${SGL_KERNEL_NPU_DIR}/.git" ]]; then
    git clone --branch "${SGL_KERNEL_NPU_REF}" \
        https://github.com/sgl-project/sgl-kernel-npu.git "${SGL_KERNEL_NPU_DIR}"
fi
(
    cd "${SGL_KERNEL_NPU_DIR}"
    bash build.sh
    "${PYTHON_BIN}" -m pip install output/sgl*.whl output/torch*.whl output/deepep*.whl
)

DEEP_EP_SITE="$("${PYTHON_BIN}" -c "import os, deep_ep; print(os.path.dirname(os.path.dirname(deep_ep.__file__)))")"
for library in "${DEEP_EP_SITE}"/deep_ep/deep_ep_cpp*.so; do
    [[ -e "${library}" ]] || continue
    ln -sf "${library}" "${DEEP_EP_SITE}/"
done

if [[ ! -d "${SGLANG_DIR}/.git" ]]; then
    git clone --branch "v${SGLANG_VERSION}" \
        https://github.com/sgl-project/sglang.git "${SGLANG_DIR}"
fi
"${PYTHON_BIN}" -m pip install -e "${SGLANG_DIR}/python[srt_npu]" --no-build-isolation

"${PYTHON_BIN}" -m pip install \
    accelerate "ray[default]>=2.47.0" pylatexenc datasets \
    "latex2sympy2 @ git+https://github.com/RLinf/latex2sympy2.git" \
    sentencepiece torchdata "wandb<0.25.1" word2number regex \
    "transformers==4.56.1" -i "${PYPI_INDEX_URL}"

# SGLang NPU packages provide triton-ascend instead of the CUDA triton wheel.
"${PYTHON_BIN}" -m pip uninstall -y triton triton-ascend
"${PYTHON_BIN}" -m pip install -i "${TEST_PYPI_INDEX_URL}" "triton-ascend<3.2.0rc"

echo "Ascend environment installation completed."
