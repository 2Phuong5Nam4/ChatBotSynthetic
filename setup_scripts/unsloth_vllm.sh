uv venv --python 3.12
source .venv/bin/activate
export UV_TORCH_BACKEND=cu128
uv pip install unsloth vllm --torch-backend=${UV_TORCH_BACKEND}