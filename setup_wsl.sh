#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
env_dir=${SPLICE_ENV_DIR:-"$HOME/.local/share/splice-probing/venv"}
if ! command -v uv >/dev/null && [[ -x "$HOME/.local/bin/uv" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

uv python install 3.10.20
if [[ ! -x "$env_dir/bin/python" ]]; then
    uv venv --python 3.10.20 "$env_dir"
fi
"$env_dir/bin/python" -c 'import sys; assert sys.version_info[:3] == (3, 10, 20)'
requirements="$project_dir/requirements.txt"
uv pip install --python "$env_dir/bin/python" \
    --index https://download.pytorch.org/whl/cu118 --index-strategy unsafe-best-match \
    -r "$requirements"
uv pip check --python "$env_dir/bin/python"
"$env_dir/bin/python" -c 'import torch, mamba_ssm.ops.triton.layernorm, causal_conv1d_cuda, selective_scan_cuda; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'
