#!/bin/sh
set -eu
sh setup_scripts/cuda.sh
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Hugging Face CLI (REQUIRES bash)
curl -LsSf https://hf.co/cli/install.sh | bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"


if [ ! -f "$ENV_FILE" ]; then
  echo "❌ .env file not found"
  exit 1
fi

set -a
. "$ENV_FILE"
set +a

if [ -z "${HF_TOKEN:-}" ]; then
  echo "❌ HF_TOKEN not set in .env"
  exit 1
fi

echo "➡ Installing Git LFS"

if ! command -v git >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y git
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y git-lfs
fi

git lfs install

echo "➡ Logging into Hugging Face using HF_TOKEN"
git config --global credential.helper store
hf auth login --token "$HF_TOKEN" --add-to-git-credential
