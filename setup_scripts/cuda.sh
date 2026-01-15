#!/bin/sh
set -eu

echo "➡ Updating package lists and installing dependencies"
sudo apt-get update
sudo apt-get install -y \
  ubuntu-drivers-common \
  cmake ccache nvtop xorg nvidia-settings \
  build-essential libomp-dev

############################
# USER CONFIG
############################
CUDA_VERSION=128   # 130=13.0, 131=13.1, 124=12.4

# POSIX-safe split (128 -> 12 / 8)
CUDA_MAJOR="$(printf '%s' "$CUDA_VERSION" | cut -c1-2)"
CUDA_MINOR="$(printf '%s' "$CUDA_VERSION" | cut -c3)"

############################
# DETECT OS VERSION
############################
if [ ! -f /etc/os-release ]; then
  echo "❌ Cannot detect OS"
  exit 1
fi

. /etc/os-release

if [ "$ID" != "ubuntu" ]; then
  echo "❌ This script supports Ubuntu only"
  exit 1
fi

# 22.04 -> 2204, 24.04 -> 2404
UBUNTU_VERSION="$(printf '%s' "$VERSION_ID" | tr -d '.')"

if [ "$UBUNTU_VERSION" != "2204" ] && [ "$UBUNTU_VERSION" != "2404" ]; then
  echo "❌ Unsupported Ubuntu version: $VERSION_ID"
  exit 1
fi

############################
# DETECT ARCH
############################
ARCH="$(uname -m)"

case "$ARCH" in
  x86_64)
    CUDA_ARCH="x86_64"
    ;;
  aarch64|arm64)
    CUDA_ARCH="sbsa"
    ;;
  *)
    echo "❌ Unsupported architecture: $ARCH"
    exit 1
    ;;
esac

############################
# INSTALL CUDA KEYRING
############################
KEYRING_PKG="cuda-keyring_1.1-1_all.deb"
CUDA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/${CUDA_ARCH}"

echo "➡ Installing CUDA keyring for ubuntu${UBUNTU_VERSION} / ${CUDA_ARCH}"
wget -q "${CUDA_REPO_URL}/${KEYRING_PKG}"
sudo dpkg -i "${KEYRING_PKG}"
rm -f "${KEYRING_PKG}"

############################
# INSTALL CUDA TOOLKIT
############################
echo "➡ Installing CUDA Toolkit ${CUDA_MAJOR}.${CUDA_MINOR}"
sudo apt-get update
sudo apt-get install -y "cuda-toolkit-${CUDA_MAJOR}-${CUDA_MINOR}"

############################
# SET ENV (IDEMPOTENT)
############################
echo "➡ Setting CUDA environment variables"

############################
# FIX CUDA PATH (from CUDA_VERSION)
############################

CUDA_INSTALL_DIR="/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}"

if [ ! -d "$CUDA_INSTALL_DIR" ]; then
  echo "❌ CUDA directory not found: $CUDA_INSTALL_DIR"
  exit 1
fi

# Create /usr/local/cuda symlink (active CUDA)
if [ ! -L /usr/local/cuda ]; then
  echo "➡ Creating symlink /usr/local/cuda -> cuda-${CUDA_MAJOR}.${CUDA_MINOR}"
  sudo ln -s "$CUDA_INSTALL_DIR" /usr/local/cuda
fi

# Persist environment variables (idempotent)
BASHRC="$HOME/.bashrc"

grep -q 'CUDA_HOME=' "$BASHRC" || echo 'export CUDA_HOME=/usr/local/cuda' >> "$BASHRC"
grep -q '/cuda/bin' "$BASHRC"  || echo 'export PATH=$CUDA_HOME/bin:$PATH' >> "$BASHRC"
grep -q '/cuda/lib64' "$BASHRC" || echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> "$BASHRC"


############################
# VERIFY
############################
echo "➡ Verifying installation"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "⚠ nvcc not found (open new shell or check PATH)"
fi

echo "✅ CUDA Toolkit installation complete"
