# Source before Python/YOLO on Jetson if PyTorch fails with libopenblas.so.0:
#   source scripts/jetson_gpu_env.sh
#
# Permanent fix (recommended): install OpenBLAS once system-wide:
#   sudo apt-get update && sudo apt-get install -y libopenblas0-pthread
#
# After apt install, the linker usually finds libopenblas without extra PATH.
# This file adds the pthread OpenBLAS dir if the library is present.

_blas_dir="/usr/lib/aarch64-linux-gnu/openblas-pthread"
if [ -d "$_blas_dir" ] && [ -f "$_blas_dir/libopenblas.so.0" ]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${_blas_dir}:"*) ;;
    *) export LD_LIBRARY_PATH="${_blas_dir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
fi
unset _blas_dir
