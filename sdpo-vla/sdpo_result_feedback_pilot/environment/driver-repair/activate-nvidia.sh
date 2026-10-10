# Scope matching NVIDIA userspace libraries to this conda environment only.
_sdpo_driver_version=580.173.02
_sdpo_loaded_version=$(cat /sys/module/nvidia/version 2>/dev/null || true)
_sdpo_driver_lib="$CONDA_PREFIX/opt/nvidia-driver-$_sdpo_driver_version/usr/lib/x86_64-linux-gnu"
if [ "$_sdpo_loaded_version" = "$_sdpo_driver_version" ] && [ -f "$_sdpo_driver_lib/libcuda.so.1" ]; then
    if [ "${_SDPO_PILOT_NVIDIA_ACTIVE:-0}" != 1 ]; then
        export _SDPO_PILOT_PRE_LD_WAS_SET="${LD_LIBRARY_PATH+x}"
        export _SDPO_PILOT_PRE_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
        export LD_LIBRARY_PATH="$_sdpo_driver_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export _SDPO_PILOT_NVIDIA_ACTIVE=1
    fi
else
    printf '%s\n' "sdpo-vla-pilot: matching-library override not applied (loaded driver: ${_sdpo_loaded_version:-none}); recheck CUDA after driver changes." >&2
fi
unset _sdpo_driver_version _sdpo_loaded_version _sdpo_driver_lib
