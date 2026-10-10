# Restore the caller's original library path, including its unset state.
if [ "${_SDPO_PILOT_NVIDIA_ACTIVE:-0}" = 1 ]; then
    if [ "${_SDPO_PILOT_PRE_LD_WAS_SET:-}" = x ]; then
        export LD_LIBRARY_PATH="${_SDPO_PILOT_PRE_LD_LIBRARY_PATH-}"
    else
        unset LD_LIBRARY_PATH
    fi
fi
unset _SDPO_PILOT_PRE_LD_WAS_SET _SDPO_PILOT_PRE_LD_LIBRARY_PATH _SDPO_PILOT_NVIDIA_ACTIVE
