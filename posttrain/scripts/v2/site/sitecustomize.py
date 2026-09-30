# Plan v2.11: torch 2.11 on the cluster sends attention to cuDNN, which rebuilt its plan every
# step (profiled 2026-09-29: 6x slower). Switch it off for every Python process of this project.
try:
    import torch
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
except Exception:
    pass
