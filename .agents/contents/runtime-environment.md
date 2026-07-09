# Runtime Environment Context

- Default conda environment: `retargeting`.
- Python path: `/home/ymr/miniconda3/envs/retargeting/bin/python`.
- Installed for Phase 0: `pytest`, `numpy`, `scipy`, `scikit-learn`, `nlopt`, `pinocchio`, `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`, `torchaudio==2.11.0+cu128`.
- PyTorch CUDA works when run outside the default Codex sandbox with escalated execution.
- In the default Codex command sandbox, `/dev/nvidia*` may be hidden, causing `torch.cuda.is_available()` to return `False`.
