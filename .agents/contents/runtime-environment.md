# Runtime Environment Context

- Default conda environment: `retargeting`.
- Python path: `/home/ymr/miniconda3/envs/retargeting/bin/python`.
- The current environment provides the headless-test dependencies: `pytest`, `numpy`, `scipy`, `scikit-learn`, `nlopt`, `pinocchio`, and PyTorch. Treat installed versions as runtime state rather than a project compatibility contract; check them when diagnosing an environment issue.
- The last checked GPU build used PyTorch `2.11.0+cu128`, `torchvision==0.26.0+cu128`, and `torchaudio==2.11.0+cu128`.
- PyTorch CUDA works when run outside the default Codex sandbox with escalated execution.
- In the default Codex command sandbox, `/dev/nvidia*` may be hidden, causing `torch.cuda.is_available()` to return `False`.

Inspect the active environment without changing it:

```bash
/home/ymr/miniconda3/envs/retargeting/bin/python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.cuda.is_available())"
```
