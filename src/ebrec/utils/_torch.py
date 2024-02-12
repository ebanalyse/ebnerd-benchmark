from pathlib import Path

try:
    import torch
except ImportError:
    print("torch not available")


def save_checkpoint(model, path="model_state_dict.pt"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving model weights: {path}")
    torch.save(model.state_dict(), path.as_posix())
