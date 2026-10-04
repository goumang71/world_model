"""Require training rollout and canonical per-car inference to agree."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "World-Model-2018-main"))
from models.vae import VAE
from models.mdnrnn import MDNRNN
from paper_controller import PaperController
from paper_controller_evaluation import PaperCanonicalEvaluator
from paper_train_evaluator import Evaluator


def main():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cp = ROOT / "World-Model-2018-main" / "checkpoints"
    vision = VAE(3, 32).to(device).eval()
    memory = MDNRNN(32, 3, 256, 5).to(device).eval()
    vision.load_state_dict(torch.load(cp / "vae.pth", map_location=device, weights_only=True))
    memory.load_state_dict(torch.load(cp / "memory.pth", map_location=device, weights_only=True))
    checkpoint = ROOT / "outputs" / "paper_controller_train_20261003" / "generations" / "controller_gen004.pth"
    controller = PaperController(288, 3).to(device).eval()
    controller.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    params = torch.nn.utils.parameters_to_vector(controller.parameters()).detach().cpu().numpy()
    track_seeds = list(range(140000, 140008))
    latent_seeds = [300000001 + seed * 1009 for seed in track_seeds]
    train = Evaluator(16, vision, memory, device)
    canonical = PaperCanonicalEvaluator(8, vision, memory, device)
    try:
        # Duplicate 8 cases to reproduce the 16 lanes used for CMA-ES.
        train_returns, train_steps = train.evaluate([params] * 16, track_seeds * 2,
                                                     latent_seeds=latent_seeds * 2)
        canonical_rows = canonical.evaluate([params] * 8, track_seeds, latent_seeds)
    finally:
        train.close()
        canonical.close()
    canonical_returns = np.array([r["episode_return"] for r in canonical_rows])
    canonical_steps = np.array([r["steps"] for r in canonical_rows])
    differences = train_returns[:8] - canonical_returns
    print("seed train canonical difference train_steps canonical_steps")
    for i in range(8):
        print(track_seeds[i], round(train_returns[i], 4), round(canonical_returns[i], 4),
              round(differences[i], 4), train_steps[i], canonical_steps[i])
    print("max_abs_return_difference", float(np.max(np.abs(differences))))
    print("max_abs_duplicate_difference", float(np.max(np.abs(train_returns[:8] - train_returns[8:]))))
    assert np.allclose(train_returns[:8], canonical_returns, atol=1e-5, rtol=0)
    assert np.allclose(train_returns[:8], train_returns[8:], atol=1e-5, rtol=0)
    assert np.array_equal(train_steps[:8], canonical_steps)
    assert np.array_equal(train_steps[:8], train_steps[8:])
    print("PAPER_TRAIN_CANONICAL_PARITY_PASS")


if __name__ == "__main__":
    main()
