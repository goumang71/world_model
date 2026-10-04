"""Check the author action mapping and canonical evaluator against saved runs."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "World-Model-2018-main"))
from models.vae import VAE
from models.mdnrnn import MDNRNN
from models.controller import Controller
from paper_controller import PaperController, warm_start_from_local
from paper_controller_evaluation import PaperCanonicalEvaluator


def main():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v = VAE(3, 32).to(device).eval()
    m = MDNRNN(32, 3, 256, 5).to(device).eval()
    old = Controller(288, 3).to(device).eval()
    for obj, name in ((v, "vae"), (m, "memory"), (old, "controller")):
        obj.load_state_dict(torch.load(ROOT / "World-Model-2018-main" / "checkpoints" / f"{name}.pth",
                                       map_location=device, weights_only=True))
    c = warm_start_from_local(PaperController(288, 3).to(device), old.state_dict()).eval()
    x = torch.nn.utils.parameters_to_vector(c.parameters()).detach().cpu().numpy()
    evaluator = PaperCanonicalEvaluator(2, v, m, device)
    try:
        rows = evaluator.evaluate([x, x], [100000, 100000], [160900001, 161900004])
    finally:
        evaluator.close()
    for row in rows:
        path = ROOT / "outputs" / "paper_mapping_development_candidate_20261003" / "cases" / f"track_{row['seed']}_noise_{row['latent_seed']}.json"
        ref = json.loads(path.read_text(encoding="utf-8"))
        assert abs(row["episode_return"] - ref["episode_return"]) < 1e-5
        assert row["fully_off_track_steps"] == ref["fully_off_steps"]
    print("PAPER_CANONICAL_PARITY_PASS", flush=True)


if __name__ == "__main__":
    main()
