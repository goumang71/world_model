"""Compare candidate failures with its successes on the same tracks."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from trace_paper_failures import ROOT, REPO, RUN, OUT, rollout, window_stats
sys.path.insert(0, str(REPO))
from models.vae import VAE
from models.mdnrnn import MDNRNN
from paper_controller import PaperController


def main():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision = VAE(3, 32).to(device).eval()
    memory = MDNRNN(32, 3, 256, 5).to(device).eval()
    paper = PaperController(288, 3).to(device).eval()
    for model, path in ((vision, REPO / "checkpoints" / "vae.pth"),
                        (memory, REPO / "checkpoints" / "memory.pth"),
                        (paper, RUN / "tested_checkpoints" / "controller.pth")):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    comparison = json.loads((RUN / "comparison.json").read_text(encoding="utf-8"))
    failures = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    candidates = defaultdict(list)
    for row in comparison["candidate_rows"]:
        if not row["severe_failure"]:
            candidates[row["seed"]].append(row)
    controls = {}
    env = gym.make("CarRacing-v3", continuous=True, lap_complete_percent=1.0,
                   domain_randomize=False, max_episode_steps=1000)
    try:
        for seed in sorted(set(row["seed"] for row in failures)):
            control = max(candidates[seed], key=lambda row: row["episode_return"])
            total, rows = rollout(env, vision, memory, paper, seed,
                                  control["latent_seed"], device, True)
            if abs(total - control["episode_return"]) > 1e-5:
                raise RuntimeError(f"Control trace parity failed: {seed}")
            controls[seed] = dict(latent_seed=control["latent_seed"], episode_return=total,
                                  trace=rows)
            print("CONTROL", len(controls), "/", 10, seed, round(total, 1), flush=True)
    finally:
        env.close()
    comparisons = []
    for failure in failures:
        seed = failure["seed"]
        at = failure["candidate_first_sustained_off"]
        if at is None:
            continue
        success_window = window_stats(controls[seed]["trace"], at)
        fail_window = failure["candidate_before_event"]
        comparisons.append(dict(seed=seed, failed_latent_seed=failure["latent_seed"],
                                success_latent_seed=controls[seed]["latent_seed"],
                                failed_window=fail_window, success_window=success_window,
                                event_step=at))
    OUT.joinpath("matched_success_comparison.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8")
    for key in ("mean_abs_steer", "mean_gas", "mean_brake", "mean_speed",
                "large_steer_reversals"):
        failed = np.array([x["failed_window"][key] for x in comparisons])
        success = np.array([x["success_window"][key] for x in comparisons])
        print(key, "failed_mean", float(failed.mean()), "same_track_success_mean",
              float(success.mean()), "failed_higher", int((failed > success).sum()),
              "of", len(comparisons), flush=True)


if __name__ == "__main__":
    main()
