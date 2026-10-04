"""Measure sampled-z versus posterior-mean gas on identical policy states."""
import json
import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "World-Model-2018-main"
DEV = ROOT / "outputs" / "paper_channel_ablation_dev_20261003"
sys.path.insert(0, str(REPO))
from models.vae import VAE
from models.mdnrnn import MDNRNN
from paper_controller import PaperController


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("sample", "mean_gas"), default="sample")
    mode = parser.parse_args().mode
    out = ROOT / "outputs" / f"paper_gas_channel_probe_{mode}_20261003"
    out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision = VAE(3, 32).to(device).eval()
    memory = MDNRNN(32, 3, 256, 5).to(device).eval()
    controller = PaperController(288, 3).to(device).eval()
    for model, path in ((vision, REPO / "checkpoints" / "vae.pth"),
                        (memory, REPO / "checkpoints" / "memory.pth"),
                        (controller, ROOT / "outputs" / "paper_controller_aligned_20261003" /
                         "tested_checkpoints" / "controller.pth")):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    expected = json.loads((DEV / f"paper_{mode}.json").read_text(encoding="utf-8"))
    selected = [r for r in expected if r["latent_seed"] == 170000001 + r["seed"] * 1009]
    assert len(selected) == 16
    env = gym.make("CarRacing-v3", continuous=True, lap_complete_percent=1.0,
                   domain_randomize=False, max_episode_steps=1000)
    records = []
    try:
        for case in selected:
            obs, _ = env.reset(seed=case["seed"])
            hidden = memory.rnn.init_hidden(1, device)
            generator = torch.Generator(device=device).manual_seed(case["latent_seed"])
            rows = []
            total = 0.0
            with torch.inference_mode():
                for step in range(1, 1001):
                    image = torch.as_tensor(obs.copy(), device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                    image = F.interpolate(image[:, :, :-12, :], size=(96, 96), mode="bicubic")
                    mu, logvar = vision.encode(image)
                    std = (.5 * logvar).exp()
                    z = mu + torch.randn(std.shape, device=device, generator=generator) * std
                    h = hidden[0].squeeze(0)
                    sampled = controller(torch.cat([z, h], dim=-1)).clone()
                    mean = controller(torch.cat([mu, h], dim=-1))
                    sampled_gas = float(sampled[0, 1])
                    mean_gas = float(mean[0, 1])
                    rows.append(dict(step=step, sampled_gas=sampled_gas, mean_gas=mean_gas,
                                     sampled_steer=float(sampled[0, 0]), mean_steer=float(mean[0, 0]),
                                     sampled_brake=float(sampled[0, 2]), mean_brake=float(mean[0, 2]),
                                     posterior_std=float(std.mean())))
                    executed = sampled.clone()
                    if mode == "mean_gas":
                        executed[:, 1] = mean[:, 1]
                    obs, reward, terminated, truncated, _ = env.step(executed[0].cpu().numpy())
                    _, hidden = memory.rnn(z.unsqueeze(1), executed.unsqueeze(1), hidden)
                    total += float(reward)
                    if terminated or truncated:
                        break
            if abs(total - case["episode_return"]) > 1e-5 or len(rows) != case["steps"]:
                raise RuntimeError(f"Evaluation parity failed for seed {case['seed']}: {total} vs {case['episode_return']}")
            records.append(dict(seed=case["seed"], latent_seed=case["latent_seed"],
                                episode_return=total, severe_failure=case["severe_failure"], rows=rows))
            (out / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            print("GAS_PROBE", len(records), "/", len(selected), case["seed"], flush=True)
    finally:
        env.close()
    flat = [r for case in records for r in case["rows"]]
    sample = np.array([r["sampled_gas"] for r in flat])
    mean = np.array([r["mean_gas"] for r in flat])
    delta = mean - sample
    summary = dict(mode=mode,cases=len(records), steps=len(flat),
                   sampled_gas_mean=float(sample.mean()), mean_gas_mean=float(mean.mean()),
                   mean_minus_sample_gas=float(delta.mean()),
                   fraction_mean_gas_higher=float(np.mean(delta > 0)),
                   fraction_absolute_gas_gap_over_point_two=float(np.mean(np.abs(delta) > .2)),
                   sampled_gas_over_point_eight=float(np.mean(sample > .8)),
                   mean_gas_over_point_eight=float(np.mean(mean > .8)),
                   sampled_gas_under_point_two=float(np.mean(sample < .2)),
                   mean_gas_under_point_two=float(np.mean(mean < .2)),
                   episode_differences=[dict(seed=c["seed"], severe_failure=c["severe_failure"],
                                             mean_minus_sample_gas=float(np.mean([r["mean_gas"] - r["sampled_gas"] for r in c["rows"]])))
                                        for c in records])
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps({k:v for k,v in summary.items() if k!="episode_differences"}), flush=True)


if __name__ == "__main__":
    main()
