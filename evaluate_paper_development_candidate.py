"""Resumable broad controller diagnosis on fresh tracks and latent-noise streams.

Uses singleton neural inference per lane, parallel simulator stepping, and records
simulator-only metrics for analysis. Policy inputs never contain simulator metrics.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.vector import AsyncVectorEnv

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "World-Model-2018-main"
sys.path.insert(0, str(REPO))
from models.vae import VAE
from models.mdnrnn import MDNRNN
from models.controller import Controller

OUT = ROOT / "outputs" / "paper_mapping_development_candidate_20261003"
TRACKS = range(100000, 100016)
REPLICATES = 3
BATCH = 8


def metric_env():
    import gymnasium as gym

    class Metrics(gym.Wrapper):
        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            car = self.unwrapped.car
            info["road_wheels"] = sum(bool(w.tiles) for w in car.wheels)
            info["speed"] = float(np.linalg.norm(np.asarray(car.hull.linearVelocity)))
            info["coverage"] = float(self.unwrapped.tile_visited_count / len(self.unwrapped.track))
            return obs, reward, terminated, truncated, info

    return Metrics(gym.make("CarRacing-v3", continuous=True, lap_complete_percent=1.0,
                            domain_randomize=False, max_episode_steps=1000))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_models(device):
    models = (VAE(3, 32).to(device).eval(), MDNRNN(32, 3, 256, 5).to(device).eval(),
              Controller(288, 3).to(device).eval())
    for model, name in zip(models, ("vae", "memory", "controller")):
        model.load_state_dict(torch.load(REPO / "checkpoints" / f"{name}.pth", map_location=device, weights_only=True))
    # (tanh(x/2)+1)/2 == sigmoid(x): preserve initial gas output exactly.
    with torch.no_grad():
        models[2].model[0].weight[1].mul_(0.5)
        models[2].model[0].bias[1].mul_(0.5)
    return models


def run_batch(cases, vision, memory, controller, device):
    envs = AsyncVectorEnv([metric_env] * len(cases), shared_memory=True, context="spawn")
    n = len(cases)
    try:
        observations, _ = envs.reset(seed=[t for t, _ in cases])
        hidden = [memory.rnn.init_hidden(1, device) for _ in cases]
        generators = [torch.Generator(device=device).manual_seed(s) for _, s in cases]
        previous = [0.0] * n
        done = np.zeros(n, dtype=bool)
        totals = np.zeros(n)
        traces = [[] for _ in cases]
        with torch.inference_mode():
            for step in range(1, 1001):
                active = ~done.copy()
                actions_np = np.zeros((n, 3), dtype=np.float32)
                zs = [None] * n
                actions = [None] * n
                speeds_before = [None] * n
                for i in np.flatnonzero(active):
                    image = torch.as_tensor(observations[i].copy(), device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                    image = F.interpolate(image[:, :, :-12, :], size=(96, 96), mode="bicubic")
                    mu, logvar = vision.encode(image)
                    eps = torch.randn(mu.shape, device=device, generator=generators[i])
                    z = mu + eps * torch.exp(0.5 * logvar)
                    raw_logits = controller.model(torch.cat([z, hidden[i][0].squeeze(0)], dim=-1))
                    action = torch.tanh(raw_logits)
                    action = torch.cat([action[:, 0:1], (action[:, 1:2] + 1.0) / 2.0,
                                        torch.clamp(action[:, 2:3], 0.0, 1.0)], dim=1)
                    raw_action = action
                    # Original author implementation issues steering directly.
                    previous[i] = action[0, 0].item()
                    if not torch.isfinite(action).all():
                        raise ValueError("Non-finite action")
                    zs[i] = z
                    actions[i] = action
                    actions_np[i] = action[0].cpu().numpy()
                    # Only actual policy inputs and outputs, no privileged state.
                    traces[i].append(dict(step=step, raw_steer=float(raw_action[0, 0]),
                                          steer=float(action[0, 0]), gas=float(action[0, 1]),
                                          brake=float(action[0, 2]),
                                          mean_z_steer=float(controller(torch.cat([mu, hidden[i][0].squeeze(0)], dim=-1))[0, 0])))
                observations, rewards, terminated, truncated, info = envs.step(actions_np)
                road = np.asarray(info["road_wheels"])
                speed = np.asarray(info["speed"])
                coverage = np.asarray(info["coverage"])
                for i in np.flatnonzero(active):
                    totals[i] += float(rewards[i])
                    row = traces[i][-1]
                    row.update(reward=float(rewards[i]), road_wheels=int(road[i]),
                               speed=float(speed[i]), coverage=float(coverage[i]))
                    _, hidden[i] = memory.rnn(zs[i].unsqueeze(1), actions[i].unsqueeze(1), hidden[i])
                done |= terminated | truncated
                if done.all():
                    break
        results = []
        for i, (track, noise) in enumerate(cases):
            trace = traces[i]
            full = sum(r["road_wheels"] == 0 for r in trace)
            boundary_exit = bool(len(trace) < 1000 and trace[-1]["coverage"] < 1.0 - 1e-12)
            results.append(dict(track_seed=track, latent_seed=noise, episode_return=float(totals[i]),
                                steps=len(trace), fully_off_steps=full,
                                fully_off_fraction=full / len(trace),
                                boundary_exit=boundary_exit,
                                severe_failure=bool(totals[i] < 500 or full / len(trace) > .2 or boundary_exit),
                                trace=trace))
        return results
    finally:
        envs.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=OUT)
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = [(track, 60000001 + track * 1009 + repeat * 1000003)
             for track in TRACKS for repeat in range(REPLICATES)]
    protocol = dict(cases=cases, batch_size=BATCH, device=str(device),
                    environment="CarRacing-v3 continuous, lap_complete_percent=1.0, domain_randomize=False, max_episode_steps=1000",
                    source_sha256=digest(Path(__file__)),
                    evaluator_sha256=digest(ROOT / "evaluate_baseline.py"),
                    weight_sha256={name: digest(REPO / "checkpoints" / f"{name}.pth") for name in ("vae", "memory", "controller")},
                    purpose="author-paper action mapping, no steering slew, warm adapted downloaded C weights; development only")
    config = out / "config.json"
    if config.exists():
        if json.loads(config.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("Protocol/source/weights changed; use a new output directory")
    else:
        save(config, protocol)
        (out / "source_evaluate_paper_development_candidate.py").write_bytes(Path(__file__).read_bytes())
    vision, memory, controller = load_models(device)
    for start in range(0, len(cases), BATCH):
        batch = cases[start:start + BATCH]
        pending = [(t, s) for t, s in batch if not (out / "cases" / f"track_{t}_noise_{s}.json").exists()]
        if not pending:
            continue
        save(out / "status.json", dict(status="running", completed=len(cases) - sum(not (out / "cases" / f"track_{t}_noise_{s}.json").exists() for t, s in cases),
                                       current=pending, total=len(cases)))
        print("Running", pending, flush=True)
        for row in run_batch(pending, vision, memory, controller, device):
            save(out / "cases" / f"track_{row['track_seed']}_noise_{row['latent_seed']}.json", row)
    save(out / "status.json", dict(status="completed", completed=len(cases), total=len(cases)))
    print("Completed", len(cases), flush=True)


if __name__ == "__main__":
    main()
