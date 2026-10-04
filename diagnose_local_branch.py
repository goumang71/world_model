"""Replay a canonical prefix, then compare paired local steering interventions.

Run from the workspace root with .venv/Scripts/python.exe -B diagnose_local_branch.py.
Each case is written separately and complete cases are skipped on restart.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "World-Model-2018-main"
sys.path.insert(0, str(REPO))
from models.vae import VAE
from models.mdnrnn import MDNRNN
from models.controller import Controller
from driving_hud import driving_metrics

BASE = ROOT / "outputs" / "action_cause_diagnosis_20261002"
DEFAULT_OUT = ROOT / "outputs" / "local_branch_diagnosis_20261003"
STATES = {4470: 141, 52: 263}  # one-based first intervention step
HORIZON = 150
MODES = ("baseline", "smooth_05")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def models(device):
    vision = VAE(3, 32).to(device).eval()
    memory = MDNRNN(32, 3, 256, 5).to(device).eval()
    controller = Controller(288, 3).to(device).eval()
    for model, name in ((vision, "vae"), (memory, "memory"), (controller, "controller")):
        state = torch.load(REPO / "checkpoints" / f"{name}.pth", map_location=device, weights_only=True)
        model.load_state_dict(state)
    return vision, memory, controller


def get_action(obs, hidden, previous, vision, controller, device, *, generator=None, smooth=False):
    image = torch.as_tensor(obs.copy(), device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    image = F.interpolate(image[:, :, :-12, :], size=(96, 96), mode="bicubic")
    mu, logvar = vision.encode(image)
    if generator is None:
        z = vision.reparameterize(mu, logvar)
    else:
        eps = torch.randn(mu.shape, device=device, generator=generator)
        z = mu + eps * torch.exp(0.5 * logvar)
    action = controller(torch.cat([z, hidden[0].squeeze(0)], dim=-1)).clone()
    if smooth:
        action[:, 0] = 0.5 * action[:, 0] + 0.5 * previous
    action[:, 0] = torch.clamp(action[:, 0], min=previous - 1.0, max=previous + 1.0)
    return z, action


def run(seed, branch_step, noise, mode, expected, vision, memory, controller, device):
    env = gym.make("CarRacing-v3", continuous=True, lap_complete_percent=1.0,
                   domain_randomize=False, max_episode_steps=1000)
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        obs, _ = env.reset(seed=seed)
        hidden = memory.rnn.init_hidden(1, device)
        previous = 0.0
        prefix_actions = []
        with torch.inference_mode():
            for i in range(branch_step - 1):
                z, action = get_action(obs, hidden, previous, vision, controller, device)
                actual = action[0].cpu().tolist()
                if not np.allclose(actual, expected[i]["action"], rtol=0, atol=1e-6):
                    raise AssertionError(f"Prefix action mismatch seed={seed}, step={i+1}: {actual} vs {expected[i]['action']}")
                car = env.unwrapped.car
                if not np.allclose(np.asarray(car.hull.position), expected[i]["position"], rtol=0, atol=1e-4):
                    raise AssertionError(f"Prefix position mismatch seed={seed}, step={i+1}")
                prefix_actions.append(actual)
                previous = action[0, 0].item()
                obs, _, terminated, truncated, _ = env.step(action[0].cpu().numpy())
                _, hidden = memory.rnn(z.unsqueeze(1), action.unsqueeze(1), hidden)
                if terminated or truncated:
                    raise AssertionError(f"Prefix ended before branch seed={seed}, step={i+1}")
            # Both branches use the same post-prefix draws. The original-stream
            # replica copies the exact PyTorch global RNG state at this point.
            gen = torch.Generator(device=device)
            if noise == "original_stream":
                gen.set_state(torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state())
            else:
                gen.manual_seed(int(noise))
            pre = dict(
                observation_sha256=sha(obs.tobytes()),
                hidden_sha256=sha(hidden[0].cpu().numpy().tobytes() + hidden[1].cpu().numpy().tobytes()),
                prefix_actions_sha256=sha(np.asarray(prefix_actions, dtype=np.float32).tobytes()),
                position=[float(x) for x in env.unwrapped.car.hull.position],
                speed=float(np.linalg.norm(np.asarray(env.unwrapped.car.hull.linearVelocity))),
                road_wheels=driving_metrics(env)["road_wheels"],
                previous_steering=previous,
                coverage=driving_metrics(env)["track_coverage"],
            )
            assert np.allclose(pre["position"], expected[branch_step - 1]["position"], rtol=0, atol=1e-4)
            rows = []
            for i in range(branch_step - 1, min(branch_step - 1 + HORIZON, 1000)):
                before = driving_metrics(env)
                speed = float(np.linalg.norm(np.asarray(env.unwrapped.car.hull.linearVelocity)))
                z, action = get_action(obs, hidden, previous, vision, controller, device,
                                       generator=gen, smooth=(mode == "smooth_05"))
                previous = action[0, 0].item()
                obs, reward, terminated, truncated, info = env.step(action[0].cpu().numpy())
                _, hidden = memory.rnn(z.unsqueeze(1), action.unsqueeze(1), hidden)
                after = driving_metrics(env)
                rows.append(dict(step=i + 1, action=action[0].cpu().tolist(),
                                 reward=float(reward), road_wheels_before=before["road_wheels"],
                                 road_wheels_after=after["road_wheels"], speed=speed,
                                 track_coverage=after["track_coverage"],
                                 terminated=bool(terminated), truncated=bool(truncated),
                                 lap_finished=bool(info.get("lap_finished", False))))
                if terminated or truncated:
                    break
            if noise == "original_stream" and mode == "baseline":
                for row in rows:
                    ref = expected[row["step"] - 1]
                    if not np.allclose(row["action"], ref["action"], rtol=0, atol=1e-6):
                        raise AssertionError(f"Original suffix action mismatch seed={seed}, step={row['step']}")
                    if row["road_wheels_after"] != ref["road_wheels_after"]:
                        raise AssertionError(f"Original suffix wheel mismatch seed={seed}, step={row['step']}")
            first = next((r["step"] for r in rows if r["road_wheels_after"] == 0), None)
            summary = dict(seed=seed, branch_step=branch_step, noise=noise, mode=mode,
                           post_steps=len(rows), first_full_off_step=first,
                           fully_off_steps=sum(r["road_wheels_after"] == 0 for r in rows),
                           partly_off_steps=sum(r["road_wheels_after"] < 4 for r in rows),
                           reward_sum=sum(r["reward"] for r in rows),
                           coverage_change=rows[-1]["track_coverage"] - pre["coverage"],
                           final_coverage=rows[-1]["track_coverage"],
                           stopped_early=len(rows) < HORIZON,
                           original_trace_match=bool(noise == "original_stream" and mode == "baseline"))
            return dict(prefix=pre, summary=summary, trajectory=rows)
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
    for relative, digest in old["hashes"].items():
        if sha((ROOT / relative).read_bytes()) != digest:
            raise RuntimeError(f"Previously diagnosed source/weight changed: {relative}")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = dict(states=STATES, horizon=HORIZON, modes=MODES,
                  noises={str(s): ["original_stream"] + [10000000 + s * 1009 + j * 1000003 for j in range(1, 5)] for s in STATES},
                  environment="CarRacing-v3; 1000-step max; lap_complete_percent=1.0; domain_randomize=False",
                  source_hash=sha(Path(__file__).read_bytes()), prior_hashes=old["hashes"], device=str(device))
    if (out / "config.json").exists():
        if json.loads((out / "config.json").read_text(encoding="utf-8")) != json.loads(json.dumps(config)):
            raise RuntimeError("Protocol or script changed. Use a new output directory.")
    else:
        atomic_json(out / "config.json", config)
        (out / "source_diagnose_local_branch.py").write_bytes(Path(__file__).read_bytes())
    vision, memory, controller = models(device)
    complete = []
    for seed, branch_step in STATES.items():
        expected = json.loads((BASE / "baseline" / f"seed_{seed}" / f"trace_seed{seed}.json").read_text())
        for j, noise in enumerate(config["noises"][str(seed)]):
            for mode in MODES:
                name = f"seed_{seed}/noise_{j}_{mode}"
                target = out / name / "result.json"
                if target.exists():
                    complete.append(name)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(out / "status.json", dict(status="running", current=name, completed=complete, total=20))
                print("Running", name, flush=True)
                result = run(seed, branch_step, noise, mode, expected, vision, memory, controller, device)
                atomic_json(target, result)
                complete.append(name)
                atomic_json(out / "status.json", dict(status="running", completed=complete, total=20))
    atomic_json(out / "status.json", dict(status="completed", completed=complete, total=20))
    print("Completed", len(complete), "paired branches", flush=True)


if __name__ == "__main__":
    main()
