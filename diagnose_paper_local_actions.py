"""Paired local interventions before new paper-controller off-track failures.

The 12 cases are post-test diagnostics. They are not an independent holdout.
"""
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
RUN = ROOT / "outputs" / "paper_controller_aligned_20261003"
TRACES = RUN / "failure_diagnosis"
OUT = RUN / "local_action_branches"
sys.path.insert(0, str(REPO))
from models.vae import VAE
from models.mdnrnn import MDNRNN
from paper_controller import PaperController

MODES = ("baseline", "gas_x08", "steer_slew_1")
NOISE_MODES = ("original_stream", "new_stream_1", "new_stream_2")
INTERVENTION_STEPS = 50
HORIZON = 200


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def action_for(obs, hidden, generator, vision, controller, device):
    image = torch.as_tensor(obs.copy(), device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    image = F.interpolate(image[:, :, :-12, :], size=(96, 96), mode="bicubic")
    mu, logvar = vision.encode(image)
    eps = torch.randn(logvar.shape, device=device, generator=generator)
    z = mu + eps * (0.5 * logvar).exp()
    action = controller(torch.cat([z, hidden[0].squeeze(0)], dim=-1)).clone()
    return z, action


def sustained_off_start(rows, min_run=10):
    run = 0
    for row in rows:
        run = run + 1 if row["fully_off"] else 0
        if run >= min_run:
            return row["step"] - run + 1
    return None


def run_case(case, expected, mode, noise_mode, vision, memory, controller, device):
    seed, latent_seed = case["seed"], case["latent_seed"]
    first_off = case["candidate_first_sustained_off"]
    branch_step = max(2, first_off - INTERVENTION_STEPS)
    env = gym.make("CarRacing-v3", continuous=True, lap_complete_percent=1.0,
                   domain_randomize=False, max_episode_steps=1000)
    try:
        obs, _ = env.reset(seed=seed)
        hidden = memory.rnn.init_hidden(1, device)
        generator = torch.Generator(device=device).manual_seed(latent_seed)
        previous_steering = 0.0
        prefix_actions = []
        with torch.inference_mode():
            for step in range(1, branch_step):
                z, action = action_for(obs, hidden, generator, vision, controller, device)
                actual = action[0].cpu().numpy()
                if not np.allclose(actual, [expected[step - 1][k] for k in ("steer", "gas", "brake")],
                                   rtol=0, atol=1e-6):
                    raise RuntimeError(f"Prefix action mismatch {seed}/{latent_seed} step {step}")
                previous_steering = float(action[0, 0])
                prefix_actions.append(actual.copy())
                obs, _, terminated, truncated, _ = env.step(actual)
                _, hidden = memory.rnn(z.unsqueeze(1), action.unsqueeze(1), hidden)
                actual_off = not any(bool(w.tiles) for w in env.unwrapped.car.wheels)
                if actual_off != expected[step - 1]["fully_off"]:
                    raise RuntimeError(f"Prefix wheel mismatch {seed}/{latent_seed} step {step}")
                if terminated or truncated:
                    raise RuntimeError("Prefix ended before intervention")
            prefix = dict(observation_sha256=digest(obs.tobytes()),
                          hidden_sha256=digest(hidden[0].cpu().numpy().tobytes() +
                                               hidden[1].cpu().numpy().tobytes()),
                          actions_sha256=digest(np.asarray(prefix_actions, dtype=np.float32).tobytes()),
                          rng_sha256=digest(generator.get_state().cpu().numpy().tobytes()),
                          branch_step=branch_step)
            if noise_mode != "original_stream":
                idx = int(noise_mode.rsplit("_", 1)[1])
                generator.manual_seed(500000000 + latent_seed + idx * 1000003)
            coverage_start = int(env.unwrapped.tile_visited_count)
            rows = []
            for step in range(branch_step, min(branch_step + HORIZON, 1001)):
                z, action = action_for(obs, hidden, generator, vision, controller, device)
                if step < branch_step + INTERVENTION_STEPS:
                    if mode == "gas_x08":
                        action[0, 1] *= .8
                    elif mode == "steer_slew_1":
                        action[0, 0] = torch.clamp(action[0, 0],
                                                   min=previous_steering - 1.0,
                                                   max=previous_steering + 1.0)
                previous_steering = float(action[0, 0])
                speed = float(np.linalg.norm(np.asarray(env.unwrapped.car.hull.linearVelocity)))
                obs, reward, terminated, truncated, _ = env.step(action[0].cpu().numpy())
                _, hidden = memory.rnn(z.unsqueeze(1), action.unsqueeze(1), hidden)
                off = not any(bool(w.tiles) for w in env.unwrapped.car.wheels)
                row = dict(step=step, action=action[0].cpu().tolist(), reward=float(reward),
                           fully_off=off, speed=speed,
                           tile_visited_count=int(env.unwrapped.tile_visited_count))
                rows.append(row)
                if noise_mode == "original_stream" and mode == "baseline":
                    reference = expected[step - 1]
                    if not np.allclose(row["action"],
                                       [reference[k] for k in ("steer", "gas", "brake")],
                                       rtol=0, atol=1e-6) or off != reference["fully_off"]:
                        raise RuntimeError(f"Suffix parity failed {seed}/{latent_seed} step {step}")
                if terminated or truncated:
                    break
            summary = dict(seed=seed, latent_seed=latent_seed, mode=mode, noise_mode=noise_mode,
                           branch_step=branch_step, steps=len(rows),
                           fully_off_steps=sum(r["fully_off"] for r in rows),
                           first_sustained_off_step=sustained_off_start(rows),
                           reward_sum=sum(r["reward"] for r in rows),
                           tiles_gained=rows[-1]["tile_visited_count"] - coverage_start,
                           terminated_early=len(rows) < HORIZON)
            return dict(prefix=prefix, summary=summary, trajectory=rows)
    finally:
        env.close()


def main():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision = VAE(3, 32).to(device).eval()
    memory = MDNRNN(32, 3, 256, 5).to(device).eval()
    controller = PaperController(288, 3).to(device).eval()
    paths = (REPO / "checkpoints" / "vae.pth",
             REPO / "checkpoints" / "memory.pth",
             RUN / "tested_checkpoints" / "controller.pth")
    for model, path in zip((vision, memory, controller), paths):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    cases = json.loads((TRACES / "summary.json").read_text(encoding="utf-8"))
    if len(cases) != 12:
        raise RuntimeError("Expected 12 prespecified candidate-only failures")
    OUT.mkdir(exist_ok=True)
    config = dict(cases=[(x["seed"], x["latent_seed"], x["candidate_first_sustained_off"]) for x in cases],
                  modes=MODES, noise_modes=NOISE_MODES, intervention_steps=INTERVENTION_STEPS,
                  horizon=HORIZON, source_sha256=digest(Path(__file__).read_bytes()),
                  weights_sha256={str(path): digest(path.read_bytes()) for path in paths},
                  original_test_cases_are_diagnostic_only=True)
    config_path = OUT / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise RuntimeError("Protocol changed; use a new output directory")
    else:
        write_json(config_path, config)
        (OUT / "source_diagnose_paper_local_actions.py").write_bytes(Path(__file__).read_bytes())
    completed = []
    total = len(cases) * len(NOISE_MODES) * len(MODES)
    for case in cases:
        seed, latent_seed = case["seed"], case["latent_seed"]
        trace_path = TRACES / f"track_{seed}_noise_{latent_seed}.json"
        expected = json.loads(trace_path.read_text(encoding="utf-8"))["candidate_trace"]
        for noise_mode in NOISE_MODES:
            prefixes = []
            for mode in MODES:
                name = f"track_{seed}_noise_{latent_seed}/{noise_mode}_{mode}"
                target = OUT / name / "result.json"
                if target.exists():
                    result = json.loads(target.read_text(encoding="utf-8"))
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    write_json(OUT / "status.json", dict(status="running", current=name,
                                                         completed=len(completed), total=total))
                    result = run_case(case, expected, mode, noise_mode, vision, memory, controller, device)
                    write_json(target, result)
                prefixes.append(result["prefix"])
                completed.append(name)
                print("BRANCH", len(completed), "/", total, name, flush=True)
            if prefixes[0] != prefixes[1] or prefixes[0] != prefixes[2]:
                raise RuntimeError(f"Interventions do not share identical prefix: {seed}")
    write_json(OUT / "status.json", dict(status="completed", completed=len(completed), total=total))
    print("COMPLETED", total, flush=True)


if __name__ == "__main__":
    main()
