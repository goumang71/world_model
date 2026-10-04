"""Trace every independent-test case where only the paper controller fails.

This is post-test diagnosis, not model selection. The saved independent test
tracks must never be reused as a holdout for a later tuned candidate.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "World-Model-2018-main"
RUN = ROOT / "outputs" / "paper_controller_aligned_20261003"
OUT = RUN / "failure_diagnosis"
sys.path.insert(0, str(REPO))
from models.vae import VAE
from models.mdnrnn import MDNRNN
from models.controller import Controller
from paper_controller import PaperController


def rollout(env, vision, memory, controller, track_seed, latent_seed, device, paper):
    obs, _ = env.reset(seed=track_seed)
    hidden = memory.rnn.init_hidden(1, device)
    generator = torch.Generator(device=device).manual_seed(latent_seed)
    previous_steering = 0.0
    total = 0.0
    rows = []
    with torch.inference_mode():
        for step in range(1000):
            image = torch.as_tensor(obs.copy(), device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            image = F.interpolate(image[:, :, :-12, :], size=(96, 96), mode="bicubic")
            mu, logvar = vision.encode(image)
            eps = torch.randn(logvar.shape, device=device, generator=generator)
            z = mu + eps * (0.5 * logvar).exp()
            action = controller(torch.cat([z, hidden[0].squeeze(0)], dim=-1)).clone()
            if not paper:
                action[:, 0] = torch.clamp(action[:, 0], min=previous_steering - 1.0,
                                            max=previous_steering + 1.0)
            previous_steering = action[0, 0].item()
            car = env.unwrapped.car
            speed = float(np.linalg.norm(np.asarray(car.hull.linearVelocity)))
            obs, reward, terminated, truncated, _ = env.step(action[0].cpu().numpy())
            _, hidden = memory.rnn(z.unsqueeze(1), action.unsqueeze(1), hidden)
            total += float(reward)
            off = not any(bool(w.tiles) for w in env.unwrapped.car.wheels)
            rows.append(dict(step=step + 1, steer=float(action[0, 0]), gas=float(action[0, 1]),
                             brake=float(action[0, 2]), speed=speed, fully_off=off))
            if terminated or truncated:
                break
    return total, rows


def first_sustained_off(rows, min_run=10, start_step=30):
    run = 0
    for row in rows:
        run = run + 1 if row["fully_off"] else 0
        if row["step"] >= start_step and run >= min_run:
            return row["step"] - run + 1
    return None


def window_stats(rows, end_step):
    window = [r for r in rows if end_step - 50 <= r["step"] < end_step]
    if not window:
        return None
    steer = np.array([r["steer"] for r in window])
    return dict(mean_abs_steer=float(np.abs(steer).mean()),
                large_steer_reversals=int(((steer[1:] * steer[:-1] < 0) &
                                           (np.abs(steer[1:]) > .5) &
                                           (np.abs(steer[:-1]) > .5)).sum()),
                mean_gas=float(np.mean([r["gas"] for r in window])),
                mean_brake=float(np.mean([r["brake"] for r in window])),
                mean_speed=float(np.mean([r["speed"] for r in window])),
                fully_off_steps=int(sum(r["fully_off"] for r in window)))


def main():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision = VAE(3, 32).to(device).eval()
    memory = MDNRNN(32, 3, 256, 5).to(device).eval()
    original = Controller(288, 3).to(device).eval()
    paper = PaperController(288, 3).to(device).eval()
    for model, path in ((vision, REPO / "checkpoints" / "vae.pth"),
                        (memory, REPO / "checkpoints" / "memory.pth"),
                        (original, REPO / "checkpoints" / "controller.pth"),
                        (paper, RUN / "tested_checkpoints" / "controller.pth")):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    comparison = json.loads((RUN / "comparison.json").read_text(encoding="utf-8"))
    selected = [(a, b) for a, b in zip(comparison["original_rows"], comparison["candidate_rows"])
                if b["severe_failure"] and not a["severe_failure"]]
    if len(selected) != 12:
        raise RuntimeError("The recorded set of candidate-only failures changed")
    OUT.mkdir(exist_ok=True)
    env = gym.make("CarRacing-v3", continuous=True, lap_complete_percent=1.0,
                   domain_randomize=False, max_episode_steps=1000)
    summaries = []
    try:
        for expected_original, expected_paper in selected:
            track_seed = expected_original["seed"]
            latent_seed = expected_original["latent_seed"]
            a_total, a_rows = rollout(env, vision, memory, original, track_seed, latent_seed, device, False)
            b_total, b_rows = rollout(env, vision, memory, paper, track_seed, latent_seed, device, True)
            if abs(a_total - expected_original["episode_return"]) > 1e-5:
                raise RuntimeError(f"Original trace parity failed: {track_seed}, {latent_seed}")
            if abs(b_total - expected_paper["episode_return"]) > 1e-5:
                raise RuntimeError(f"Paper trace parity failed: {track_seed}, {latent_seed}")
            first = first_sustained_off(b_rows)
            summary = dict(seed=track_seed, latent_seed=latent_seed, original_return=a_total,
                           candidate_return=b_total, candidate_first_sustained_off=first,
                           original_at_event=window_stats(a_rows, first) if first else None,
                           candidate_before_event=window_stats(b_rows, first) if first else None,
                           original_off_fraction=sum(r["fully_off"] for r in a_rows) / len(a_rows),
                           candidate_off_fraction=sum(r["fully_off"] for r in b_rows) / len(b_rows))
            summaries.append(summary)
            (OUT / f"track_{track_seed}_noise_{latent_seed}.json").write_text(
                json.dumps(dict(summary=summary, original_trace=a_rows, candidate_trace=b_rows)),
                encoding="utf-8")
            (OUT / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
            print("TRACE", len(summaries), "/", len(selected), track_seed, latent_seed,
                  round(a_total, 1), round(b_total, 1), flush=True)
    finally:
        env.close()
    events = [x for x in summaries if x["candidate_first_sustained_off"] is not None]
    def avg(key, name):
        return float(np.mean([x[name][key] for x in events]))
    result = dict(cases=len(summaries), distinct_tracks=len(set(x["seed"] for x in summaries)),
                  sustained_off_cases=len(events),
                  mean_original_gas_at_candidate_event=avg("mean_gas", "original_at_event"),
                  mean_candidate_gas_before_event=avg("mean_gas", "candidate_before_event"),
                  mean_original_brake_at_candidate_event=avg("mean_brake", "original_at_event"),
                  mean_candidate_brake_before_event=avg("mean_brake", "candidate_before_event"),
                  mean_original_abs_steer_at_candidate_event=avg("mean_abs_steer", "original_at_event"),
                  mean_candidate_abs_steer_before_event=avg("mean_abs_steer", "candidate_before_event"),
                  mean_original_speed_at_candidate_event=avg("mean_speed", "original_at_event"),
                  mean_candidate_speed_before_event=avg("mean_speed", "candidate_before_event"))
    (OUT / "aggregate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
