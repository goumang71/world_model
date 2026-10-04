"""Frozen VAE/RNN and latent-to-steer diagnostics on matched paper-policy runs."""
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
OUT = RUN / "world_model_failure_probe"
sys.path.insert(0, str(REPO))
from models.vae import VAE
from models.mdnrnn import MDNRNN
from paper_controller import PaperController


def run_episode(env, vision, memory, controller, seed, latent_seed, end_step, device):
    obs, _ = env.reset(seed=seed)
    hidden = memory.rnn.init_hidden(1, device)
    generator = torch.Generator(device=device).manual_seed(latent_seed)
    weight = controller.model[0].weight[0, :32]
    total = 0.0
    pending = None
    rows = []
    with torch.inference_mode():
        for step in range(1, 1001):
            image = torch.as_tensor(obs.copy(), device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            image = F.interpolate(image[:, :, :-12, :], size=(96, 96), mode="bicubic")
            mu, logvar = vision.encode(image)
            eps = torch.randn(logvar.shape, device=device, generator=generator)
            z = mu + eps * (.5 * logvar).exp()
            if pending is not None and end_step - 50 <= pending["step"] < end_step:
                pi, pred_mu, pred_sigma = pending["pi"], pending["mu"], pending["sigma"]
                log_prob = torch.distributions.Normal(pred_mu, pred_sigma).log_prob(z[0]).sum(-1)
                nll = -torch.logsumexp(pi.log() + log_prob, dim=-1) / 32
                predicted_mean = (pi[:, None] * pred_mu).sum(0)
                pending["row"]["mdn_nll_per_dim"] = float(nll)
                pending["row"]["mdn_mean_mse"] = float(((predicted_mean - mu[0])**2).mean())
                pending["row"]["persistence_mean_mse"] = float(((pending["current_mu"] - mu[0])**2).mean())
            x = torch.cat([z, hidden[0].squeeze(0)], dim=-1)
            action = controller(x).clone()
            if end_step - 50 <= step < end_step:
                recon = vision.decode(mu)
                raw_mean = controller.model[0](torch.cat([mu, hidden[0].squeeze(0)], dim=-1))[0, 0]
                noise_std = ((weight.square() * logvar[0].exp()).sum()).sqrt()
                row = dict(step=step,
                           vae_recon_mse=float(((recon - image)**2).mean()),
                           vae_posterior_std=float((.5 * logvar).exp().mean()),
                           raw_steering_mean=float(raw_mean),
                           raw_steering_noise_std=float(noise_std),
                           action_steer=float(action[0, 0]),
                           action_gas=float(action[0, 1]),
                           action_brake=float(action[0, 2]))
                rows.append(row)
            else:
                row = None
            obs, reward, terminated, truncated, _ = env.step(action[0].cpu().numpy())
            outs, hidden = memory.rnn(z.unsqueeze(1), action.unsqueeze(1), hidden)
            pi, next_mu, sigma = memory.mdn(outs, tau=1.0)
            pending = dict(step=step, pi=pi[0, 0], mu=next_mu[0, 0],
                           sigma=sigma[0, 0], current_mu=mu[0], row=row)
            total += float(reward)
            if terminated or truncated:
                break
    return total, rows


def main():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision = VAE(3, 32).to(device).eval()
    memory = MDNRNN(32, 3, 256, 5).to(device).eval()
    controller = PaperController(288, 3).to(device).eval()
    weights = ((vision, REPO / "checkpoints" / "vae.pth"),
               (memory, REPO / "checkpoints" / "memory.pth"),
               (controller, RUN / "tested_checkpoints" / "controller.pth"))
    for model, path in weights:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    comparison = json.loads((RUN / "comparison.json").read_text(encoding="utf-8"))
    by_case = {(r["seed"], r["latent_seed"]): r for r in comparison["candidate_rows"]}
    matches = json.loads((RUN / "failure_diagnosis" / "matched_success_comparison.json").read_text(encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    env = gym.make("CarRacing-v3", continuous=True, lap_complete_percent=1.0,
                   domain_randomize=False, max_episode_steps=1000)
    records = []
    try:
        for entry in matches:
            seed = entry["seed"]
            end_step = entry["event_step"]
            pair = {}
            for label, latent_seed in (("failure", entry["failed_latent_seed"]),
                                       ("same_track_success", entry["success_latent_seed"])):
                total, rows = run_episode(env, vision, memory, controller, seed,
                                          latent_seed, end_step, device)
                expected = by_case[(seed, latent_seed)]["episode_return"]
                if abs(total - expected) > 1e-5:
                    raise RuntimeError(f"Score parity failed {seed}/{latent_seed}: {total} vs {expected}")
                if len(rows) != 50 or not all("mdn_nll_per_dim" in r for r in rows):
                    raise RuntimeError(f"Incomplete pre-event diagnostics {seed}/{latent_seed}")
                pair[label] = dict(latent_seed=latent_seed, episode_return=total,
                                   metrics={key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key != "step"},
                                   rows=rows)
            records.append(dict(seed=seed, end_step=end_step, **pair))
            (OUT / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            print("MODEL_PAIR", len(records), "/", len(matches), seed, end_step, flush=True)
    finally:
        env.close()
    print("COMPLETED", len(records), flush=True)


if __name__ == "__main__":
    main()
