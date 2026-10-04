"""Evaluate the downloaded BrunooCS pretrained baseline, without retraining."""
import argparse
import json
import hashlib
from datetime import datetime
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

ROOT = Path(__file__).resolve().parent
REPO = ROOT / 'World-Model-2018-main'
sys.path.insert(0, str(REPO))
from models.vae import VAE
from models.mdnrnn import MDNRNN
from models.controller import Controller
from driving_hud import DrivingHUD, driving_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--human', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+', help='Explicit evaluation seed list')
    parser.add_argument('--output-dir', type=Path, help='New output directory (must not exist)')
    parser.add_argument('--checkpoint-dir', type=Path, default=REPO / 'checkpoints')
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--latent-mode', choices=['sample', 'mean'], default='sample')
    parser.add_argument('--control-z-mode', choices=['mean_steer', 'sample'], default='mean_steer',
                        help='Use VAE posterior mean for steering only (validated default), or original sampled-z action')
    parser.add_argument('--record-trace', action='store_true', help='Save steering/state diagnostics without changing policy inputs')
    args = parser.parse_args()
    seeds = args.seeds if args.seeds is not None else list(range(args.seed, args.seed + args.episodes))
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error('Provide at least one episode and unique seeds.')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vision = VAE(3, 32).to(device)
    memory = MDNRNN(32, 3, 256, 5).to(device)
    controller = Controller(288, 3).to(device)
    checkpoints = {}
    for model, name in [(vision, 'vae'), (memory, 'memory'), (controller, 'controller')]:
        path = args.checkpoint_dir.resolve() / (name + '.pth')
        checkpoints[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        weights = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(weights)
        model.eval()
    output = args.output_dir or ROOT / 'outputs' / ('evaluation_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    print('output =', output, flush=True)
    env = gym.make('CarRacing-v3', render_mode='rgb_array' if args.human or not args.no_video else None, lap_complete_percent=1.0, domain_randomize=False, continuous=True, max_episode_steps=1000)
    print('device =', device, 'gymnasium =', gym.__version__, flush=True)
    rows = []
    writer = None
    hud = DrivingHUD(human=args.human) if args.human or not args.no_video else None
    user_closed = False
    try:
        with torch.inference_mode():
            for episode, seed in enumerate(seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)
                obs, _ = env.reset(seed=seed)
                hidden = memory.rnn.init_hidden(1, device)
                total = 0.0
                previous_steering = 0.0
                max_steering_change = 1.0
                off_track_steps = fully_off_track_steps = 0
                lap_finished = False
                trace = []
                probe_generator = torch.Generator(device=device).manual_seed(seed + 1000000)
                track = np.asarray(env.unwrapped.track)
                for step in range(1000):
                    image = torch.as_tensor(obs.copy(), device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                    image = F.interpolate(image[:, :, :-12, :], size=(96, 96), mode='bicubic')
                    mu, logvar = vision.encode(image)
                    z = vision.reparameterize(mu, logvar) if args.latent_mode == 'sample' else mu
                    action = controller(torch.cat([z, hidden[0].squeeze(0)], dim=-1))
                    action = action.clone()
                    if args.control_z_mode == 'mean_steer':
                        # Keep sampled z for gas, brake and the RNN; change only C's steering input.
                        mean_action = controller(torch.cat([mu, hidden[0].squeeze(0)], dim=-1))
                        action[:, 0] = mean_action[:, 0]
                 
                    action[:, 0] = torch.clamp(
                        action[:, 0],
                        min=previous_steering - max_steering_change,
                        max=previous_steering + max_steering_change,
                    )

                    previous_steering = action[0, 0].item()
                    if args.record_trace:
                        car = env.unwrapped.car
                        pos = np.asarray(car.hull.position)
                        nearest = int(np.argmin(np.sum((track[:, 2:4] - pos)**2, axis=1)))
                        angles = track[(np.arange(nearest-3, nearest+4) % len(track)), 1]
                        diffs = (np.diff(angles) + np.pi) % (2*np.pi) - np.pi
                        prior_metrics = driving_metrics(env)
                        h = hidden[0].squeeze(0)
                        weight = controller.model[0].weight[0]
                        mean_action = controller(torch.cat([mu, h], dim=-1))
                        row_trace = dict(step=step+1, action=action[0].cpu().tolist(),
                                         mean_z_action_same_history=mean_action[0].cpu().tolist(),
                                         speed=float(np.linalg.norm(np.asarray(car.hull.linearVelocity))),
                                         yaw_rate=float(car.hull.angularVelocity),
                                         position=pos.tolist(), road_wheels_before=prior_metrics['road_wheels'],
                                         straight_proxy=bool(np.abs(diffs).sum() < 0.1),
                                         raw_steering_z=float((z[0]*weight[:32]).sum()),
                                         raw_steering_h=float((h[0]*weight[32:]).sum()),
                                         raw_steering_bias=float(controller.model[0].bias[0]),
                                         raw_steering_sampling_std=float(((weight[:32]**2)*logvar[0].exp()).sum().sqrt()))
                        if step % 25 == 0:
                            # Independent RNG keeps the executed policy's sampling stream intact.
                            noise = torch.randn(64,32,device=device,generator=probe_generator)
                            alternatives = mu + (0.5*logvar).exp()*noise
                            alternatives = controller(torch.cat([alternatives,h.expand(64,-1)],dim=-1))
                            row_trace['fixed_state_action_std'] = alternatives.std(dim=0).cpu().tolist()
                    obs, reward, terminated, truncated, info = env.step(action[0].cpu().numpy())
                    _, hidden = memory.rnn(z.unsqueeze(1), action.unsqueeze(1), hidden)
                    total += float(reward)
                    metrics = driving_metrics(env)
                    off_track_steps += int(metrics['off_track'])
                    fully_off_track_steps += int(metrics['fully_off_track'])
                    lap_finished = bool(info.get('lap_finished', False))
                    if args.record_trace:
                        row_trace.update(road_wheels_after=metrics['road_wheels'],track_coverage=metrics['track_coverage'])
                        trace.append(row_trace)
                    if args.human or (episode == 0 and not args.no_video):
                        frame, user_closed = hud.frame(env.render(), step=step+1, seed=seed,
                                                       total=total, metrics=metrics, lap_finished=lap_finished,
                                                       terminated=terminated, truncated=truncated)
                    if episode == 0 and not args.human and not args.no_video:
                        import cv2
                        if writer is None:
                            writer = cv2.VideoWriter(str(output / f'baseline_seed{seed}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 50, (frame.shape[1], frame.shape[0]))
                            if not writer.isOpened():
                                raise RuntimeError('Cannot open MP4 writer')
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if terminated or truncated or user_closed:
                        break
                row = dict(seed=seed, episode_return=total, steps=step + 1, terminated=bool(terminated), truncated=bool(truncated),
                           track_coverage=metrics['track_coverage'], lap_finished=lap_finished,
                           off_track_steps=off_track_steps, fully_off_track_steps=fully_off_track_steps,
                           off_track_fraction=off_track_steps/(step+1), user_closed=user_closed)
                rows.append(row)
                if args.record_trace:
                    (output / f'trace_seed{seed}.json').write_text(json.dumps(trace), encoding='utf-8')
                (output / 'episodes.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
                print(row, flush=True)
                if user_closed:
                    break
        returns = np.array([r['episode_return'] for r in rows])
        report = dict(source='https://github.com/BrunooCS/World-Model-2018',
                      checkpoint_origin='downloaded pretrained' if args.checkpoint_dir.resolve() == (REPO / 'checkpoints').resolve() else 'custom directory',
                      environment='CarRacing-v3', lap_complete_percent=1.0, domain_randomize=False,
                      continuous=True, max_episode_steps=1000, seeds=[r['seed'] for r in rows], requested_seeds=seeds, checkpoints=checkpoints,
                      interrupted=user_closed, off_track_definition='At least one wheel has no road-tile contact; full off-track means zero wheels in contact',
                      lap_finished_count=sum(r['lap_finished'] for r in rows),
                      torch=torch.__version__, gymnasium=gym.__version__, numpy=np.__version__, device=str(device),
                      gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                      cudnn_deterministic=True, cudnn_benchmark=False, stochastic_z=args.latent_mode == 'sample',
                      latent_mode=args.latent_mode, control_z_mode=args.control_z_mode,
                      trace_recorded=args.record_trace,
                      episodes=rows, mean_return=float(returns.mean()),
                      std_return=float(returns.std(ddof=1)) if len(returns) > 1 else None,
                      median_return=float(np.median(returns)), min_return=float(returns.min()),
                      max_return=float(returns.max()), terminated_count=sum(r['terminated'] for r in rows),
                      truncated_count=sum(r['truncated'] for r in rows))
        (output / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('mean return =', report['mean_return'], flush=True)
    finally:
        env.close()
        if writer is not None:
            writer.release()
        if hud is not None:
            hud.close()


if __name__ == '__main__':
    main()
