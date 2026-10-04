"""Small real-environment training smoke test; never overwrites shipped weights.

Extracts original training functions without executing notebook demos or full runs.
Spawn-based collection and num_workers=0 exercise the Windows-safe defaults.
This is a pipeline check, not a benchmark or a complete Notebook Run All test.
"""
import ast
import gc
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / 'World-Model-2018-main'


def definitions(path, names, scope):
    if path.suffix == '.ipynb':
        nb = json.loads(path.read_text(encoding='utf-8'))
        source = '\n'.join(
            line for cell in nb['cells'] if cell['cell_type'] == 'code'
            for line in ''.join(cell['source']).splitlines()
            if not line.lstrip().startswith(('%', '!'))
        )
    else:
        source = path.read_text(encoding='utf-8')
    nodes = [n for n in ast.parse(source).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names), (path, names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)


def main():
    import sys
    sys.path.insert(0, str(REPO))
    import cma
    import gymnasium as gym
    import h5py
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from gymnasium.vector import AsyncVectorEnv
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from tqdm import tqdm
    from models.vae import VAE
    from models.mdnrnn import MDNRNN, gaussian_nll_loss
    from models.controller import Controller
    from rollout_data import collect_data, merge_episode_files

    if not torch.cuda.is_available():
        raise RuntimeError('Original controller helpers require CUDA for this test.')
    torch.set_num_threads(1)
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device('cuda')
    out = ROOT / 'outputs' / ('training_smoke_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    originals = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (REPO / 'checkpoints').glob('*.pth')}
    report = dict(status='running', purpose='pipeline smoke test, not performance benchmark',
                  output=str(out), device=str(device), episodes=2, steps_per_episode=64,
                  vae_epochs=1, memory_epochs=1, cma_generations=2, requested_population=6,
                  cma_rollouts_per_candidate=1, cma_steps=64,
                  torch=torch.__version__, gymnasium=gym.__version__, cma=cma.__version__,
                  seed=123, collection_environment_seeded=True, collection_workers=2,
                  uses_pretrained_weights=False,
                  limitations=['No Notebook Run All; training DataLoader uses num_workers=0',
                               'Tiny data and training budget cannot establish driving quality'])
    start = time.perf_counter()
    previous = Path.cwd()
    os.chdir(out)
    print('Test output:', out, flush=True)
    scope = dict(torch=torch, np=np, gym=gym, h5py=h5py, os=os, gc=gc,
                 Image=Image, Dataset=Dataset, transforms=transforms, tqdm=tqdm,
                 F=F, device=device, cma=cma, pd=pd, time=time, Controller=Controller,
                 AsyncVectorEnv=AsyncVectorEnv, gaussian_nll_loss=gaussian_nll_loss,
                 LATENT_DIM=32, HIDDEN_DIM=256, ACTION_DIM=3, INITIAL_SIGMA=0.1)
    try:
        episodes = out / 'episodes'
        episodes.mkdir()
        print('1/5 Collecting with two Windows spawn workers, forced resets every 16 steps', flush=True)
        collect_data(num_episodes=2, max_steps=64, output_dir=episodes,
                     num_workers=2, seed=123, env_max_steps=16)
        data = out / 'smoke_data.h5'
        merge_episode_files(episodes, data, num_max_steps=64)
        with h5py.File(data, 'r') as f:
            assert f['dones'][:, 15::16].all()
            assert f['rewards'].dtype == np.dtype('float32')
            report['recorded_boundaries'] = int(f['dones'][:].sum())

        print('2/5 Training VAE with original Notebook training function', flush=True)
        definitions(REPO / '2-Vision (VAE).ipynb', ['CarRacingDataset', 'train_vae'], scope)
        vision = VAE().to(device)
        before = next(vision.parameters()).detach().clone()
        dataset = scope['CarRacingDataset'](str(data))
        loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
        scope['train_vae'](vision, loader, torch.optim.AdamW(vision.parameters(), lr=1e-3,
                           weight_decay=1e-6), num_epochs=1)
        assert not torch.equal(before, next(vision.parameters()))
        for p in vision.parameters():
            assert torch.isfinite(p).all()
        dataset.h5_file.close()
        dataset.h5_file = None
        # Reload saved artifacts at each stage, rather than only passing live models.
        vision = VAE().to(device)
        vision.load_state_dict(torch.load('checkpoints/vae.pth', weights_only=True))
        vision.eval()
        scope['vision'] = vision
        report['vae_parameters_updated'] = True

        print('3/5 Training Memory with original dataset and training function', flush=True)
        definitions(REPO / 'memory.py', ['CarRacingDataset', 'train_model'], scope)
        dataset = scope['CarRacingDataset'](str(data), vae=vision, device=device)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
        memory = MDNRNN(32, 3, 256, 5).to(device)
        before = next(memory.parameters()).detach().clone()
        losses = scope['train_model'](memory, loader, torch.optim.Adam(memory.parameters(), lr=1e-4),
                                      num_epochs=1, device=device, name='memory')
        assert np.isfinite(losses).all()
        assert not torch.equal(before, next(memory.parameters()))
        dataset.h5_file.close()
        dataset.h5_file = None
        memory = MDNRNN(32, 3, 256, 5).to(device)
        memory.load_state_dict(torch.load('checkpoints/memory.pth', weights_only=True))
        memory.eval()
        report.update(memory_parameters_updated=True, memory_nll=losses)

        print('4/5 Running original CMA-ES training and AsyncVectorEnv evaluation', flush=True)
        definitions(REPO / 'controller.py', ['make_env', 'create_vector_envs', 'encode_obs_batch',
                    'process_actions', 'load_weights', 'evaluate_policies', 'train_cma_es'], scope)
        es, metrics, best = scope['train_cma_es'](Controller, memory, max_generations=2,
                                               max_steps=64, popsize=6, checkpoint=1, rollouts=1)
        assert metrics['generation'] == [1, 2]
        assert np.isfinite(metrics['mean_reward']).all()
        report['cma_metrics'] = metrics
        controller = Controller(288).to(device)
        controller.load_state_dict(torch.load('checkpoints/controller.pth', weights_only=True))
        controller.eval()

        print('5/5 Reloaded policy evaluation: seed 9001, at most 200 steps', flush=True)
        env = gym.make('CarRacing-v3', lap_complete_percent=1.0, max_episode_steps=200)
        try:
            obs, _ = env.reset(seed=9001)
            torch.manual_seed(9001)
            hidden = memory.rnn.init_hidden(1, device)
            total = 0.0
            with torch.inference_mode():
                for step in range(200):
                    z = scope['encode_obs_batch'](obs[None])
                    action = controller(torch.cat([z, hidden[0].squeeze(0)], dim=-1))
                    assert torch.isfinite(action).all()
                    obs, reward, terminated, truncated, _ = env.step(action[0].cpu().numpy())
                    _, hidden = memory.rnn(z[:, None], action[:, None], hidden)
                    total += float(reward)
                    if terminated or truncated:
                        break
            report['evaluation'] = dict(seed=9001, episode_return=total, steps=step+1,
                                        terminated=bool(terminated), truncated=bool(truncated))
        finally:
            env.close()
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        os.chdir(previous)
        report['elapsed_seconds'] = time.perf_counter() - start
        report['original_weights_unchanged'] = all(
            hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
            for p, digest in originals.items())
        (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
