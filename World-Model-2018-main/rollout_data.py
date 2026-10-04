"""Importable, Windows-spawn-safe collection of fixed-length trajectory chunks.

Chunks may contain resets. `dones[t]` marks every terminal/truncated transition;
Memory training must split there and exclude the link to the next reset image.
"""
import argparse
import multiprocessing as mp
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
from PIL import Image


def random_action(t):
    if t < 20:
        return np.array([-0.1, 1, 0], dtype=np.float32)
    choices = [[0, np.random.random(), 0], [-np.random.random(), 0, 0],
               [np.random.random(), 0, 0], [0, 0, np.random.random()]]
    return np.asarray(choices[np.random.choice(4, p=[.35, .3, .3, .05])], dtype=np.float32)


def resize_obs(image, target_size=(96, 96)):
    return np.asarray(Image.fromarray(image[:-12]).resize(target_size, Image.Resampling.LANCZOS))


def collect_episode(env_name, max_steps, episode_id, worker_id, output_dir,
                    seed=None, env_max_steps=None):
    if max_steps < 2:
        raise ValueError('Collect at least two observations per chunk.')
    seed = 123 + worker_id * 100000 + episode_id if seed is None else seed
    np.random.seed(seed)
    kwargs = {} if env_max_steps is None else {'max_episode_steps': env_max_steps}
    env = gym.make(env_name, **kwargs)
    arrays = dict(images=[], actions=[], rewards=[], terminated=[], truncated=[])
    try:
        obs, _ = env.reset(seed=seed)
        local_step = 0
        for _ in range(max_steps):
            arrays['images'].append(resize_obs(obs))
            if local_step % 20 == 0:
                action = random_action(local_step)
            arrays['actions'].append(action.copy())
            obs, reward, terminated, truncated, _ = env.step(action)
            arrays['rewards'].append(reward)
            arrays['terminated'].append(terminated)
            arrays['truncated'].append(truncated)
            local_step += 1
            if terminated or truncated:
                obs, _ = env.reset()
                local_step = 0
    finally:
        env.close()
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'worker_{worker_id}_episode_{episode_id}.h5'
    with h5py.File(path, 'x') as f:
        for key, values in arrays.items():
            dtype = 'uint8' if key == 'images' else ('bool' if key in ('terminated', 'truncated') else 'float32')
            f.create_dataset(key, data=np.asarray(values, dtype=dtype))
        f.create_dataset('dones', data=np.logical_or(arrays['terminated'], arrays['truncated']))
        f.attrs.update(seed=seed, environment=env_name, format='fixed_length_chunk_with_boundaries')
    return str(path)


def _collect_job(args):
    return collect_episode(*args)


def collect_data(env_name='CarRacing-v3', num_episodes=128, max_steps=1000,
                 output_dir='temp_episodes_data', num_workers=2, seed=123, env_max_steps=None):
    if num_episodes < 1 or max_steps < 2 or num_workers < 1:
        raise ValueError('Positive episodes/workers and at least two steps are required.')
    jobs = [(env_name, max_steps, i, 0, str(Path(output_dir).resolve()), seed+i, env_max_steps)
            for i in range(num_episodes)]
    if num_workers == 1:
        return [_collect_job(job) for job in jobs]
    with mp.get_context('spawn').Pool(min(num_workers, num_episodes)) as pool:
        return pool.map(_collect_job, jobs)


def merge_episode_files(episodes_dir='temp_episodes_data', output_file='merged_data.h5',
                        env_name='CarRacing-v3', num_max_steps=1000):
    files = sorted(Path(episodes_dir).glob('*.h5'))
    if not files:
        raise ValueError('No trajectory files found.')
    keys = ('images', 'actions', 'rewards', 'dones', 'terminated', 'truncated')
    for path in files:
        with h5py.File(path, 'r') as f:
            if any(len(f[k]) != num_max_steps for k in keys):
                raise ValueError(f'Unexpected trajectory length: {path}')
    with h5py.File(output_file, 'x') as dst:
        dst.attrs.update(environment=env_name, format='fixed_length_chunks_with_boundaries')
        with h5py.File(files[0], 'r') as first:
            for key in keys:
                shape = (len(files),) + first[key].shape
                dst.create_dataset(key, shape=shape, dtype=first[key].dtype,
                                   chunks=(1, min(num_max_steps, 32)) + first[key].shape[1:],
                                   compression='lzf')
        for i, path in enumerate(files):
            with h5py.File(path, 'r') as src:
                for key in keys:
                    dst[key][i] = src[key][:]
        dst.create_dataset('seeds', data=[_file_seed(p) for p in files])
    return str(Path(output_file).resolve())


def _file_seed(path):
    with h5py.File(path, 'r') as f:
        return int(f.attrs['seed'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=128)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    chunks = output.parent / (output.stem + '_chunks')
    collect_data(num_episodes=args.episodes, max_steps=args.steps, num_workers=args.workers,
                 seed=args.seed, output_dir=chunks)
    print(merge_episode_files(chunks, output, num_max_steps=args.steps))
