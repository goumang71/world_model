"""Windows-safe CMA-ES controller training with frozen pretrained V/M.

Training, selection and final comparison use separate seed sets. Each run owns
its output directory and never writes to the downloaded checkpoints.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT / 'World-Model-2018-main'


def make_env():
    # Keep worker imports small; the Windows spawn children do not load torch.
    import gymnasium as gym
    return gym.make('CarRacing-v3', continuous=True, lap_complete_percent=1.0,
                    domain_randomize=False, max_episode_steps=1000)


class Evaluator:
    def __init__(self, count, vision, memory, device, steering_delta=2.0):
        from gymnasium.vector import AsyncVectorEnv
        self.envs = AsyncVectorEnv([make_env for _ in range(count)], shared_memory=True, context='spawn')
        self.count, self.vision, self.memory, self.device = count, vision, memory, device
        if not 0 < steering_delta <= 2.0:
            raise ValueError("steering_delta must be in (0, 2]")
        self.steering_delta = steering_delta

    def close(self):
        self.envs.close()

    def evaluate(self, solutions, seeds, max_steps=1000, latent_seeds=None):
        import numpy as np
        import torch
        import torch.nn.functional as F
        from models.controller import Controller
        assert len(solutions) == len(seeds) == self.count
        latent_seeds = seeds if latent_seeds is None else latent_seeds
        if len(latent_seeds) != self.count:
            raise ValueError("One latent seed is required per environment")
        controllers = []
        for params in solutions:
            model = Controller(288,3).to(self.device).eval()
            torch.nn.utils.vector_to_parameters(torch.as_tensor(params,device=self.device,dtype=torch.float32),model.parameters())
            controllers.append(model)
        # Independent per-episode streams: candidates on the same seed receive
        # identical latent noise, regardless of population size or completion.
        generators = [torch.Generator(device=self.device).manual_seed(int(seed)) for seed in latent_seeds]
        obs, _ = self.envs.reset(seed=[int(s) for s in seeds])
        hidden = self.memory.rnn.init_hidden(self.count,self.device)
        previous_steering = torch.zeros(self.count, device=self.device)
        done = np.zeros(self.count,dtype=bool)
        returns = np.zeros(self.count,dtype=np.float64)
        steps = np.zeros(self.count,dtype=int)
        with torch.inference_mode():
            for _ in range(max_steps):
                image = torch.as_tensor(obs.copy(),device=self.device,dtype=torch.float32).permute(0,3,1,2)/255
                image = F.interpolate(image[:,:,:-12,:],size=(96,96),mode='bicubic')
                if self.count == 1:
                    # Match the singleton evaluator's NCHW view strides.
                    image = image[0].unsqueeze(0)
                mu, logvar = self.vision.encode(image)
                noise = torch.stack([torch.randn(32,device=self.device,generator=g) for g in generators])
                z = mu + (0.5*logvar).exp()*noise
                x = torch.cat([z,hidden[0].squeeze(0)],dim=-1)
                actions = torch.cat([model(x[i:i+1]) for i,model in enumerate(controllers)],dim=0)
                actions[:,0] = torch.clamp(actions[:,0], min=previous_steering-self.steering_delta,
                                             max=previous_steering+self.steering_delta)
                previous_steering = actions[:,0].clone()
                if not torch.isfinite(actions).all():
                    raise ValueError('Non-finite controller actions')
                if getattr(self, 'trace_callback', None) is not None:
                    self.trace_callback(image,z,mu,hidden,actions,obs)
                obs,rewards,terminated,truncated,_ = self.envs.step(actions.cpu().numpy())
                returns += np.asarray(rewards)*(~done)
                steps += ~done
                done |= terminated | truncated
                _, hidden = self.memory.rnn(z[:,None],actions[:,None],hidden)
                if done.all(): break
        return returns,steps


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--generations',type=int,default=20)
    parser.add_argument('--population',type=int,default=16)
    parser.add_argument('--rollouts',type=int,default=3)
    parser.add_argument('--seed',type=int,default=20260927)
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--selection-every',type=int,default=5)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if min(args.generations,args.population,args.rollouts,args.steps,args.selection_every)<1 or args.population<6:
        parser.error('Positive budgets and population >= 6 required')
    if args.generations*args.rollouts>1000:
        parser.error('Training seeds would overlap selection seeds; choose a smaller budget or revise seed protocol.')
    sys.path.insert(0,str(REPO))
    import numpy as np
    import torch
    import cma
    from models.vae import VAE
    from models.mdnrnn import MDNRNN
    from models.controller import Controller
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output=(args.output_dir or ROOT/'outputs'/('controller_train_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    output.mkdir(parents=True,exist_ok=False)
    checkpoints=output/'checkpoints';checkpoints.mkdir()
    history_dir=output/'generations';history_dir.mkdir()
    source_hashes={}
    for name in ['vae','memory','controller']:
        p=REPO/'checkpoints'/f'{name}.pth'
        source_hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        if name!='controller': shutil.copy2(p,checkpoints/p.name)
    vision=VAE().to(device)
    memory=MDNRNN(32,3,256,5).to(device)
    for name,model in [('vae',vision),('memory',memory)]:
        model.load_state_dict(torch.load(checkpoints/f'{name}.pth',map_location=device,weights_only=True))
        model.eval().requires_grad_(False)
    initial=Controller(288,3).to(device)
    x0=torch.nn.utils.parameters_to_vector(initial.parameters()).detach().cpu().numpy().astype(np.float64)
    es=cma.CMAEvolutionStrategy(x0,0.1,{'popsize':args.population,'seed':args.seed,'verbose':-9})
    config=dict(generations=args.generations,population=args.population,rollouts=args.rollouts,
                steps=args.steps,seed=args.seed,sigma=0.1,initialization='random Controller, not pretrained',
                frozen=['vae','memory'],environment='CarRacing-v3',lap_complete_percent=1.0,
                training_seeds=list(range(1000,1000+args.generations*args.rollouts)),
                selection_seeds=list(range(2000,2005)),final_comparison_seeds=list(range(42,62)),
                selection_every=args.selection_every,source_sha256=source_hashes,torch=torch.__version__,
                cma=cma.__version__,device=str(device))
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('TRAINING_OUTPUT',output,flush=True)
    status=dict(status='running',generation=0,config=config,history=[])
    start=time.perf_counter()
    train_eval=selection_eval=None
    best_validation=-float('inf')

    def save_params(params,path):
        model=Controller(288,3)
        torch.nn.utils.vector_to_parameters(torch.as_tensor(params,dtype=torch.float32).clone(),model.parameters())
        torch.save(model.state_dict(),path)

    def save_status():
        status['elapsed_seconds']=time.perf_counter()-start
        tmp=output/'status.tmp'
        tmp.write_text(json.dumps(status,indent=2),encoding='utf-8')
        tmp.replace(output/'status.json')

    try:
        train_eval=Evaluator(args.population,vision,memory,device)
        selection_eval=Evaluator(5,vision,memory,device)
        for generation in range(1,args.generations+1):
            gen_start=time.perf_counter()
            solutions=es.ask()
            if len(solutions)!=args.population: raise RuntimeError('CMA population size changed unexpectedly')
            scores=[]
            seeds=list(range(1000+(generation-1)*args.rollouts,1000+generation*args.rollouts))
            for seed in seeds:
                values,_=train_eval.evaluate(solutions,[seed]*args.population,args.steps)
                scores.append(values)
            rewards=np.mean(scores,axis=0)
            es.tell(solutions,(-rewards).tolist())
            winner=solutions[int(np.argmax(rewards))]
            save_params(winner,history_dir/f'controller_gen{generation:03d}.pth')
            row=dict(generation=generation,training_seeds=seeds,mean_return=float(rewards.mean()),
                     best_return=float(rewards.max()),worst_return=float(rewards.min()),sigma=float(es.sigma))
            if generation%args.selection_every==0 or generation==args.generations:
                validation,_=selection_eval.evaluate([winner]*5,config['selection_seeds'],args.steps)
                row['selection_returns']=validation.tolist()
                row['selection_mean']=float(validation.mean())
                if row['selection_mean']>best_validation:
                    best_validation=row['selection_mean']
                    save_params(winner,checkpoints/'controller.pth')
                    status.update(selected_generation=generation,selection_mean=best_validation)
            row['seconds']=time.perf_counter()-gen_start
            status['history'].append(row)
            status['generation']=generation
            # Trusted, locally generated CMA state for recovery; not third-party pickle.
            temp=output/'cma_state.tmp';temp.write_bytes(es.pickle_dumps());temp.replace(output/'cma_state.pkl')
            save_status()
            print(json.dumps(row),flush=True)
        status['status']='completed'
    except BaseException as exc:
        status.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc))
        raise
    finally:
        if train_eval is not None:train_eval.close()
        if selection_eval is not None:selection_eval.close()
        status['source_weights_unchanged']=all(hashlib.sha256((REPO/'checkpoints'/f'{name}.pth').read_bytes()).hexdigest()==value
                                               for name,value in source_hashes.items())
        save_status()
        print('TRAINING_STATUS',status['status'],'elapsed_seconds',status['elapsed_seconds'],flush=True)


if __name__=='__main__':
    main()
