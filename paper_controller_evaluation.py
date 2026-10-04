"""Canonical singleton inference for the paper's direct CarRacing action map."""

def make_metrics_env():
    import gymnasium as gym
    class Metrics(gym.Wrapper):
        def step(self, action):
            obs,reward,terminated,truncated,info=self.env.step(action)
            info['fully_off_track']=not any(bool(w.tiles) for w in self.unwrapped.car.wheels)
            return obs,reward,terminated,truncated,info
    return Metrics(gym.make('CarRacing-v3',continuous=True,lap_complete_percent=1.0,
                            domain_randomize=False,max_episode_steps=1000))

class PaperCanonicalEvaluator:
    def __init__(self,count,vision,memory,device,steering_delta=1.0,control_z_mode='sample'):
        from gymnasium.vector import AsyncVectorEnv
        if control_z_mode not in ('sample','mean','mean_steer','mean_gas','mean_brake'):
            raise ValueError('Unsupported control_z_mode')
        self.envs=AsyncVectorEnv([make_metrics_env]*count,shared_memory=True,context='spawn')
        self.count=count;self.vision=vision;self.memory=memory;self.device=device;self.delta=steering_delta;self.control_z_mode=control_z_mode
    def close(self): self.envs.close()
    def evaluate(self,solutions,track_seeds,latent_seeds,max_steps=1000):
        import numpy as np
        import torch
        import torch.nn.functional as F
        from paper_controller import PaperController
        if not len(solutions)==len(track_seeds)==len(latent_seeds)==self.count:raise ValueError('Lane count mismatch')
        controllers=[]
        for params in solutions:
            c=PaperController(288,3).to(self.device).eval()
            torch.nn.utils.vector_to_parameters(torch.as_tensor(params,device=self.device,dtype=torch.float32).clone(),c.parameters())
            controllers.append(c)
        generators=[torch.Generator(device=self.device).manual_seed(int(s)) for s in latent_seeds]
        obs,_=self.envs.reset(seed=[int(s) for s in track_seeds])
        hidden=[self.memory.rnn.init_hidden(1,self.device) for _ in track_seeds]
        previous=[0.0]*self.count;done=np.zeros(self.count,dtype=bool)
        totals=np.zeros(self.count);steps=np.zeros(self.count,dtype=int);off=np.zeros(self.count,dtype=int)
        boundary=np.zeros(self.count,dtype=bool)
        with torch.inference_mode():
            for _ in range(max_steps):
                active=~done.copy();env_actions=np.zeros((self.count,3),dtype=np.float32);latents={};actions={}
                for i in np.flatnonzero(active):
                    # Preserve every singleton tensor shape and view used by evaluate_baseline.py.
                    image=torch.as_tensor(obs[i].copy(),device=self.device,dtype=torch.float32).permute(2,0,1).unsqueeze(0)/255.0
                    image=F.interpolate(image[:,:,:-12,:],size=(96,96),mode='bicubic')
                    mu,logvar=self.vision.encode(image)
                    std=torch.exp(.5*logvar)
                    eps=torch.randn(std.shape,device=self.device,generator=generators[i])
                    z=mu+eps*std
                    h=hidden[i][0].squeeze(0)
                    action=controllers[i](torch.cat([z,h],dim=-1)).clone()
                    if self.control_z_mode!='sample':
                        mean_action=controllers[i](torch.cat([mu,h],dim=-1))
                        if self.control_z_mode=='mean':
                            action=mean_action.clone()
                        else:
                            channel={'mean_steer':0,'mean_gas':1,'mean_brake':2}[self.control_z_mode]
                            action[:,channel]=mean_action[:,channel]
                    # Author's model.py issues the tanh steering command directly.
                    previous[i]=action[0,0].item()
                    if not torch.isfinite(action).all():raise ValueError('Non-finite action')
                    env_actions[i]=action[0].cpu().numpy();latents[i]=z;actions[i]=action
                obs,reward,terminated,truncated,info=self.envs.step(env_actions)
                totals+=np.asarray(reward)*active;steps+=active
                off+=np.asarray(info.get('fully_off_track',np.zeros(self.count,dtype=bool)),dtype=bool)&active
                boundary|=np.asarray(terminated)&~np.asarray(info.get('lap_finished',np.zeros(self.count,dtype=bool)),dtype=bool)&active
                for i in np.flatnonzero(active):
                    _,hidden[i]=self.memory.rnn(latents[i].unsqueeze(1),actions[i].unsqueeze(1),hidden[i])
                done|=terminated|truncated
                if done.all():break
        return [dict(seed=int(track_seeds[i]),latent_seed=int(latent_seeds[i]),episode_return=float(totals[i]),
                     steps=int(steps[i]),fully_off_track_steps=int(off[i]),fully_off_fraction=float(off[i]/steps[i]),
                     boundary_exit=bool(boundary[i]),severe_failure=bool(totals[i]<500 or off[i]/steps[i]>.2 or boundary[i])) for i in range(self.count)]

def summarize(rows):
    import numpy as np
    values=np.array([r['episode_return'] for r in rows])
    return dict(mean_return=float(values.mean()),severe_failures=sum(r['severe_failure'] for r in rows),
                fully_off_fraction=sum(r['fully_off_track_steps'] for r in rows)/sum(r['steps'] for r in rows),episodes=len(rows))

def should_replace(candidate,incumbent):
    """No new reward penalties: a conservative, prespecified validation gate."""
    if candidate['episodes']!=incumbent['episodes']:raise ValueError('Unequal validation budgets')
    return (candidate['mean_return']>incumbent['mean_return']+1e-6
            and candidate['severe_failures']<=incumbent['severe_failures']
            and candidate['fully_off_fraction']<=incumbent['fully_off_fraction']+1e-12)
