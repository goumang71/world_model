"""Inspect the first-state latent and raw controller output without changing weights."""
import gymnasium as gym
import torch
import torch.nn.functional as F
from evaluate_flydexo import load_models

torch.set_num_threads(1)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae,rnn,controller=load_models(device)
env=gym.make("CarRacing-v3",continuous=True,lap_complete_percent=1.0,
             domain_randomize=False,max_episode_steps=1000)
try:
    obs,_=env.reset(seed=42)
    with torch.inference_mode():
        for crop in ("crop12","full"):
            image=torch.as_tensor(obs.copy(),device=device,dtype=torch.float32).permute(2,0,1).unsqueeze(0)/255
            if crop=="crop12":image=image[:,:,:-12,:]
            image=F.interpolate(image,size=(64,64),mode="bilinear",align_corners=False)
            for encoding, candidate in (("rgb01",image),("bgr01",image[:,[2,1,0]]),
                                        ("rgb_minus1",image*2-1)):
                torch.manual_seed(42)
                z,mu,log_sigma=vae.encode(candidate)
                h=torch.zeros((1,256),device=device)
                raw=controller(torch.cat([z,h],dim=-1))
                raw_mean=controller(torch.cat([mu,h],dim=-1))
                raw_hz=controller(torch.cat([h,z],dim=-1))
                print(crop,encoding,"mu mean/std/range",float(mu.mean()),float(mu.std()),float(mu.min()),float(mu.max()),
                      "posterior std",float(log_sigma.exp().mean()),"raw",raw[0].tolist(),"raw_mu",raw_mean[0].tolist(),"raw_hz",raw_hz[0].tolist())
finally:
    env.close()
