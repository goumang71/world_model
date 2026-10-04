"""Isolated evaluation of flydexo's public CarRacing-v3 World Models weights."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "external_models" / "flydexo_world_models_carracing_v3"


def load_models(device):
    spec = importlib.util.spec_from_file_location("flydexo_model", MODEL / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = yaml.safe_load((MODEL / "config.yaml").read_text(encoding="utf-8"))
    cfg = SimpleNamespace(**{key: SimpleNamespace(**value) for key,value in config.items()})
    vae = module.AutoEncoder(cfg).to(device).eval()
    rnn = module.RNN(cfg).to(device).eval()
    controller = torch.nn.Linear(288,3).to(device).eval()
    vae.load_state_dict(torch.load(MODEL / "vae.pt",map_location=device,weights_only=True))
    rnn.load_state_dict(torch.load(MODEL / "rnn.pt",map_location=device,weights_only=True))
    state=torch.load(MODEL / "controller.pt",map_location=device,weights_only=True)
    controller.load_state_dict({"weight":state["layer.weight"],"bias":state["layer.bias"]})
    return vae,rnn,controller


def action_from_raw(raw, mapping):
    if mapping == "paper":
        activated = torch.tanh(raw)
        return torch.cat([activated[:,0:1],(activated[:,1:2]+1)/2,
                          activated[:,2:3].clamp(0,1)],dim=-1)
    if mapping == "sigmoid":
        return torch.cat([raw[:,0:1].tanh(),raw[:,1:2].sigmoid(),
                          raw[:,2:3].sigmoid()],dim=-1)
    if mapping == "raw_clip":
        return torch.cat([raw[:,0:1].clamp(-1,1),raw[:,1:2].clamp(0,1),
                          raw[:,2:3].clamp(0,1)],dim=-1)
    raise ValueError(mapping)


def run_episode(env,models,seed,device,crop,mapping,latent_mode,max_steps):
    vae,rnn,controller=models
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs,_=env.reset(seed=seed)
    hidden=None
    total=0.0
    off=0
    actions=[]
    with torch.inference_mode():
        for step in range(max_steps):
            image=torch.as_tensor(obs.copy(),device=device,dtype=torch.float32).permute(2,0,1).unsqueeze(0)/255
            if crop == "crop12":
                image=image[:,:,:-12,:]
            image=F.interpolate(image,size=(64,64),mode="bilinear",align_corners=False)
            z,mu,_=vae.encode(image)
            if latent_mode == "mean":
                z=mu
            h=torch.zeros((1,256),device=device) if hidden is None else hidden[0][-1]
            raw=controller(torch.cat([z,h],dim=-1))
            action=action_from_raw(raw,mapping)
            arr=action[0].cpu().numpy()
            obs,reward,terminated,truncated,_=env.step(arr)
            _,_,_,hidden,_=rnn(z.unsqueeze(1),action.unsqueeze(1),hidden)
            total+=float(reward)
            off+=int(not any(bool(w.tiles) for w in env.unwrapped.car.wheels))
            actions.append(arr.tolist())
            if terminated or truncated:
                break
    actions=np.asarray(actions)
    return dict(seed=int(seed),episode_return=float(total),steps=step+1,
                terminated=bool(terminated),truncated=bool(truncated),
                fully_off_track_steps=int(off),fully_off_fraction=float(off/(step+1)),
                mean_steer=float(actions[:,0].mean()),mean_gas=float(actions[:,1].mean()),
                mean_brake=float(actions[:,2].mean()),
                severe_failure=bool(total<500 or off/(step+1)>.2 or (terminated and total<900)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",nargs="+",type=int,required=True)
    parser.add_argument("--crop",choices=("full","crop12"),default="crop12")
    parser.add_argument("--mapping",choices=("paper","sigmoid","raw_clip"),default="paper")
    parser.add_argument("--latent-mode",choices=("sample","mean"),default="sample")
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--max-steps",type=int,default=1000)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models=load_models(device)
    config=dict(source="https://huggingface.co/flydexo/world-models-carracing-v3",
                revision=json.loads((MODEL/"provenance.json").read_text())["revision"],
                model_sha256={name:hashlib.sha256((MODEL/name).read_bytes()).hexdigest()
                              for name in ("vae.pt","rnn.pt","controller.pt","model.py","config.yaml")},
                crop=args.crop,mapping=args.mapping,latent_mode=args.latent_mode,
                seeds=args.seeds,max_steps=args.max_steps,device=str(device),
                environment="CarRacing-v3",lap_complete_percent=1.0)
    (args.output_dir/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    env=gym.make("CarRacing-v3",continuous=True,lap_complete_percent=1.0,
                 domain_randomize=False,max_episode_steps=args.max_steps)
    rows=[]
    try:
        for seed in args.seeds:
            row=run_episode(env,models,seed,device,args.crop,args.mapping,args.latent_mode,args.max_steps)
            rows.append(row)
            (args.output_dir/"episodes.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
            print(row,flush=True)
    finally:
        env.close()
    summary=dict(mean_return=float(np.mean([r["episode_return"] for r in rows])),
                 severe_failures=int(sum(r["severe_failure"] for r in rows)),
                 fully_off_fraction=float(sum(r["fully_off_track_steps"] for r in rows)/sum(r["steps"] for r in rows)),
                 episodes=len(rows))
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("SUMMARY",summary,flush=True)


if __name__=="__main__":
    main()
