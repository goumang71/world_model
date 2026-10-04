"""100 fresh tracks using the validated per-car parallel evaluator."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
REPO=ROOT/"World-Model-2018-main"
OUT=ROOT/"outputs"/"current_default_100_parallel_20261004"
sys.path.insert(0,str(REPO))
from models.vae import VAE
from models.mdnrnn import MDNRNN
from models.controller import Controller
from controller_evaluation import CanonicalEvaluator, summarize


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae=VAE(3,32).to(device).eval()
    memory=MDNRNN(32,3,256,5).to(device).eval()
    controller=Controller(288,3).to(device).eval()
    weights={"vae":REPO/"checkpoints"/"vae.pth",
             "memory":REPO/"checkpoints"/"memory.pth",
             "controller":REPO/"checkpoints"/"controller.pth"}
    for name,model in (("vae",vae),("memory",memory),("controller",controller)):
        model.load_state_dict(torch.load(weights[name],map_location=device,weights_only=True))
    params=torch.nn.utils.parameters_to_vector(controller.parameters()).detach().cpu().numpy()
    seeds=list(range(260000,260100))
    config=dict(seeds=seeds,latent_seeds=seeds,control_z_mode="mean_steer",
                latent_mode="sample",steps=1000,environment="CarRacing-v3",
                weights_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest()
                                for name,path in weights.items()})
    (OUT/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    rows=[]
    evaluator8=CanonicalEvaluator(8,vae,memory,device,control_z_mode="mean_steer")
    try:
        for start in range(0,96,8):
            chunk=seeds[start:start+8]
            rows.extend(evaluator8.evaluate([params]*8,chunk,chunk))
            (OUT/"episodes.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
            print("completed",len(rows),"/100",flush=True)
            if start==0:
                prior=json.loads((ROOT/"outputs"/"current_default_100_20261004"/"episodes.json").read_text(encoding="utf-8"))
                for earlier,current in zip(prior,rows):
                    if earlier["seed"]!=current["seed"] or abs(earlier["episode_return"]-current["episode_return"])>1e-5:
                        raise RuntimeError(f"User-facing evaluator parity failed: {earlier} vs {current}")
                print("user_facing_parity_passed",len(prior),flush=True)
    finally:
        evaluator8.close()
    evaluator4=CanonicalEvaluator(4,vae,memory,device,control_z_mode="mean_steer")
    try:
        chunk=seeds[96:]
        rows.extend(evaluator4.evaluate([params]*4,chunk,chunk))
        (OUT/"episodes.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
        print("completed",len(rows),"/100",flush=True)
    finally:
        evaluator4.close()
    result=summarize(rows)
    result["median_return"]=float(np.median([r["episode_return"] for r in rows]))
    result["min_return"]=float(min(r["episode_return"] for r in rows))
    result["max_return"]=float(max(r["episode_return"] for r in rows))
    result["user_facing_parity_passed"]=True
    result["root_weights_unchanged"]=all(hashlib.sha256(path.read_bytes()).hexdigest()==config["weights_sha256"][name]
                                         for name,path in weights.items())
    (OUT/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("RESULT",json.dumps(result),flush=True)


if __name__=="__main__":
    main()
