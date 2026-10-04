"""Independent replication of the fixed mean-steering-only policy."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "World-Model-2018-main"
PAPER = ROOT / "outputs" / "paper_controller_aligned_20261003" / "tested_checkpoints"
DEV = ROOT / "outputs" / "paper_channel_ablation_dev_20261003"
OUT = ROOT / "outputs" / "paper_channel_ablation_replication_20261003"
sys.path.insert(0, str(REPO))
from models.vae import VAE
from models.mdnrnn import MDNRNN
from models.controller import Controller
from controller_evaluation import CanonicalEvaluator
from paper_controller import PaperController
from paper_controller_evaluation import PaperCanonicalEvaluator, summarize

MODES=("original","paper_sample","paper_mean_steer")
CASES=[dict(seed=s,latent_seed=190000001+s*1009+r*1000003)
       for s in range(220000,220032) for r in range(3)]


def main():
    dev=json.loads((DEV/"comparison.json").read_text(encoding="utf-8"))
    if dev["selected_for_new_holdout"]!="paper_mean_steer":
        raise RuntimeError("Unexpected development selection")
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v=VAE(3,32).to(device).eval()
    m=MDNRNN(32,3,256,5).to(device).eval()
    old=Controller(288,3).to(device).eval()
    paper=PaperController(288,3).to(device).eval()
    paths={"vae":REPO/"checkpoints"/"vae.pth",
           "memory":REPO/"checkpoints"/"memory.pth",
           "original_controller":REPO/"checkpoints"/"controller.pth",
           "paper_controller":PAPER/"controller.pth"}
    for model,path in ((v,paths["vae"]),(m,paths["memory"]),
                       (old,paths["original_controller"]),(paper,paths["paper_controller"])):
        model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
    old_params=torch.nn.utils.parameters_to_vector(old.parameters()).detach().cpu().numpy()
    paper_params=torch.nn.utils.parameters_to_vector(paper.parameters()).detach().cpu().numpy()
    config=dict(cases=CASES,modes=MODES,selected_from_dev=dev["selected_for_new_holdout"],
                holdout_used_for_selection=False,replication_of='paper_channel_ablation_holdout_20261003',
                weights_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    evaluators={}
    try:
        evaluators["original"]=CanonicalEvaluator(8,v,m,device)
        evaluators["paper_sample"]=PaperCanonicalEvaluator(8,v,m,device)
        evaluators["paper_mean_steer"]=PaperCanonicalEvaluator(8,v,m,device,control_z_mode="mean_steer")
        stats={}
        for mode in MODES:
            e=evaluators[mode]
            params=old_params if mode=="original" else paper_params
            rows=[]
            for i in range(0,len(CASES),8):
                chunk=CASES[i:i+8]
                rows.extend(e.evaluate([params]*8,[x["seed"] for x in chunk],
                                       [x["latent_seed"] for x in chunk]))
                (OUT/f"{mode}.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
                print(mode,len(rows),"/",len(CASES),flush=True)
            stats[mode]=summarize(rows)
        rows_old=json.loads((OUT/"original.json").read_text(encoding="utf-8"))
        rows_new=json.loads((OUT/"paper_mean_steer.json").read_text(encoding="utf-8"))
        diff=np.array([b["episode_return"]-a["episode_return"]
                       for a,b in zip(rows_old,rows_new)]).reshape(32,3)
        rng=np.random.default_rng(20261003)
        boot=rng.choice(diff.mean(1),(20000,32),replace=True).mean(1)
        ci=np.percentile(boot,[2.5,97.5]).tolist()
        a,b=stats["original"],stats["paper_mean_steer"]
        gate=(b["mean_return"]>a["mean_return"] and
              b["severe_failures"]<=a["severe_failures"] and
              b["fully_off_fraction"]<=a["fully_off_fraction"] and ci[0]>0)
        result=dict(summaries=stats,mean_difference_vs_original=float(diff.mean()),
                    cluster_bootstrap_95=ci,improved_pairs=int((diff>1e-6).sum()),
                    worse_pairs=int((diff<-1e-6).sum()),
                    final_gate_passed=bool(gate),
                    root_weights_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==config["weights_sha256"][k]
                                               for k,p in paths.items()))
        (OUT/"comparison.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        print("RESULT",json.dumps(result),flush=True)
    finally:
        for e in evaluators.values():
            e.close()


if __name__=="__main__":
    main()
