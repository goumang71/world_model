"""Prospective paired test of posterior-mean steering on the original controller."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "World-Model-2018-main"
OUT = ROOT / "outputs" / "original_mean_steer_20261003"
sys.path.insert(0, str(REPO))
from models.vae import VAE
from models.mdnrnn import MDNRNN
from models.controller import Controller
from controller_evaluation import CanonicalEvaluator, summarize, should_replace


def cases(first, count, offset):
    return [dict(seed=s, latent_seed=offset+s*1009+r*1000003)
            for s in range(first, first+count) for r in range(3)]


def cluster_ci(base, candidate, seed=260003, draws=10000):
    tracks = sorted(set(x["seed"] for x in base))
    by_track = []
    for track in tracks:
        differences = [y["episode_return"]-x["episode_return"]
                       for x,y in zip(base,candidate) if x["seed"] == track]
        by_track.append(np.mean(differences))
    rng = np.random.default_rng(seed)
    a = np.asarray(by_track)
    means = a[rng.integers(0,len(a),(draws,len(a)))].mean(axis=1)
    return [float(v) for v in np.quantile(means,[.025,.975])]


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision=VAE(3,32).to(device).eval()
    memory=MDNRNN(32,3,256,5).to(device).eval()
    original=Controller(288,3).to(device).eval()
    paths={"vae":REPO/"checkpoints"/"vae.pth",
           "memory":REPO/"checkpoints"/"memory.pth",
           "controller":REPO/"checkpoints"/"controller.pth"}
    for model,path in ((vision,paths["vae"]),(memory,paths["memory"]),(original,paths["controller"])):
        model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
    params=torch.nn.utils.parameters_to_vector(original.parameters()).detach().cpu().numpy()
    splits={"development":cases(230000,16,210000001),
            "holdout":cases(240000,32,230000001)}
    config=dict(splits=splits,protocol="Only steering receives posterior mean; gas/brake and RNN z remain sampled. Original checkpoints frozen.",
                weights_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    evaluators={}
    try:
        evaluators={"sample":CanonicalEvaluator(8,vision,memory,device),
                    "mean_steer":CanonicalEvaluator(8,vision,memory,device,control_z_mode="mean_steer")}
        for split,entries in splits.items():
            rows={}
            for mode in ("sample","mean_steer"):
                records=[]
                for start in range(0,len(entries),8):
                    chunk=entries[start:start+8]
                    records.extend(evaluators[mode].evaluate([params]*8,
                        [x["seed"] for x in chunk],[x["latent_seed"] for x in chunk]))
                    (OUT/f"{split}_{mode}.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
                    print(split,mode,len(records),"/",len(entries),flush=True)
                rows[mode]=records
            summary={k:summarize(v) for k,v in rows.items()}
            diff=np.array([b["episode_return"]-a["episode_return"] for a,b in zip(rows["sample"],rows["mean_steer"])])
            report=dict(summaries=summary,mean_difference=float(diff.mean()),
                        track_cluster_bootstrap_95=cluster_ci(rows["sample"],rows["mean_steer"]),
                        improved_pairs=int(np.sum(diff>0)),worse_pairs=int(np.sum(diff<0)),
                        basic_gate_passed=should_replace(summary["mean_steer"],summary["sample"]),
                        root_weights_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==config["weights_sha256"][k]
                                                   for k,p in paths.items()))
            report["reliable_gate_passed"]=(report["basic_gate_passed"] and report["track_cluster_bootstrap_95"][0]>0)
            (OUT/f"{split}_comparison.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
            print("RESULT",split,json.dumps(report),flush=True)
            if split=="development" and not report["basic_gate_passed"]:
                print("Development gate failed; holdout remains untouched",flush=True)
                break
    finally:
        for evaluator in evaluators.values():
            evaluator.close()


if __name__=="__main__":
    main()
