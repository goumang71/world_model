"""Isolate which action channel causes the posterior-mean policy failure."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "World-Model-2018-main"
PAPER = ROOT / "outputs" / "paper_controller_aligned_20261003" / "tested_checkpoints"
OUT = ROOT / "outputs" / "paper_channel_ablation_dev_20261003"
sys.path.insert(0, str(REPO))
from models.vae import VAE
from models.mdnrnn import MDNRNN
from models.controller import Controller
from controller_evaluation import CanonicalEvaluator
from paper_controller import PaperController
from paper_controller_evaluation import PaperCanonicalEvaluator, summarize, should_replace

MODES = ("original", "paper_sample", "paper_mean_steer",
         "paper_mean_gas", "paper_mean_brake")
CASES = [dict(seed=s, latent_seed=170000001+s*1009+r*1000003)
         for s in range(200000,200016) for r in range(3)]


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision = VAE(3,32).to(device).eval()
    memory = MDNRNN(32,3,256,5).to(device).eval()
    original = Controller(288,3).to(device).eval()
    paper = PaperController(288,3).to(device).eval()
    paths = {"vae":REPO/"checkpoints"/"vae.pth",
             "memory":REPO/"checkpoints"/"memory.pth",
             "original_controller":REPO/"checkpoints"/"controller.pth",
             "paper_controller":PAPER/"controller.pth"}
    for model,path in ((vision,paths["vae"]),(memory,paths["memory"]),
                       (original,paths["original_controller"]),
                       (paper,paths["paper_controller"])):
        model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
    params_original=torch.nn.utils.parameters_to_vector(original.parameters()).detach().cpu().numpy()
    params_paper=torch.nn.utils.parameters_to_vector(paper.parameters()).detach().cpu().numpy()
    config=dict(cases=CASES,modes=MODES,steps=1000,
                protocol="Only one C action channel receives mu; remaining C channels use sampled z; RNN always receives sampled z and actual action",
                weights_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    evaluators={}
    try:
        evaluators["original"]=CanonicalEvaluator(8,vision,memory,device)
        for mode in MODES[1:]:
            control_mode=mode.removeprefix("paper_")
            evaluators[mode]=PaperCanonicalEvaluator(8,vision,memory,device,
                                                      control_z_mode=control_mode)
        summaries={}
        for mode in MODES:
            e=evaluators[mode]
            params=params_original if mode=="original" else params_paper
            rows=[]
            for i in range(0,len(CASES),8):
                chunk=CASES[i:i+8]
                rows.extend(e.evaluate([params]*8,[x["seed"] for x in chunk],
                                       [x["latent_seed"] for x in chunk]))
                (OUT/f"{mode}.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
                print(mode,len(rows),"/",len(CASES),flush=True)
            summaries[mode]=summarize(rows)
        gates={mode:should_replace(summaries[mode],summaries["original"])
               for mode in MODES[2:]}
        eligible=[mode for mode in MODES[2:] if gates[mode]]
        selected=max(eligible,key=lambda m:summaries[m]["mean_return"]) if eligible else None
        result=dict(summaries=summaries,development_gates=gates,
                    selected_for_new_holdout=selected,
                    root_weights_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==config["weights_sha256"][k]
                                               for k,p in paths.items()))
        (OUT/"comparison.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        print("RESULT",json.dumps(result),flush=True)
    finally:
        for e in evaluators.values():
            e.close()


if __name__=="__main__":
    main()
