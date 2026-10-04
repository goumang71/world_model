"""Population test of using VAE posterior mean for C only, while RNN keeps sampled z."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "World-Model-2018-main"
PAPER = ROOT / "outputs" / "paper_controller_aligned_20261003" / "tested_checkpoints"
sys.path.insert(0, str(REPO))
from models.vae import VAE
from models.mdnrnn import MDNRNN
from models.controller import Controller
from controller_evaluation import CanonicalEvaluator
from paper_controller import PaperController
from paper_controller_evaluation import PaperCanonicalEvaluator, summarize, should_replace


def cases(split):
    if split == "dev":
        seeds, offset = range(180000, 180016), 140000001
    else:
        seeds, offset = range(190000, 190032), 150000001
    return [dict(seed=s, latent_seed=offset + s * 1009 + r * 1000003)
            for s in seeds for r in range(3)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    args = parser.parse_args()
    out = ROOT / "outputs" / f"paper_control_z_{args.split}_20261003"
    if args.split == "test":
        dev = json.loads((ROOT / "outputs" / "paper_control_z_dev_20261003" /
                          "comparison.json").read_text(encoding="utf-8"))
        if not dev["dev_gate_passed"]:
            raise RuntimeError("The candidate did not pass the fixed development gate")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision = VAE(3, 32).to(device).eval()
    memory = MDNRNN(32, 3, 256, 5).to(device).eval()
    original = Controller(288, 3).to(device).eval()
    paper = PaperController(288, 3).to(device).eval()
    paths = {"vae": REPO / "checkpoints" / "vae.pth",
             "memory": REPO / "checkpoints" / "memory.pth",
             "original_controller": REPO / "checkpoints" / "controller.pth",
             "paper_controller": PAPER / "controller.pth"}
    for model, path in ((vision, paths["vae"]), (memory, paths["memory"]),
                        (original, paths["original_controller"]),
                        (paper, paths["paper_controller"])):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    original_params = torch.nn.utils.parameters_to_vector(original.parameters()).detach().cpu().numpy()
    paper_params = torch.nn.utils.parameters_to_vector(paper.parameters()).detach().cpu().numpy()
    items = cases(args.split)
    config = dict(split=args.split, cases=items, episode_max_steps=1000,
                  modes=("original", "paper_sample", "paper_mean_control_only"),
                  frozen_models=True,
                  model_sha256={k:hashlib.sha256(path.read_bytes()).hexdigest()
                                for k,path in paths.items()},
                  development_gate="mean_return > original AND severe_failures <= original AND fully_off_fraction <= original",
                  final_gate="cluster_bootstrap_lower > 0 AND severe_failures <= original AND fully_off_fraction <= original",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    evaluators = {}
    try:
        evaluators["original"] = CanonicalEvaluator(8, vision, memory, device)
        evaluators["paper_sample"] = PaperCanonicalEvaluator(8, vision, memory, device)
        evaluators["paper_mean_control_only"] = PaperCanonicalEvaluator(
            8, vision, memory, device, control_z_mode="mean")
        results = {}
        for name in config["modes"]:
            evaluator = evaluators[name]
            params = original_params if name == "original" else paper_params
            rows = []
            for i in range(0, len(items), 8):
                chunk = items[i:i+8]
                rows.extend(evaluator.evaluate([params] * 8,
                                                [c["seed"] for c in chunk],
                                                [c["latent_seed"] for c in chunk]))
                (out / f"{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
                print(name, len(rows), "/", len(items), flush=True)
            results[name] = summarize(rows)
        baseline, candidate = results["original"], results["paper_mean_control_only"]
        gate = should_replace(candidate, baseline)
        old_rows = json.loads((out / "original.json").read_text(encoding="utf-8"))
        new_rows = json.loads((out / "paper_mean_control_only.json").read_text(encoding="utf-8"))
        track_count = 16 if args.split == "dev" else 32
        differences = np.array([b["episode_return"] - a["episode_return"]
                                for a,b in zip(old_rows,new_rows)]).reshape(track_count,3)
        rng = np.random.default_rng(20261003)
        bootstrap = rng.choice(differences.mean(axis=1), (20000,track_count),
                               replace=True).mean(axis=1)
        ci = np.percentile(bootstrap,[2.5,97.5]).tolist()
        final_gate = gate and ci[0] > 0 if args.split == "test" else None
        report = dict(split=args.split, summaries=results,
                      mean_difference_vs_original=float(differences.mean()),
                      cluster_bootstrap_95=ci,
                      improved_pairs=int((differences>1e-6).sum()),
                      worse_pairs=int((differences<-1e-6).sum()),
                      dev_gate_passed=gate if args.split=="dev" else None,
                      final_gate_passed=final_gate,
                      root_weights_unchanged=all(hashlib.sha256(path.read_bytes()).hexdigest()==config["model_sha256"][k]
                                                 for k,path in paths.items()))
        (out / "comparison.json").write_text(json.dumps(report,indent=2), encoding="utf-8")
        print("RESULT",json.dumps(report),flush=True)
    finally:
        for evaluator in evaluators.values():
            evaluator.close()


if __name__ == "__main__":
    main()
