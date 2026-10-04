"""Summarize paired local-action branches without selecting a default policy."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "paper_controller_aligned_20261003" / "local_action_branches"
MODES = ("baseline", "gas_x08", "steer_slew_1")
NOISES = ("original_stream", "new_stream_1", "new_stream_2")


def main():
    config = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    status = json.loads((OUT / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "completed":
        raise RuntimeError("Wait for all branches before analysis")
    grouped = defaultdict(list)
    for seed, latent_seed, _ in config["cases"]:
        for noise in NOISES:
            path = OUT / f"track_{seed}_noise_{latent_seed}"
            cases = {mode: json.loads((path / f"{noise}_{mode}" / "result.json").read_text(encoding="utf-8"))
                     for mode in MODES}
            prefixes = [cases[m]["prefix"] for m in MODES]
            if prefixes[0] != prefixes[1] or prefixes[0] != prefixes[2]:
                raise RuntimeError("Prefix equality check failed")
            baseline = cases["baseline"]["summary"]
            for mode in MODES[1:]:
                candidate = cases[mode]["summary"]
                grouped[mode].append(dict(seed=seed, latent_seed=latent_seed, noise=noise,
                                          off_delta=candidate["fully_off_steps"] - baseline["fully_off_steps"],
                                          reward_delta=candidate["reward_sum"] - baseline["reward_sum"],
                                          tiles_delta=candidate["tiles_gained"] - baseline["tiles_gained"],
                                          baseline_off=baseline["fully_off_steps"],
                                          candidate_off=candidate["fully_off_steps"],
                                          baseline_first=baseline["first_sustained_off_step"],
                                          candidate_first=candidate["first_sustained_off_step"],
                                          baseline_reward=baseline["reward_sum"],
                                          candidate_reward=candidate["reward_sum"]))
    rng = np.random.default_rng(20261003)
    reports = {}
    for mode, rows in grouped.items():
        tracks = sorted(set(x["seed"] for x in rows))
        by_track = np.array([[np.mean([r[key] for r in rows if r["seed"] == seed])
                              for key in ("off_delta", "reward_delta", "tiles_delta")]
                             for seed in tracks])
        boot = rng.choice(by_track[:, 0], (20000, len(tracks)), replace=True).mean(1)
        reports[mode] = dict(branches=len(rows), tracks=len(tracks),
                             off_delta_mean=float(np.mean([x["off_delta"] for x in rows])),
                             reward_delta_mean=float(np.mean([x["reward_delta"] for x in rows])),
                             tiles_delta_mean=float(np.mean([x["tiles_delta"] for x in rows])),
                             off_better=int(sum(x["off_delta"] < 0 for x in rows)),
                             off_worse=int(sum(x["off_delta"] > 0 for x in rows)),
                             off_tied=int(sum(x["off_delta"] == 0 for x in rows)),
                             reward_better=int(sum(x["reward_delta"] > 0 for x in rows)),
                             reward_worse=int(sum(x["reward_delta"] < 0 for x in rows)),
                             off_delta_track_bootstrap_95=np.percentile(boot, [2.5, 97.5]).tolist(),
                             original_stream=[r for r in rows if r["noise"] == "original_stream"],
                             rows=rows)
    (OUT / "comparison.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    lines = ["# 同状态局部动作干预（诊断）", "",
             "12个已知候选独有失败案例，分布于10条赛道。每局在首次持续全轮离道前50步分叉，原动作、油门×0.8、转向每步变化≤1三种模式；局部干预50步，随后恢复论文式控制器，观察从分叉起最多200步。每个案例有原始后续采样流及两条新流，合计36组配对、108个分支。干预前观察、RNN状态、动作历史和采样器状态在每组完全相同；原流的基线后续动作及车轮状态与保存轨迹逐步一致。", "",
             "|模式|离道步数差/200步|改善/变差/持平（36组）|奖励差|新增赛道块差|离道差95%区间（按10赛道重采样）|",
             "|---|---:|---:|---:|---:|---:|"]
    for mode in MODES[1:]:
        r = reports[mode]
        ci = r["off_delta_track_bootstrap_95"]
        lines.append(f"|{mode}|{r['off_delta_mean']:+.2f}|{r['off_better']}/{r['off_worse']}/{r['off_tied']}|{r['reward_delta_mean']:+.2f}|{r['tiles_delta_mean']:+.2f}|[{ci[0]:+.2f},{ci[1]:+.2f}]|")
    lines += ["", "## 后续采样流分解", "", "|模式|原流离道差|新流1离道差|新流2离道差|按赛道平均改善数|", "|---|---:|---:|---:|---:|"]
    for mode in MODES[1:]:
        rows = reports[mode]["rows"]
        values = [np.mean([r["off_delta"] for r in rows if r["noise"] == noise]) for noise in NOISES]
        seeds = set(r["seed"] for r in rows)
        improved = sum(np.mean([r["off_delta"] for r in rows if r["seed"] == seed]) < 0 for seed in seeds)
        lines.append(f"|{mode}|{values[0]:+.2f}|{values[1]:+.2f}|{values[2]:+.2f}|{improved}/{len(seeds)}|")
    lines += ["", "负离道差代表更少全轮离道；正奖励差代表更好。原始后续流已知会发生失败，另外两条流检验该状态在不同潜变量噪声下的稳定性。原流上的改善不能单独解释为通用修复，尤其转向限幅在新流1平均增加离道。所有案例都选自已看过的失败集，局部200步结果不是1000步完整赛道成绩。不能直接改默认权重。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
