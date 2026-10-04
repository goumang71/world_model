"""Summarize the paired latent-control branch experiment."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "paper_controller_aligned_20261003" / "latent_control_branches"


def reversals(rows, branch_step, window=50):
    steer = np.array([r["action"][0] for r in rows if r["step"] < branch_step + window])
    return int(((steer[1:] * steer[:-1] < 0) &
                (np.abs(steer[1:]) > .5) &
                (np.abs(steer[:-1]) > .5)).sum())


def main():
    status = json.loads((OUT / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "completed":
        raise RuntimeError("Experiment is incomplete")
    config = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    rows = []
    for seed, latent_seed, _ in config["cases"]:
        for noise in config["noise_modes"]:
            path = OUT / f"track_{seed}_noise_{latent_seed}"
            a = json.loads((path / f"{noise}_baseline" / "result.json").read_text(encoding="utf-8"))
            b = json.loads((path / f"{noise}_mean_control_z" / "result.json").read_text(encoding="utf-8"))
            if a["prefix"] != b["prefix"]:
                raise RuntimeError("Prefix mismatch")
            sa, sb = a["summary"], b["summary"]
            rows.append(dict(seed=seed, latent_seed=latent_seed, noise=noise,
                             off_delta=sb["fully_off_steps"] - sa["fully_off_steps"],
                             reward_delta=sb["reward_sum"] - sa["reward_sum"],
                             tiles_delta=sb["tiles_gained"] - sa["tiles_gained"],
                             reversal_delta=reversals(b["trajectory"], sb["branch_step"]) -
                                            reversals(a["trajectory"], sa["branch_step"]),
                             baseline_off=sa["fully_off_steps"],
                             candidate_off=sb["fully_off_steps"]))
    tracks = sorted(set(r["seed"] for r in rows))
    track_means = np.array([np.mean([r["off_delta"] for r in rows if r["seed"] == seed])
                            for seed in tracks])
    rng = np.random.default_rng(20261003)
    ci = np.percentile(rng.choice(track_means, (20000, len(tracks)), replace=True).mean(1), [2.5, 97.5])
    result = dict(branches=len(rows), tracks=len(tracks),
                  mean_off_delta=float(np.mean([r["off_delta"] for r in rows])),
                  off_improved=sum(r["off_delta"] < 0 for r in rows),
                  off_worsened=sum(r["off_delta"] > 0 for r in rows),
                  off_tied=sum(r["off_delta"] == 0 for r in rows),
                  mean_reward_delta=float(np.mean([r["reward_delta"] for r in rows])),
                  mean_tiles_delta=float(np.mean([r["tiles_delta"] for r in rows])),
                  mean_reversal_delta=float(np.mean([r["reversal_delta"] for r in rows])),
                  off_delta_track_bootstrap_95=ci.tolist(),
                  rows=rows)
    fresh = [r for r in rows if r["noise"] != "original_stream"]
    fresh_tracks = sorted(set(r["seed"] for r in fresh))
    fresh_means = np.array([np.mean([r["off_delta"] for r in fresh if r["seed"] == seed])
                            for seed in fresh_tracks])
    fresh_ci = np.percentile(rng.choice(fresh_means, (20000, len(fresh_tracks)), replace=True).mean(1), [2.5, 97.5])
    result["new_stream_only"] = dict(mean_off_delta=float(np.mean([r["off_delta"] for r in fresh])),
                                      off_delta_track_bootstrap_95=fresh_ci.tolist(),
                                      improved=sum(r["off_delta"] < 0 for r in fresh),
                                      worsened=sum(r["off_delta"] > 0 for r in fresh),
                                      tied=sum(r["off_delta"] == 0 for r in fresh))
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# 控制器潜变量采样通路局部分叉", "",
             "在12个已知候选失败状态、10条赛道上，从首次持续离道前50步开始，比较原论文式控制器与仅将控制器输入z换为VAE后验均值的方案。干预50步，观察后续200步；RNN在两个方案中都继续接收实际采样z和实际执行动作。每个状态用原后续采样流及两条新流，共36组配对、72分支。干预前状态/采样器完全相同，原流基线后缀逐步复现保存轨迹。", "",
             "|后续采样流|离道步数差|改善/变差/持平|奖励差|转向大幅反向次数差（干预50步）|",
             "|---|---:|---:|---:|---:|"]
    for noise in config["noise_modes"]:
        sample = [r for r in rows if r["noise"] == noise]
        lines.append(f"|{noise}|{np.mean([r['off_delta'] for r in sample]):+.2f}|{sum(r['off_delta']<0 for r in sample)}/{sum(r['off_delta']>0 for r in sample)}/{sum(r['off_delta']==0 for r in sample)}|{np.mean([r['reward_delta'] for r in sample]):+.2f}|{np.mean([r['reversal_delta'] for r in sample]):+.2f}|")
    lines += ["", f"总体：离道步数差{result['mean_off_delta']:+.2f}/200步，赛道聚类95%区间[{ci[0]:+.2f},{ci[1]:+.2f}]；奖励差{result['mean_reward_delta']:+.2f}，转向大幅反向次数差{result['mean_reversal_delta']:+.2f}/50步。负离道差表示改善。", "",
              f"排除已知会失败的原流，仅看两条新增后续采样流：平均离道差{result['new_stream_only']['mean_off_delta']:+.2f}/200步，赛道聚类95%区间[{fresh_ci[0]:+.2f},{fresh_ci[1]:+.2f}]，改善/变差/持平{result['new_stream_only']['improved']}/{result['new_stream_only']['worsened']}/{result['new_stream_only']['tied']}。原流的大幅收益会高估推广效果。", "",
              "这是已知失败状态的200步机理试验，不是完整1000步或新赛道泛化评估。仅控制器当前输入使用均值z，RNN仍用原采样z；即使局部改善，也不能直接关闭整个项目的VAE采样。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
