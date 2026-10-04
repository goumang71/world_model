"""Untouched holdout comparison for the predeclared 0.75 latent-noise candidate."""
import json
from pathlib import Path

import numpy as np
from controller_evaluation import should_replace

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs" / "population_holdout_baseline_20261003"
CANDIDATE = ROOT / "outputs" / "population_holdout_noise075_20261003"
OUT = ROOT / "outputs" / "population_holdout_comparison_20261003"


def corrected(row):
    last = row["trace"][-1]
    boundary = row["steps"] < 1000 and last["coverage"] < 1.0 - 1e-12
    off = row["fully_off_steps"] / row["steps"]
    return dict(track_seed=row["track_seed"], latent_seed=row["latent_seed"],
                episode_return=row["episode_return"], steps=row["steps"],
                fully_off_steps=row["fully_off_steps"], boundary_exit=boundary,
                lap_finished=bool(row["steps"] < 1000 and not boundary),
                severe_failure=bool(row["episode_return"] < 500 or off > .2 or boundary))


def summary(rows):
    return dict(episodes=len(rows), mean_return=float(np.mean([r["episode_return"] for r in rows])),
                severe_failures=sum(r["severe_failure"] for r in rows),
                fully_off_fraction=sum(r["fully_off_steps"] for r in rows) / sum(r["steps"] for r in rows),
                lap_finishes=sum(r["lap_finished"] for r in rows),
                boundary_exits=sum(r["boundary_exit"] for r in rows))


def main():
    config_base = json.loads((BASE / "config.json").read_text())
    config_new = json.loads((CANDIDATE / "config.json").read_text())
    assert config_base["cases"] == config_new["cases"]
    assert config_base["weight_sha256"] == config_new["weight_sha256"]
    pairs = []
    for track, noise in config_base["cases"]:
        name = f"track_{track}_noise_{noise}.json"
        old_path = BASE / "cases" / name
        new_path = CANDIDATE / "cases" / name
        if not old_path.exists() or not new_path.exists():
            continue
        a, b = (corrected(json.loads(p.read_text())) for p in (old_path, new_path))
        pairs.append(dict(track_seed=track, latent_seed=noise, baseline=a, candidate=b,
                          return_difference=b["episode_return"] - a["episode_return"],
                          off_difference=b["fully_off_steps"] - a["fully_off_steps"]))
    if not pairs:
        print("No complete pairs")
        return
    old = summary([p["baseline"] for p in pairs])
    new = summary([p["candidate"] for p in pairs])
    gate = should_replace(new, old)
    track_diffs = [np.mean([p["return_difference"] for p in pairs if p["track_seed"] == track])
                   for track in sorted({p["track_seed"] for p in pairs})]
    rng = np.random.default_rng(20261003)
    draws = rng.choice(track_diffs, size=(10000, len(track_diffs)), replace=True).mean(axis=1)
    interval = np.quantile(draws, [.025, .975]).tolist()
    # Promotion requires all three safety/score metrics and a positive
    # track-level bootstrap lower bound on paired return improvement.
    promotion = len(pairs) == len(config_base["cases"]) and gate and interval[0] > 0
    result = dict(completed_pairs=len(pairs), expected_pairs=len(config_base["cases"]),
                  baseline=old, candidate=new, gate_passed=gate, promotion_criteria_met=promotion,
                  paired_improved=sum(p["return_difference"] > 1e-9 for p in pairs),
                  paired_worsened=sum(p["return_difference"] < -1e-9 for p in pairs),
                  mean_return_difference=new["mean_return"] - old["mean_return"],
                  track_cluster_bootstrap_95pct=interval, pairs=pairs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# 潜变量采样幅度0.75：独立测试配对结果", "",
             f"已完成 {len(pairs)}/{len(config_base['cases'])} 对。相同赛道与同一批标准正态随机数，候选将z采样扰动乘0.75，VAE/MDN/C权重不变。历史动作与画面在第一步后可能分叉，属于闭环对照。", "",
             "|指标|原策略|候选0.75|", "|---|---:|---:|",
             f"|均分|{old['mean_return']:.2f}|{new['mean_return']:.2f}|",
             f"|严重失败|{old['severe_failures']}|{new['severe_failures']}|",
             f"|全轮离道比例|{old['fully_off_fraction']:.2%}|{new['fully_off_fraction']:.2%}|",
             f"|正常完圈|{old['lap_finishes']}|{new['lap_finishes']}|",
             f"|边界退出|{old['boundary_exits']}|{new['boundary_exits']}|", "",
             f"成对分数提高 {result['paired_improved']} 局、降低 {result['paired_worsened']} 局；平均分差 {result['mean_return_difference']:+.2f}；按赛道重采样的95%区间[{interval[0]:+.2f}, {interval[1]:+.2f}]。", "",
             f"预设选模门槛（均分提高、严重失败不增加、全轮离道比例不增加）：{'通过' if gate else '未通过'}。",
             f"默认策略推广门槛还要求全部96对完成，且按赛道重采样的平均分差95%区间下界>0：{'通过' if promotion else '未通过'}。",
             "", "## 各赛道（3组噪声）", "", "|赛道|原平均分|候选平均分|改善/下降局数|", "|---|---:|---:|---:|"]
    for track in sorted({p["track_seed"] for p in pairs}):
        subset = [p for p in pairs if p["track_seed"] == track]
        lines.append(f"|{track}|{np.mean([p['baseline']['episode_return'] for p in subset]):.1f}|{np.mean([p['candidate']['episode_return'] for p in subset]):.1f}|{sum(p['return_difference']>0 for p in subset)}/{sum(p['return_difference']<0 for p in subset)}|")
    lines += ["", "严重失败=回报<500、全轮离道>20%步数或跑出边界。完圈会提前结束，按最后覆盖率=100%与边界退出区分。独立测试未用于选择0.75参数；若推广门槛未过，默认策略保持原样。此测试仍只覆盖CarRacing-v3和所选赛道分布，不能保证真机性能。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
