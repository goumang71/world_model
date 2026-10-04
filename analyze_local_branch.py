"""Summarize the saved paired same-state branching experiment."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "local_branch_diagnosis_20261003"


def main():
    config = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    results = []
    for seed in (4470, 52):
        for j in range(5):
            pair = []
            for mode in ("baseline", "smooth_05"):
                path = OUT / f"seed_{seed}" / f"noise_{j}_{mode}" / "result.json"
                if not path.exists():
                    continue
                r = json.loads(path.read_text(encoding="utf-8"))
                pair.append(r)
            if len(pair) == 2:
                assert pair[0]["prefix"] == pair[1]["prefix"], (seed, j)
                a, b = (r["summary"] for r in pair)
                results.append(dict(seed=seed, noise=j, prefix=pair[0]["prefix"], baseline=a,
                                    smooth_05=b,
                                    delta_coverage=b["coverage_change"] - a["coverage_change"],
                                    delta_off=b["fully_off_steps"] - a["fully_off_steps"],
                                    delta_reward=b["reward_sum"] - a["reward_sum"]))
    lines = ["# 同一入弯状态局部对照", "",
             "每组都用原策略和原采样重放至分叉点，核对原轨迹动作与位置；两支从相同画面、RNN状态、位置、速度和已行驶动作出发。分叉后使用同一组潜变量采样随机数；新轨迹会使后续VAE均值/方差与记忆状态变化，因此是闭环对照，不是强行保持每一帧相同潜变量。",
             "", "每个状态5组噪声，第一组继续原评估的随机数流。仅比较干预后的150步，coverage change是已访问赛道块比例的变化，不等于几何完赛率。", "",
             "|赛道|噪声组|原策略首次全轮离道|平滑首次全轮离道|原策略全轮离道步数|平滑全轮离道步数|原策略赛道进度增量|平滑赛道进度增量|",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        a, b = r["baseline"], r["smooth_05"]
        lines.append(f"|{r['seed']}|{r['noise']}|{a['first_full_off_step'] or '无'}|{b['first_full_off_step'] or '无'}|{a['fully_off_steps']}|{b['fully_off_steps']}|{a['coverage_change']:.1%}|{b['coverage_change']:.1%}|")
    lines += ["", "## 按状态汇总", "",
              "|赛道|完成对数|平滑离道步数更少/相等/更多|平滑进度更高/相等/更低|平均离道步数差（平滑－原）|平均进度差（平滑－原）|",
              "|---|---:|---:|---:|---:|---:|"]
    for seed in (4470, 52):
        subset = [r for r in results if r["seed"] == seed]
        if not subset:
            continue
        d_off = np.array([r["delta_off"] for r in subset])
        d_cov = np.array([r["delta_coverage"] for r in subset])
        lines.append(f"|{seed}|{len(subset)}|{sum(d_off < 0)}/{sum(d_off == 0)}/{sum(d_off > 0)}|{sum(d_cov > 1e-12)}/{sum(abs(d_cov) <= 1e-12)}/{sum(d_cov < -1e-12)}|{d_off.mean():+.1f}|{d_cov.mean():+.1%}|")
    lines += ["", "## 本轮结论", "",
              "原策略在4470同一分叉状态下，只更换后续VAE采样流，150步全轮离道可从0步变为79步，证明这个闭环决策对采样流敏感；并未定位到是哪一次采样先引起分叉。入弯前50步开启平滑在两状态、各5组采样下没有稳定减少离道，赛道进度也没有一致改善。先前从起跑就平滑救回4470，可能依赖141步之前形成的不同车辆状态，不能由此推荐固定平滑参数。保留原策略。",
              "", "## 解释边界", "",
              "这两个状态来自已知离道案例，噪声组只有5组；比较的是局部因果干预对150步闭环结果的影响，不能由均值推断随机赛道泛化。原策略分叉前的轨迹一致性经过逐步验证，original_stream的分叉后150步也与历史轨迹核对。若平滑仅对一个状态有效，下一步应分析状态触发条件与恢复行为，不能把平滑直接加入默认控制器。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "analysis.json").write_text(json.dumps(dict(config=config, paired=results), indent=2, ensure_ascii=False), encoding="utf-8")
    print("Analyzed", len(results), "pairs")


if __name__ == "__main__":
    main()
