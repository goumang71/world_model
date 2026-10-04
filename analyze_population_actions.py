"""Summarize pre-departure behavior across fresh development tracks."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "population_action_diagnosis_20261003"


def features(trace):
    if not trace:
        return None
    steer = np.array([r["steer"] for r in trace])
    raw = np.array([r["raw_steer"] for r in trace])
    gas = np.array([r["gas"] for r in trace])
    brake = np.array([r["brake"] for r in trace])
    return dict(n=len(trace), speed_mean=float(np.mean([r["speed"] for r in trace])),
                speed_max=float(max(r["speed"] for r in trace)),
                gas_mean=float(gas.mean()), brake_mean=float(brake.mean()),
                raw_saturated_fraction=float(np.mean(np.abs(raw) > .95)),
                steer_saturated_fraction=float(np.mean(np.abs(steer) > .95)),
                large_reverse_count=int(np.sum((steer[1:] * steer[:-1] < 0) &
                                               (np.abs(steer[1:] - steer[:-1]) > .8))),
                sampled_vs_mean_z_sign_disagreement=float(np.mean(raw * np.array([r["mean_z_steer"] for r in trace]) < 0)))


def summarize_episode(row):
    trace = row["trace"]
    # The frozen collector treated every early end as a boundary exit. A lap
    # finishes early too; its last recorded coverage is exactly one.
    boundary_exit = len(trace) < 1000 and trace[-1]["coverage"] < 1.0 - 1e-12
    full = np.array([r["road_wheels"] == 0 for r in trace])
    initial = next((i for i in range(30, len(trace)) if full[i]), None)
    events = []
    j = 0
    while j < len(trace):
        if not full[j]:
            j += 1
            continue
        start = j
        while j < len(trace) and full[j]:
            j += 1
        if j - start >= 10 and start > 30:
            events.append(dict(step=start + 1, duration=j - start,
                               pre50=features([r for r in trace[max(0, start - 50):start] if r["road_wheels"] > 0])))
    before = features([r for r in trace[max(0, initial - 50):initial] if r["road_wheels"] > 0]) if initial is not None else None
    onroad = features([r for r in trace if r["road_wheels"] == 4])
    return dict(track_seed=row["track_seed"], latent_seed=row["latent_seed"],
                episode_return=row["episode_return"], fully_off_steps=row["fully_off_steps"],
                boundary_exit=boundary_exit,
                severe_failure=bool(row["episode_return"] < 500 or row["fully_off_steps"] / row["steps"] > .2 or boundary_exit),
                first_full_off_step=initial + 1 if initial is not None else None,
                first_pre50=before, sustained_departures=events, onroad=onroad)


def main():
    config = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    cases = []
    for track, noise in config["cases"]:
        path = OUT / "cases" / f"track_{track}_noise_{noise}.json"
        if path.exists():
            cases.append(summarize_episode(json.loads(path.read_text(encoding="utf-8"))))
    if not cases:
        print("No cases yet")
        return
    np_values = np.array([r["episode_return"] for r in cases])
    summary = dict(episodes=len(cases), mean_return=float(np_values.mean()),
                   median_return=float(np.median(np_values)),
                   severe_failures=sum(r["severe_failure"] for r in cases),
                   mean_full_off_steps=float(np.mean([r["fully_off_steps"] for r in cases])),
                   episodes_with_full_off=sum(r["first_full_off_step"] is not None for r in cases),
                   sustained_departures=sum(len(r["sustained_departures"]) for r in cases))
    groups = {}
    for name, sample in (("severe", [r for r in cases if r["severe_failure"]]),
                         ("nonsevere", [r for r in cases if not r["severe_failure"]])):
        first = [r["first_pre50"] for r in sample if r["first_pre50"]]
        if first:
            groups[name] = dict(n=len(first), **{key: float(np.mean([r[key] for r in first])) for key in
                                                    ("speed_mean", "gas_mean", "brake_mean", "raw_saturated_fraction",
                                                     "steer_saturated_fraction", "large_reverse_count", "sampled_vs_mean_z_sign_disagreement")})
    (OUT / "analysis.json").write_text(json.dumps(dict(summary=summary, groups=groups, cases=cases), indent=2), encoding="utf-8")
    lines = ["# 多赛道原策略动作诊断", "",
             "16条未用于此前调参的开发赛道×3组潜变量采样。此处是找共性和提出候选的开发集；90000至90031赛道保留作独立测试。", "",
             f"已完成 {len(cases)}/{len(config['cases'])} 局；均分 {summary['mean_return']:.2f}；严重失败 {summary['severe_failures']} 局；全轮离道均值 {summary['mean_full_off_steps']:.1f} 步/局。", "",
             "## 首次全轮离道前50步", "",
             "|组别|案例数|均速（仿真单位）|平均油门|平均刹车|原始转向饱和|执行转向饱和|大幅反向次数|采样/均值z转向异号|",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, g in groups.items():
        lines.append(f"|{name}|{g['n']}|{g['speed_mean']:.2f}|{g['gas_mean']:.3f}|{g['brake_mean']:.5f}|{g['raw_saturated_fraction']:.1%}|{g['steer_saturated_fraction']:.1%}|{g['large_reverse_count']:.1f}|{g['sampled_vs_mean_z_sign_disagreement']:.1%}|")
    lines += ["", "严重失败=回报<500、全轮离道>20%步数或边界退出。此表仅含出现全轮离道的案例；严重失败与非严重组并不是随机分配，速度或转向差异是相关性而非因果证明。仿真器轮胎接触与速度仅用于离线分析，不进入控制策略。", "",
              "## 各赛道多噪声结果", "", "|赛道|三组回报|三组全轮离道步数|", "|---|---|---|"]
    for track in sorted({r["track_seed"] for r in cases}):
        rows = [r for r in cases if r["track_seed"] == track]
        scores = " / ".join(f"{r['episode_return']:.1f}" for r in rows)
        off_steps = " / ".join(str(r["fully_off_steps"]) for r in rows)
        lines.append(f"|{track}|{scores}|{off_steps}|")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
