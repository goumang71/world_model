"""Make a concise report from the saved paper-map CMA-ES run."""
import json
import argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "paper_controller_train_20261003"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    out = parser.parse_args().output_dir.resolve()
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    old = status.get("baseline_validation")
    initial = status.get("paper_initial_validation")
    lines = ["# 论文控制方式CMA-ES训练结果", "",
             "原论文版动作：steer=tanh(raw0), gas=(tanh(raw1)+1)/2, brake=clip(tanh(raw2),0,1)，直接输出转向。训练冻结本项目已下载VAE/MDN；项目的96×96预处理和CarRacing-v3保持原状。因此检验的是控制器动作映射与小预算再训练，不是完整原论文复现。", "",
             f"状态：{status['status']}；已完成代数：{status['generation']}/{config['generations']}；验证选中代数：{status['selected_generation']}（0表示验证阶段保留原控制器；最终推广还须看独立测试）。", "",
             "|新验证集，32赛道×2噪声|均分|严重失败|全轮离道比例|", "|---|---:|---:|---:|"]
    for label, item in (("原控制器", old), ("论文映射初始权重", initial)):
        if item:
            lines.append(f"|{label}|{item['mean_return']:.2f}|{item['severe_failures']}|{item['fully_off_fraction']:.2%}|")
    for row in status["history"]:
        if "validation" in row:
            item = row["validation"]
            lines.append(f"|论文映射第{row['generation']}代 {'通过' if row['accepted'] else '未通过'}|{item['mean_return']:.2f}|{item['severe_failures']}|{item['fully_off_fraction']:.2%}|")
    if initial:
        direction = "高于" if initial["mean_return"] > old["mean_return"] else "低于"
        lines += ["", f"论文映射初始权重在本批验证集均分{direction}原控制器；这只是一个赛道划分上的结果，不能单独用于替换默认权重。"]
    accepted = [row for row in status["history"] if row.get("accepted")]
    if accepted:
        selected = accepted[-1]["generation"]
        baseline_rows = json.loads((out / "validation_baseline.json").read_text(encoding="utf-8"))
        selected_rows = json.loads((out / f"validation_gen{selected:03d}.json").read_text(encoding="utf-8"))
        repeats = len(baseline_rows) // 32
        diff = np.array([b["episode_return"] - a["episode_return"]
                         for a, b in zip(baseline_rows, selected_rows)]).reshape(32, repeats).mean(axis=1)
        rng = np.random.default_rng(20261003)
        interval = np.percentile(rng.choice(diff, (20000, 32), replace=True).mean(axis=1), [2.5, 97.5])
        lines += ["", f"第{selected}代通过预设验证门槛；验证配对均分差{diff.mean():+.2f}，按赛道重采样95%区间[{interval[0]:+.2f},{interval[1]:+.2f}]。验证通过只是进入独立测试的条件。"]
    else:
        lines += ["", "训练赢家均未通过预设验证门槛，默认控制器未获替换候选。"]
    if (out / "comparison.json").exists():
        test = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
        lines += ["", "## 独立测试，32赛道×3噪声", "",
                  f"测试对象：论文映射第{test['tested_generation']}代；{'仅诊断，未通过验证' if test['diagnostic_only'] else '验证选中候选'}。", "",
                  "|指标|原控制器|论文映射候选|", "|---|---:|---:|"]
        for label, key, fmt in (("均分", "mean_return", ".2f"), ("严重失败", "severe_failures", "d"),
                                ("全轮离道比例", "fully_off_fraction", ".2%")):
            a, b = test["original"][key], test["candidate"][key]
            lines.append(f"|{label}|{format(a,fmt)}|{format(b,fmt)}|")
        ci = test["cluster_bootstrap_95_interval"]
        lines += ["", f"配对均分差 {test['mean_difference']:+.2f}，提高{test['improved_pairs']}局、降低{test['worse_pairs']}局、持平{test['tied_pairs']}局；按赛道重采样95%区间[{ci[0]:+.2f},{ci[1]:+.2f}]。"]
        test_pass = (not test["diagnostic_only"] and ci[0] > 0 and
                     test["candidate"]["severe_failures"] <= test["original"]["severe_failures"] and
                     test["candidate"]["fully_off_fraction"] <= test["original"]["fully_off_fraction"])
        lines += ["", f"最终推广门槛：{'通过' if test_pass else '未通过'}。独立测试不用于重新选择代数；即使某候选通过验证，若独立测试未显示可靠均分提升并保持安全指标，默认控制器仍不替换。"]
    else:
        lines += ["", "独立测试尚未结束，先读取status.json并等待所有结果；不要以部分测试选模。"]
    lines += ["", "原始逐局回报与模型参数在本目录。原下载权重和正式评估脚本不被本训练覆盖。有限代数和赛道下的失败不能证明论文式映射无效；本实验仍冻结现有VAE/MDN，不能排除其表征或泛化限制。"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", out / "REPORT.md")


if __name__ == "__main__":
    main()
