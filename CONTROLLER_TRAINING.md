# 独立控制器训练

入口是 train_controller.py。固定下载的 VAE 和 Memory，仅从随机初始化训练867参数的 Controller，不需要10k离线数据集。CarRacing 模拟器实时提供观测与奖励。训练、挑选权重与最终比较使用不同种子。

本次配置：20代，每代16个候选，每个候选3局，每局最多1000步；CMA初始sigma=0.1。所有候选在同一代使用相同赛道和相同潜变量噪声流，下一代更换3条训练赛道。每5代将该代优胜者放到独立的5条选择赛道，保留选择平均分最高的检查点。最终42至61号比较赛道不参与训练或权重选择。

训练目录：outputs/controller_retrain_20gen/。

- config.json：预算、种子、库版本和源权重SHA-256。
- status.json：每代更新的状态、得分、耗时与所选代数。
- generations/：各代优胜控制器，不等同于在同一测试集上的最佳模型。
- checkpoints/：冻结VAE/Memory的副本和选出的新controller.pth。
- cma_state.pkl：本地生成的优化器状态快照，供故障恢复；当前CLI没有自动续训参数。不要加载来源不明的pickle。

本次会保留原项目的checkpoint文件。训练结束后，默认 run_baseline.py 仍然使用旧权重。

## 看新控制器

训练完成后，在项目根目录激活 .venv，再执行：

```powershell
python run_baseline.py --episodes 1 --human --checkpoint-dir outputs/controller_retrain_20gen/checkpoints
```

固定20局复评并保存首局视频：

```powershell
python run_baseline.py --episodes 20 --seed 42 --checkpoint-dir outputs/controller_retrain_20gen/checkpoints
```

## 新建另一次训练

```powershell
python train_controller.py --generations 20 --population 16 --rollouts 3
```

默认创建新的时间戳目录；显式 --output-dir 必须是尚不存在的目录。检查点保存在自己的训练目录，不会覆盖原始权重。20代是小预算试验，不保证收敛或超过预训练模型。

## 本次实际结果

已完成20代从随机初始化的训练，用时26.22分钟。选择赛道上第20代平均181.15，最终保留第20代。

最终20条固定比较赛道：新控制器平均129.10，样本标准差71.10，最低-65.64，最高296.43；旧控制器平均715.62。新控制器尚未达到旧模型效果，默认演示继续使用旧模型。训练和保存成功不等于性能提升。

对比报告：outputs/controller_retrain_20gen/comparison.md。首局视频：outputs/controller_retrain_20gen/evaluation_fixed_42_61/baseline_seed42.mp4（1000帧已验证可读取）。

## 从下载权重附近继续优化（2026-09-30）

入口：`train_controller_warmstart.py`。这是一轮新的 CMA-ES 搜索，以下载的 Controller 权重为中心，初始 sigma=0.05；不是恢复原作者的 CMA 优化器状态。默认20代、每代16个候选、每候选3条共同训练赛道。冻结 VAE/Memory，保留原奖励和油门刹车映射，转向变化限制1.0，裁掉底部12像素后bicubic缩放96x96，采样z，每局1000步。

训练种子10000起，验证20000..20007，最终测试30000..30023。原始Controller作为第0代候选，每5代在固定验证种子上检查当代训练优胜者，验证分数超过已保留模型才更新。最终测试不参与选择。

训练评估器支持可选 `steering_delta`，默认2.0保持旧训练入口行为；warmstart入口显式传1.0。单环境图像view布局与正式评估对齐，并先检查seed42是否复现832.862190812706；若用户后来改变正式流程，这项校验应重新建立。批量计算与单环境可能有浮点差异，所以最终还须使用 `evaluate_baseline.py` 对新旧权重做同种子正式复评，不能只报告批量训练器的结果。

```powershell
.\.venv\Scripts\python.exe -B train_controller_warmstart.py
```

该命令新建输出目录，不覆盖下载权重。目录保留来源脚本快照、权重哈希、配置、每代权重、CMA状态、验证记录与批量最终对照。20代仅用于诊断，不代表充分收敛。

本轮warmstart已完成，输出 `outputs/controller_warmstart_20260930_171351`。20代训练及批量最终测试约27分钟，按验证集选择第10代（769.04 -> 818.60）。正式脚本在24条独立测试赛道上，旧模型771.07，新模型736.21；完全离道比例4.87% -> 9.13%。本轮不建议替换默认模型。完整结论见该目录REPORT.md，正式结果见canonical_comparison.json，首局视频位于canonical_candidate/。两个早期parity失败目录保留用于诊断，不是已完成训练。

## 赛道与潜变量采样分离测试

`diagnose_selection_stability.py`固定原控制器与上述第10代候选，用正式评估流程在8条原验证赛道、16条新赛道各测试3组新的潜变量噪声，并复查原验证赛道的原始噪声，共160局。不会训练或覆盖权重。`analyze_selection_stability.py <输出目录>`汇总同赛道波动、胜负翻转及按赛道聚类的区间。

本次输出：`outputs/selection_stability_20260930_221209`。新赛道原模型768.13、候选759.62；候选未显示可靠提升，同赛道采样标准差约为原模型2倍。11/16条新赛道出现采样改变后新旧胜负翻转。详见该目录REPORT.md与fresh_noise_scores.png。
