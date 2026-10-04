# 训练前检查与固定基线（2026-09-27）

## 已修复并验证

- Memory：每批先清梯度，再 backward，随后全局梯度范数裁剪到1，最后 optimizer.step。非有限梯度直接报错。脚本与 Notebook 的 Dataset / train_model 实现已核对一致。
- 数据回合边界：采集同时记录 terminated 和 truncated，dones 是两者的或。固定长度数据块可含多个回合；训练在边界处重置 LSTM，并剔除“终止前图像→重置图像”的错误预测对。终止帧未单独保存，所以这些终止转换不参与当前 M 训练。
- HDF5 合并：奖励保留 float32，不再把 -0.1 截成0；原始分块文件保留。旧数据如缺少截断标记，不能通过修改训练代码自动恢复，应重新采集。
- Windows：采集函数位于可导入的 rollout_data.py，显式使用 spawn；已实际验证双进程。训练 Notebook 的 DataLoader 默认 num_workers=0，避免 Notebook 内定义 Dataset 的多进程序列化问题。未宣称验证 num_workers>0 的训练加载。
- 显存：VAE 默认 batch_size=64；Memory 将一条序列分成每批64帧送入 VAE，最后再拼成完整 latent 序列。VAE 最后一个批次为1时保留 batch 维。

RTX 4060 Laptop 8GB 上，合成输入探测的 PyTorch 峰值已分配显存：VAE训练批量64约138.99 MiB；1000帧 Memory 数据读取及训练更新约78.39 MiB。这不包含其他进程、CUDA上下文和所有缓存，并非 nvidia-smi 总占用或长训练保证。

报告：
- outputs/pretraining_checks_20260927_163230_447449/report.json
- outputs/training_smoke_20260927_163050_725307/report.json

当前冒烟测试包含双进程采集、每16步强制截断、8个边界标记、VAE/Memory训练、CMA-ES两代、权重保存和重新加载。原始预训练权重校验未改变。

## 本地复验

在项目根目录使用 PowerShell：

```powershell
& 'D:\vs_conda_pytorch\world_models_baseline\.venv\Scripts\python.exe' 'D:\vs_conda_pytorch\world_models_baseline\verify_pretraining.py'
& 'D:\vs_conda_pytorch\world_models_baseline\.venv\Scripts\python.exe' 'D:\vs_conda_pytorch\world_models_baseline\verify_training.py'
```

每次生成独立测试目录。要求 CUDA；这些是功能检查，不是长训练或收敛验证。尚未验证所有 Notebook 可视化单元格，部分仍引用仓库未附带的 utils 绘图函数。导出的 memory.py / controller.py 含有顶层训练和演示代码，不应作为普通库直接导入；验证入口只提取所需定义。

## 固定种子基线

- 权重：原始 checkpoints/ 下三份下载权重。
- 环境：CarRacing-v3，连续动作，lap_complete_percent=1.0，关闭 domain_randomize，每局1000步。
- 种子：42至61，共20局；每局同时设置环境、NumPy和PyTorch种子，继续保留随机 z 采样。
- 平均715.6224，样本标准差160.9592，中位数777.8287，最低210，最高857.1429。
- 20局均达到时间上限；截断不能自动解释为成功或失败。
- 前3局得分与历史评估相同，平均值变化来自测试赛道集合变化，不是模型提升。
- 报告包含逐局回报、版本、种子、权重SHA-256。结果位于 outputs/baseline_fixed_42_61/。

复测（默认写入新时间戳目录）：

```powershell
& 'D:\vs_conda_pytorch\world_models_baseline\.venv\Scripts\python.exe' 'D:\vs_conda_pytorch\world_models_baseline\evaluate_baseline.py' --episodes 20 --seed 42 --no-video
```

未来评估新权重时加 --checkpoint-dir，指向含 vae.pth、memory.pth、controller.pth 的目录。固定评估种子不应用于候选控制器的训练回报；如果反复用此组挑模型，它是验证集，最终效果应另留测试种子。不同硬件/库版本不保证逐位重现。

## 以后采集较大的模拟器数据

无需外部下载数据集；模拟器交互本身可以生成数据。下面是示例命令，本次未执行128×1000的数据采集：

```powershell
& 'D:\vs_conda_pytorch\world_models_baseline\.venv\Scripts\python.exe' 'D:\vs_conda_pytorch\world_models_baseline\World-Model-2018-main\rollout_data.py' --episodes 128 --steps 1000 --workers 2 --seed 123 --output 'D:\vs_conda_pytorch\world_models_baseline\World-Model-2018-main\car_racing_data_128.h5'
```

现有目标文件不会被覆盖。128×1000张96×96 RGB原图约3.3 GiB，原始分块与合并文件同时保留，需额外磁盘空间；合并文件压缩率取决于图像内容。按整段采集轨迹/场景划分训练和验证，避免相邻帧泄漏到两个集合。

## 真机数据需要记录什么

先明确硬件接口与任务，再通过遥控、现有控制器或受限探索采集。建议每条转换记录：episode_id、观测时间戳、观测、动作下发时间戳、动作命令、实际速度/关节反馈、下一观测、terminated、truncated、结束原因；为控制器训练另定义任务奖励或成功标记。相机与执行器时钟和延迟需要对齐。

VAE可由图像自监督训练；Memory需要动作对齐的连续时序数据，不是仅有一堆照片。训练控制器还需要奖励反馈，或改用带动作标签的示范学习，不能认为采完无标注视频就能自动得到可部署策略。当前HDF5加载器专用于赛车，真机传感器和动作维度需要重新适配。
