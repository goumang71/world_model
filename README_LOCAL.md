# World Models 开源预训练基线

来源：https://github.com/BrunooCS/World-Model-2018
下载日期：2026-09-26；通过 GitHub main 分支 ZIP 下载（Git 连接失败）。
原作者源码和权重位于 World-Model-2018-main；源码已做下述本地训练修复，预训练权重未修改。
这份结果属于第三方预训练权重的本地评估，不是我们从零训练的结果。
教学项目仍在相邻的 world_model_learning 文件夹。

## 本地环境

以 D:\Anaconda\envs\pytorch_gpu_a\python.exe 创建 .venv，使用 --system-site-packages 复用 torch 2.5.1 CUDA。
新增依赖安装在 .venv：gymnasium==1.2.3、Box2D==2.3.10、pygame==2.6.1。
最初仅安装评估依赖；现已补装 h5py==3.16.0 和 cma==4.5.0 用于训练验证。

## 运行

PowerShell，在任意目录运行：

```powershell
& 'D:\Anaconda\envs\pytorch_gpu_a\python.exe' 'D:\vs_conda_pytorch\world_models_baseline\run_baseline.py' --episodes 3
```

观看实时驾驶窗口：

```powershell
& 'D:\Anaconda\envs\pytorch_gpu_a\python.exe' 'D:\vs_conda_pytorch\world_models_baseline\run_baseline.py' --episodes 1 --human
```

run_baseline.py 只负责转交给基线独立环境；evaluate_baseline.py 加载三份 state_dict（weights_only=True）并评估。
保持作者的 96×96 裁剪缩放、随机 z 采样与动作变换；使用单环境，正确累计结束步奖励并同时处理 terminated/truncated。
固定种子起始值 42，每局递增。CarRacing-v3，lap_complete_percent=1.0，最多 1000 步。
现版本每次在 outputs/evaluation_时间戳/ 下保存报告和首局视频，避免覆盖旧结果。--no-video 可关闭录像；--output-dir 可指定一个尚不存在的目录。旧 outputs/evaluation.json 和 baseline_seed42.mp4 为首次验证的历史结果。
三局仅用于运行验证，不代表论文的 100 局评估，不直接比较旧版 CarRacing-v0 的论文分数。

## 首次本地验证结果（2026-09-26）

CUDA 推理、三份权重严格加载成功。
- seed 42：825.7951
- seed 43：569.9670
- seed 44：530.3725
- 平均：642.0448

三局均在 1000 步时间上限结束。结果可演示，但有明显波动，不声称达到原论文分数。

## 本地训练流程修复（2026-09-27）

- VAE、Memory 的训练及可视化统一读取采集示例输出的 `car_racing_data_128.h5`，同步修改 Notebook 和 `memory.py`。这是 128 条轨迹的示例配置，不代表预训练权重的数据量；当前未附带数据，需要先采集。以后扩充数据时需同步调整采集数量、输出文件名和所有读取路径。
- 修正 VAE DataLoader 的 `persistent_workers` 参数拼写。
- 控制器 CMA-ES 从第 1 代开始，`max_generations` 表示本次从头训练的总代数；不支持恢复优化器状态的断点续训。
- 控制器并行评估先累计仍在进行的回合的本步奖励，再合并 `terminated` 和 `truncated`，保留终止步奖励，排除已结束回合的后续奖励。同步修改 `controller.py` 和对应 Notebook。
- 本次不运行训练、不改写预训练权重及历史评估结果。Notebook 的原有输出为历史记录，不是修复后重新运行的结果。
- 运行训练 Notebook 时工作目录应为 `World-Model-2018-main`。本地环境此前仅用于评估，训练依赖及完整训练流程尚未验证。

## 首次小规模训练链路验证（历史记录，2026-09-27）

在项目目录下执行：

```powershell
& 'D:\vs_conda_pytorch\world_models_baseline\.venv\Scripts\python.exe' 'D:\vs_conda_pytorch\world_models_baseline\verify_training.py'
```

已在项目虚拟环境补装 h5py==3.16.0、cma==4.5.0。该入口当前要求 CUDA，与原控制器辅助函数一致。

首次版本从原 Notebook / Python 文件提取采集、Dataset、训练和控制器评估函数，跳过顶层长训练及交互演示。实际串行采集 2 段各 64 步数据（总计 128 帧，不是 128 条完整轨迹），合并为 HDF5（保留浮点奖励），VAE 训练 1 轮/8 批，Memory 训练 1 轮/2 批，CMA-ES 训练 2 代，请求种群为 4，每个候选评估 1 次/64 步。随后重新加载全部新权重，在 seed=9001 下评估最多 200 步。

每次建立独立 outputs/training_smoke_时间戳 目录，不覆盖下载权重。首次版本采集未固定种子；当前版本已固定采集种子并采用双进程和种群6，CMA 本身仍可能使用不同随机种子，重复运行结果不保证相同。

本次通过报告：outputs/training_smoke_20260927_155136_546664/report.json。VAE、Memory 参数实际更新，权重保存/加载及 CMA 两代训练通过，200 步评估回报为 21.8410。原始三份权重 SHA-256 校验未变化。这是流程验证，不是驾驶效果基准，不能与原有 1000 步预训练评估直接比较。另保留首次验证入口缺少 Controller 全局绑定时的失败记录，已修复入口后重跑通过。

验证时另外修复了 memory.py / Memory Notebook 的训练衔接：from utils import VAE 改为 from models.vae import VAE；加载 checkpoints/vae.pth；保存 checkpoints/memory.pth，与前后阶段一致。

上述为首次验证范围。后续已完成 Windows 双进程采集、梯度裁剪及回合边界检查、显存探测和20局固定种子评估，见下方当前说明。完整 Notebook Run All 和全部可视化仍未验证；部分可视化单元格依赖未附带的 utils 绘图函数。

## 当前训练前检查与固定基线

详见 `TRAINING_CHECKS.md`。Memory 的梯度裁剪已移至 backward 与 step 之间；回合边界分段训练；采集同时记录 terminated/truncated；合并奖励使用 float32。数据采集移至可导入的 rollout_data.py，Windows spawn 双进程验证通过。VAE 默认 batch_size=64、num_workers=0；Memory 每64帧编码，1000帧序列的训练更新已验证。

当前基线使用原始权重，CarRacing-v3，种子42至61，共20局，每局最多1000步：平均715.6224，样本标准差160.9592，中位数777.8287，最低210.0000，最高857.1429。全部20局在时间上限截断。前3局与历史结果相同，均值变化不代表模型变好。

报告：outputs/baseline_fixed_42_61/evaluation.json。新评估程序保存种子、环境设置、库版本和权重SHA-256，可通过 --checkpoint-dir 指定新权重目录做相同条件的对比。

## 控制器从头训练结果

已冻结原VAE/Memory，从随机初始化完成20代CMA-ES训练，每代16候选，每个候选3条训练赛道；原权重保持不变。新权重在 outputs/controller_retrain_20gen/checkpoints/。同样种子42至61、1000步协议下，新控制器平均129.10，旧控制器715.62。新模型效果较弱，默认run_baseline.py仍使用旧权重；新模型观看和复评命令见 CONTROLLER_TRAINING.md。

## 驾驶状态面板（2026-09-28）

原命令不变：`python run_baseline.py --episodes 1 --seed 43 --human`。实时窗口和导出首局视频顶部显示种子、步数/1000、得分、赛道覆盖率、运行/完圈/结束状态、接触赛道车轮数。模型仍只接收原始环境观测，显示面板不进入VAE。

覆盖率为已访问赛道块/总块数；少于4个车轮接触赛道算部分出界，0个算完全出界，是物理接触口径，不是车身轮廓越线检测。评估报告新增覆盖率、完圈标记、出界步数/比例、完全出界步数。关闭实时窗口或按Esc会结束本次评估，保存已运行的回合并标记user_closed，不继续下一局。

已复测种子43：得分569.9669966996587与修改前一致，覆盖率66.997%，未完圈，出界291/1000步（含部分出界），其中完全出界143步。示例视频位于 outputs/evaluation_20260928_150308_047428/baseline_seed43.mp4。

## 潜变量采样对照（2026-09-28）

已对原权重在种子42至61进行sample/mean配对评估。mean转向变化较小，但平均得分429.83、覆盖率53.13%，差于sample的715.62、81.56%；默认仍是sample。运行时可用 --latent-mode mean 诊断，--record-trace 保存动作轨迹。详细结果：outputs/latent_diagnosis/REPORT.md。
