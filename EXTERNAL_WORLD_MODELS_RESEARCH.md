# 可直接使用的外部 World Models 复现筛选（2026-10-03）

目标：找公开、带预训练 V/M/C 权重且能在当前 `CarRacing-v3`、1000步评估中稳定取得较高回报的复现；只有本地实际验证后才考虑替换现有基线。当前默认权重和 `run_baseline.py` 未被外部模型覆盖。

|项目|公开证据|能否直接取用|
|---|---|---|
|[flydexo/world-models-carracing-v3](https://huggingface.co/flydexo/world-models-carracing-v3)|模型卡自报100局平均915.9；公开VAE、RNN、C权重和模块定义，MIT；64×64输入|权重已隔离下载并试跑；模型卡缺完整预处理和动作映射，当前无法复现其高分|
|[ctallec/world-models](https://github.com/ctallec/world-models)|MIT，完整PyTorch训练代码，原版CarRacing-v0|仓库说明要求先自行生成数据并训练；没有可直接用于本机v3的完整预训练V/M/C|
|[Abhi183/world-model](https://github.com/abhi183/world-model)|v3，作者展示单局881，同时报告5局平均325|不满足稳定高分目标|
|[Ha原作者实现](https://github.com/hardmaru/WorldModelsExperiments/tree/master/carracing)|包含原作者的预训练文件与完整旧环境运行方式|面向Gym 0.9.4/CarRacing-v0、TensorFlow 1.8；与目前v3不能直接替换，移植后仍须重测|

## flydexo 本地核对

模型仓库固定在修订版 `6b4b3a15fab37b73d2b8fdd3e231f670e5d65bd3`，下载脚本为 `fetch_flydexo_world_models.py`，文件和SHA256见 `external_models/flydexo_world_models_carracing_v3/provenance.json`。所有文件只放在 `external_models/`，以 `torch.load(weights_only=True)` 载入；原项目三项权重没有修改。

模型卡的加载示例把C写成普通 `nn.Linear`，但实际 `controller.pt` 的键为 `layer.weight`/`layer.bias`，需要显式映射才能加载。模型卡只说输入 `[z;h]`、输出动作，没有公开从原始96×96图像到64×64、从3个logit到转向/油门/刹车的完整评测代码。因此下述试跑是**按论文动作映射和常见图像预处理重建的接口**，不能证明发布权重本身没有高分能力。

使用CarRacing-v3、1000步、相同种子42/43/44，以VAE采样z、64×64双线性缩放、论文动作映射（`tanh`转向、`(tanh(gas)+1)/2`油门、`clamp(tanh(brake),0,1)`刹车）：

|输入|3局均分|每局得分|平均转向|
|---|---:|---|---:|
|裁掉底部12行再缩放|−81.78|−82.33 / −80.20 / −82.81|约−0.98|
|全图缩放|−82.88|−82.33 / −83.50 / −82.81|约−0.98|
|全图缩放、控制器读VAE均值|−85.01|−85.87 / −83.50 / −85.67|约−0.99|

第一帧在这两种预处理下，C的原始转向logit约+51至+54，明显饱和。把输入颜色RGB/BGR交换变化很小；把像素从[0,1]改为[-1,1]使logit更极端。对应输出：`outputs/flydexo_probe_crop12_paper/`、`outputs/flydexo_probe_full_paper/`、`outputs/flydexo_probe_full_mean/`。试跑脚本 `evaluate_flydexo.py`，首帧接口检查 `inspect_flydexo_interface.py`。

额外将线性输出直接裁剪为环境动作（而非论文的 `tanh` 映射），种子42仍为−82.33分，平均转向约−0.982；输出在 `outputs/flydexo_probe_full_rawclip/`。这也没解释模型卡的高分。

**决策**：不能把模型卡自报的915.9当成当前本机可用分数；没有与权重配套的完整评测接口之前，不替换已在独立96局测试中达到787.90均分的当前基线。这里仅验证了三条诊断种子和几种有根据的接口组合，不能宣称外部模型在所有可能接法下都失败。若后来拿到发布者的精确预处理/动作代码，可以用本脚本重测，再用全新赛道做100局配对比较。
