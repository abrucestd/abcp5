# abcp5：二人组评分模型研究资料

本仓库提供 `Train_two_new_mech.cpp` 二人组评分数据生成器、`reg43d_nt_v0.6` 回归模型的训练代码与数据，以及 ONNX/C++ 推理程序。目标是让后续研究能够从原始二人组文本重新生成特征、复现实验、替换模型结构并进行大规模筛选。

## 目录

```text
train/
  Train_two_new_mech.cpp       # 从原始二人组文本生成 130 维训练特征
  dedupe_train_two_sources.py  # 按来源优先级去除重复二人组
  train3_mech.py               # reg43d_nt_v0.6 训练入口和模型定义
  tester4_new_mech.py          # 误差、分段和极端样本分析
  export_mech_onnx.py          # PyTorch checkpoint -> ONNX + scale.txt
data/source/                   # 去重后的四个源文件
data/generated/                # 与上述源文件对应的已生成数据
model/reg43d_nt_v0.6.pt       # 最新 PyTorch checkpoint
inference/                     # ONNX、C++ 推理器及 Windows 可执行文件
actual_ge4650_min_error50_new.csv # 训练时使用的高分难例权重报告
```

## 数据流程

```text
trate.txt, ge4500_2.txt, lt4500_2.txt, ai-n0.txt
        |
        v
dedupe_train_two_sources.py
        |
        v
Train_two_new_mech.cpp -> data_mech.csv + data_mech.index.tsv
        |
        v
train3_mech.py -> reg43d_nt_v0.6.pt
        |
        v
export_mech_onnx.py -> model4.onnx + scale.txt
        |
        v
inference/abcp5.5.exe -> 带预测分数的筛选结果
```

### 数据去重

源文件每行格式为 `<原始评分> <成员1>+<成员2>`，首行为记录数。去重键是两个成员排序后的无序二人组，因此交换成员顺序也会视为重复。优先级为：

```text
trate.txt > ge4500_2.txt > lt4500_2.txt > ai-n0.txt
```

在仓库根目录运行：

```powershell
python train/dedupe_train_two_sources.py
python train/dedupe_train_two_sources.py --apply --backup-dir dedupe_backup
```

随后编译并生成数据（需要 Windows、MinGW/MSVC 兼容的 C++ 编译器；源码依赖原项目的 Windows API 和 OpenMP）：

```powershell
g++ -std=c++17 -O2 -fopenmp train/Train_two_new_mech.cpp -o Train_two_new_mech.exe
./Train_two_new_mech.exe data/generated/data_mech.csv data/generated/data_mech.index.tsv `
  data/source/ge4500_2.txt data/source/lt4500_2.txt `
  data/source/trate.txt data/source/ai-n0.txt
```

程序会把评分乘以 100 写入 CSV；索引 TSV 保存 CSV 行与源文件记录之间的对应关系。已提交的 `data/generated/` 是当前四个去重源文件的匹配版本，可直接用于训练。

## 环境与训练

Python 3.10 或更新版本，推荐 CUDA 版 PyTorch。安装依赖：

```powershell
python -m pip install -r requirements.txt
```

训练命令（从仓库根目录执行）：

```powershell
cd train
python train3_mech.py --csv ../data/generated/data_mech.csv --name ../model/reg43d_nt_v0.6_reproduced
cd ..
```

默认流程固定随机种子 42，按 90% 训练、10% 验证，并另留出 10% 测试集；训练分为预训练、Stage 1、难例重采样 Stage 2 和常规 Stage 3。`actual_ge4650_min_error50_new.csv` 中误差 `>= -150` 的记录会在 Stage 2 获得额外权重；报告缺失时训练仍可运行，但不会启用这部分加权。需要关闭测试集导出时可使用 `--no_split_test`。

训练会输出 checkpoint、曲线图和测试集 CSV。模型输入必须是 `Train_two_new_mech.cpp` 生成的 130 个特征加 1 列目标值，不能直接混用旧版 `Train_two` 数据。

误差分析示例：

```powershell
cd train
python tester4_new_mech.py --model ../model/reg43d_nt_v0.6.pt `
  --npy ../data/generated/data_mech.csv --no_plot
cd ..
```

## 模型结构

每个二人组由两名角色组成，每名角色 65 维，拼接后为 130 维输入。65 维按以下语义组织：8 个基础属性（包括血量）、35 个技能熟练度、9 个幻影/召唤相关字段、11 个末尾主动和末尾座位机制字段，以及生成器保留的机制状态字段。

`Regressor43D` 由四部分组成：

1. **角色特征子网络**：`IllusionNet` 将幻影/召唤字段压缩为可学习表示；`SkillNet` 分离单角色技能的直接效果和跨角色效果，并用两名角色的跨角色表示乘积表达技能配合；`MechanismNet` 专门读取分身衰减、召唤影子、技能状态、末尾主动和末尾座位字段。
2. **共享主干**：将属性、两个角色的技能表示和机制表示拼接后送入 1024 维 Mish + LayerNorm 主干、三个残差块和 384 维表示。`base_head` 给出普遍评分，`tail_head` 和 `tail_gate` 只对尾部高分结构提供受控修正。
3. **LifeWheelExpert**：显式构造单角色自身生命、低血量、命轮与分身/护符/护盾等三相组合。它不使用队友血量交换特征，避免把命轮错误建模成队友生命的普通交互；输出受 gate 和残差上限约束。
4. **PairSynergyExpert**：对除血量外的 64 维角色特征构造 `sum`、绝对差和乘积，捕获单个 SkillNet 难以表达的跨角色属性、技能和末尾机制配合。排除血量是为了防止学习“队友血量交换”的伪命轮捷径。

最终输出为：

```text
base + tail_gate * tail + LifeWheelExpert_residual + PairSynergyExpert_residual
```

专家 gate 通过稀疏正则和残差上限限制对普通样本的扰动；高分样本和难例仍可得到足够修正。训练损失在基础 SmoothL1 外加入高分欠估惩罚、低分高估惩罚和专家保持正则，并按高分区间动态采样。这样设计的原因是：普通样本需要稳定的全局回归，高分筛选更关心漏筛风险，而命轮、分身、护符及末尾座位等结构又需要保留明确的机制归因。

## 导出 ONNX

```powershell
python train/export_mech_onnx.py `
  --checkpoint model/reg43d_nt_v0.6.pt `
  --onnx inference/model4.onnx `
  --scale inference/scale.txt
```

`scale.txt` 包含目标反标准化参数以及 130 个输入特征的均值和标准差。推理程序必须使用同一版本的 `model4.onnx` 和 `scale.txt`。

## C++ 推理

`inference/` 已提供可直接运行的 Windows 构建产物。把 `abcp5.5.exe`、`model4.onnx`、`scale.txt` 和 `onnxruntime.dll` 放在同一目录：

```powershell
cd inference
./abcp5.5.exe model4.onnx input.txt scale.txt 8
./abcp5.5.exe model4.onnx input.txt scale.txt 8 4800 result_4800.txt
./result_process4.exe result_4800.txt result_4800_without_score.txt
```

输入可以首行带数量，也可以直接逐行输入二人组；每行也可以带一个旧评分。推理采用流式读取，线程数由第 4 个参数指定，因此无法预先显示总量进度；程序会定期报告已处理/保留/跳过数量，并在读完输入后提示等待队列和写出阶段。`result_process4.exe` 需要完整读取结果后排序，会分别提示读取、排序和写出阶段。

如需重新编译，`inference/build_predict_mech.ps1` 展示了 ONNX Runtime include/lib 的路径约定；需自行安装 ONNX Runtime C++ 开发包和 C++17 编译器。

## 复现与数据限制

源数据来自不同时间的评分程序运行。若相同 130 维特征对应不同实际标签，任何确定性回归模型都无法完全消除该误差；后续补充数据时应先统一评分程序版本、去重规则和原始评分缩放方式，再合并训练。`data_mech.index.tsv` 用于追溯每一行的来源，建议研究新实验时保留该索引。

本仓库中的 `reg43d_nt_v0.6.pt` 和 ONNX 文件是研究基线，不代表对所有新数据的最优结果。修改特征布局时必须同步修改 C++ 生成器、Python 模型和 ONNX/C++ 推理端的标准化逻辑。

## 许可

代码部分采用 MIT License（见 `LICENSE`）。源数据和评分结果仅用于研究与复现，请在公开发布派生数据时保留来源说明，并确认你拥有相应数据的使用权。
