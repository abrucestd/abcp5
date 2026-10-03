# abcp5.5：二人组评分模型研究资料

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

1. **角色特征子网络**：先按角色切分为 `role1=f000..f064` 和 `role2=f065..f129`。每个角色的 `f032`（分身/幻影相关主技能）与 `f045..f053`（幻影属性、血量和附体技能）组成 10 维输入，送入 `IllusionNet(10 -> 64 -> 12)`；它把不同幻影强度和召唤形态压缩成可供后续网络使用的表示。`SkillNet` 使用 `f008..f031` 与 `f033..f044` 的 36 个技能/状态字段，再拼接 12 维幻影表示，得到 48 维输入。其共享 trunk 为 `48 -> 288 -> 288`，分成 160 维 `direct_head`（该角色自身技能效果）和 80 维 `cross_head`（可与另一角色配合的潜表示）；两名角色的 `cross_head` 逐元素相乘，形成显式的技能配合通道。
2. **机制子网络 `MechanismNet`**：将难以由最终技能强度反推出的结构单独保留。`clone_tail`（32 维）读取生命、技能状态和 `f054..f064` 的末尾主动/末尾座位字段；`summon_shadow`（18 维）读取两名角色的幻影字段；`state_skill`（36 维）读取两名角色的状态技能字段。三个分支分别编码为 32、20、28 维，再融合为 24 维，因此分身衰减、末尾原始熟练度、bonus、座位规则和召唤影子不会在早期压缩时丢失。
3. **共享主干**：输入维度为 `16 + 160*2 + 80 + 24 + 12*2 = 464`（两角色基础属性、两路 direct、cross 乘积、机制表示和两路幻影表示）。主干为 `464 -> 1024` 的 Linear + LayerNorm + Mish，接三个 1024 维残差块，再压缩到 384 维。`base_head` 输出普遍评分；`tail_head(384 -> 128 -> 1)` 与 `tail_gate` 共同组成尾部修正，gate 让普通样本不会无条件受到尾部机制支路影响。
4. **LifeWheelExpert**：从每名角色的标准化字段重新构造“自身状态”特征：生命值、低血量、生命与属性/技能的乘积，以及命轮与低血量、分身、护符、护盾的二相/三相/四相组合，共 58 维。它可以读取少量队友技能状态，但明确不使用队友血量；二人组输入代表己方两名角色，而不是命轮的敌方交换对象。这样模型学习的是“命轮持有者自身因分身/护符导致的低血量结构”，而不是错误的队友血量交换。`LifeWheelExpert` 的 64 维隐藏层、sigmoid gate 和 `tanh` 残差上限共同限制最大修正量。
5. **PairSynergyExpert**：去掉血量后的每角色 64 维特征先做 `tanh`，然后构造 `role1+role2`、`abs(role1-role2)` 和 `role1*role2`，得到 `64*3=192` 维的顺序对称输入。它通过 `192 -> 96 -> 48` 的 trunk 和独立 gate 学习属性、技能、幻影及末尾机制之间的稀有跨角色配合。这里排除血量是刻意的：命轮的血量交换对象是敌人，不是队友，不能让通用二人组分支把队友血量差当成命轮信号。

最终输出为：

```text
base + tail_gate * tail + LifeWheelExpert_residual + PairSynergyExpert_residual
```

专家 gate 通过稀疏正则和残差上限限制对普通样本的扰动；高分样本和难例仍可得到足够修正。训练损失在基础 SmoothL1 外加入高分欠估惩罚、低分高估惩罚和专家保持正则，并按高分区间动态采样。这样设计的原因是：普通样本需要稳定的全局回归，高分筛选更关心漏筛风险，而命轮、分身、护符及末尾座位等结构又需要保留明确的机制归因。

模型在训练时先对 130 个输入特征逐列标准化，目标值也按训练集均值和标准差标准化；checkpoint 保存的 `x_mean/x_std/y_bias/y_scale` 会被导出到 `scale.txt`。因此 ONNX 和 C++ 端看到的不是原始属性数值，而是与训练完全相同的标准化数值。`base_output`、尾部 gate、两个专家残差都在标准化目标空间中相加，最后统一反标准化为评分。

## tester 的分析内容

`tester4_new_mech.py` 不参与训练，而是用 checkpoint 对一个或多个 CSV 做离线审计。它读取 CSV 第一列真实评分和后面的 130 个特征，使用 checkpoint 的标准化参数批量推理，然后计算 `error = pred - actual`。默认会进行以下分析：

- **总体误差**：样本数、MAE、RMSE、最大正误差和最小负误差；同时单独报告实际评分 `>=4650/4750/4850` 的 MAE、最小误差和 P10，以及实际评分 `<=4150` 的高估率和 P95 高估量。
- **筛选安全性**：对实际评分 `>=4450、4550、4650、4750、4850、4950` 的样本，计算该组真实高分样本中的最低预测值 `safe_pred_min`。以它作为预测阈值时，统计需要保留的总样本数 `selected`、筛选倍率 `ratio` 和高分召回率 `recall`。另有固定预测阈值表，便于比较实际 `>=4650/4750` 的命中数和低分分位数。
- **分段偏差**：对 `4650~4750`、`4750~4850`、`4850~4950`、`>=4950` 四个区间分别统计偏差均值、MAE、中位数、P10/P90。每个区间同时按“实际分数落入该段”和“预测分数落入该段”画图，前者观察真实高分是否被低估，后者观察预测高分中混入了多少低分样本。
- **极端样本导出**：`segment_min_samples_new.csv` 保存每个高分起点（默认每 50 分）下预测最低的样本；`actual_ge4650_min_error50_new.csv` 保存实际 `>=4650` 中误差最负的前 50 个样本，用于研究虚低；`pred_ge4650_max_error50_new.csv` 保存预测 `>=4650` 中误差最正的前 50 个样本，用于研究虚高。导出的 CSV 除误差和索引外还包含带语义的 `f000_p1_血量`、技能、幻影和末尾机制字段，便于直接观察命轮、分身、护符、bonus 等特征组合。
- **性能记录**：输出推理耗时、每样本耗时和总耗时；`--no_plot` 可关闭图形窗口但仍会生成分段统计图文件。

典型分析命令：

```powershell
cd train
python tester4_new_mech.py --model ../model/reg43d_nt_v0.6.pt `
  --npy ../data/generated/data_mech.csv `
  --actual4650_min_error_out ../actual_ge4650_min_error50_new.csv `
  --pred4650_max_error_out ../pred_ge4650_max_error50_new.csv `
  --segment_min_out ../segment_min_samples_new.csv --no_plot
cd ..
```

`tester` 报告中的 `global_index` 是传给 `--npy` 的拼接数组中的 **0 基** 行号；报告中的 `source_file/source_row` 是 tester 读入的 CSV 路径和 **1 基** CSV 行号，不是原始 txt 的行号。若只传入 `data_mech.csv`，则可以用 `csv_row = global_index + 1` 查询对应的索引记录。

## 使用 `data_mech.index.tsv` 追溯误差样本

`data_mech.csv` 只保存目标值和 130 个数值特征，模型误差较大时仅凭这些数字无法知道该二人组来自哪个源文件。生成器同步写出的 `data_mech.index.tsv` 为每个 CSV 行保存以下字段：

| 字段 | 含义 |
| --- | --- |
| `csv_row` | `data_mech.csv` 的 1 基行号（无表头 CSV 的第一行是 1） |
| `source_file` | 原始源文件名，例如 `ge4500_2.txt` |
| `source_record` | 该源文件中记录的 1 基序号 |
| `source_line` | 源 txt 的物理行号，通常为 `source_record + 1`（第 1 行是计数） |
| `raw_score` | 源文件中的原始评分；CSV 第一列通常是它乘以 100 后的值 |
| `duo` | 原始二人组文本，可直接复制回评分程序或实验输入 |

例如先找到最严重的实际高分低估样本，再查询其来源：

```powershell
$bad = Import-Csv .\actual_ge4650_min_error50_new.csv |
  Sort-Object {[double]$_.error} | Select-Object -First 1
$index = Import-Csv .\data\generated\data_mech.index.tsv -Delimiter "`t"
$origin = $index[[int]$bad.global_index]
$bad | Select-Object global_index,actual_score,pred_score,error,source_file,source_row
$origin | Format-List
```

`$origin.source_record` 和 `$origin.source_line` 可以定位到 `data/source/$($origin.source_file)` 的原始记录，`$origin.duo` 则是需要重新拆分、替换队友或运行底层评分器时使用的二人组。若 `tester` 的 `--npy` 同时传入多个 CSV，`global_index` 是所有文件拼接后的行号，需要先加上前面文件的行数，或直接使用导出报告中的 `source_file/source_row` 再在对应 CSV 的索引文件中查询。索引文件因此连接了“模型误差统计”与“可复现实验输入”：可以从一个异常样本回到原始名称、原始评分和生成时使用的源数据版本。

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
