# Marble-Implicit-IR（隐式表示海水观测建模）

概要（中文）
本项目使用“隐式表示 / coordinate-based”方法，直接以原始观测点（时间 t、经度 longitude、纬度 latitude、深度 depth）为输入，预测海水场变量：盐度 so、温度 thetao、速度 uo 和 vo。项目尽量避免对数据进行网格化或插值，默认回退模型为 Fourier-features + MLP 的隐式坐标网络（ImplicitCoordNet）。如果你本地有 Marble 包并想使用它，可在 `model.MarbleWrapper` 中接入 Marble API。

最重要的文件
- train.py           — 按组（date）划分并训练（已实现隐式模型超参暴露）
- model.py           — MarbleWrapper + ImplicitCoordNet（默认）
- data.py            — 数据读取与 OceanDataset（包含 t_numeric、date 处理）
- evaluate.py        — 评估并保存 eval_per_var.csv / eval_per_day.csv
- train_kfold.py     — （可选）按组做 Group K-Fold 交叉验证（若存在）
- group_evaluate.py  — 对指定 group-split 做评估（若存在）
- data/processed_data_mean.csv — 你的观测 CSV（请放置于 data/ 下）
- output/            — 训练与评估输出目录（自动创建）

快速开始（推荐）
1. 创建并激活虚拟环境（推荐）
   - Windows:
     python -m venv venv
     .\venv\Scripts\activate
   - macOS / Linux:
     python -m venv venv
     source venv/bin/activate

2. 安装依赖
   - 推荐先安装合适的 PyTorch（根据你的 CUDA 版本从 PyTorch 官网选择安装命令，例如带 CUDA 的 wheel）；
   - 然后运行：
     pip install -r requirements.txt

   说明（关于 PyTorch）：
   - 如果你要在带 CUDA 的 GPU（如 RTX 4060）上训练，请按照 https://pytorch.org 上的安装建议安装匹配的 torch+cuda 版本（例如 `pip` 命令里选择 CUDA 对应的 wheel）。
   - 若你不确定，可先用 CPU 版（pip install torch torchvision torchaudio）或参照 PyTorch 官网说明。

3. 准备数据
   - 把你的 CSV 放入 `data/processed_data_mean.csv`（文件列需包含 time/date, latitude, longitude, depth, so, thetao, uo, vo）
   - 脚本会自动解析时间并生成 `t_numeric`（以天为单位，相对于数据最早时间）

常用命令示例
- 基本训练（按组划分 test，train/val 在剩余组上划分）：
  python train.py --data data/processed_data_mean.csv --epochs 50 --batch-size 512 --ff-dim 64 --hidden 256 --n-layers 3

- 只用已有 checkpoint 在 group-test 上生成预测（不训练）：
  python train.py --data data/processed_data_mean.csv --epochs 0 --recompute-final

- K-fold group 交叉验证（如果你有 train_kfold.py）：
  python train_kfold.py --data data/processed_data_mean.csv --k 5 --epochs 40 --batch-size 256

建议（调试与显存）
- 如果显存不足（RTX 4060 8GB），请先把 --batch-size 降到 128 或 64，或把 --hidden/--ff-dim 调小（例如 hidden=128, ff-dim=32）。
- 脚本默认启用了 torch.amp（混合精度）以减少显存占用并加速训练。

输出（位于 output/）
- model.pt: 最佳模型 checkpoint（state_dict + scaler params）
- predictions.csv: 测试集合的真实值与预测（date,true_*,pred_*）
- eval_per_var.csv: 每变量 MAE（variable,mae,count）
- eval_per_day.csv: 每天的 MAE（date,mae_so,mae_thetao,mae_uo,mae_vo,mae_overall）
- kfold/: 若运行 K-fold，会在该目录下保存每折输出与汇总

若要使用本地 Marble
- 如果你希望用本地 Marble（非回退隐式网）：
  - 请告诉我 `pip show marble` 输出或 marble 的版本号，或将一个简单的 Marble 模型构造示例贴给我；
  - 我会把 `model.MarbleWrapper` 的 TODO 区替换为可用的 Marble 初始化与前向逻辑。

常见问题（FAQ）
- 时间解析失败：请确保时间列格式一致（建议 ISO 格式 YYYY-MM-DD 或 YYYY/MM/DD）。
- 检查 data/processed_data_mean.csv 路径：若报错找不到文件，请确认当前终端工作目录为项目根或使用绝对路径传入 `--data`。
- 数据缺失：`data.py` 会自动丢弃缺失关键字段的行并输出警告；若删除过多，请先检查 CSV。

联系方式
- 需要我进一步把 Marble wrapper 绑定到你本地 marble 版本，或添加早停 / LR 调度 / logging（TensorBoard），把 marble 版本或希望的功能告诉我，我会继续修改。
