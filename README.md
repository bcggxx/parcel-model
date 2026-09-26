# parcel-model

ParcelLens 快递取件码 OCR 模型的训练工程：全合成数据 → YOLO11n 检测 + CNN/BiLSTM/CTC 识别 → ONNX 导出 → 端到端验证。训练产物通过 GitHub Release 分发，Android 端（ParcelLens）下载后用 ONNX Runtime 离线推理。

## 模型卡片（v1，2026-09-21）

| 模型 | 文件 | 大小 | 输入 | 说明 |
|---|---|---|---|---|
| 检测 | `parcel_det.onnx` | 10 MB | 640×640 RGB letterbox | YOLO11n，输出取件码文本行框，conf 阈值 0.4 + 贪心 NMS(IoU 0.5) |
| 识别 | `parcel_rec.onnx` | 7.9 MB | 灰度 高32 × 宽动态(≤192，不足右填充) | CNN+BiLSTM+CTC，字符集 `0-9 A-Z - 空格`（38 类 + blank） |

- 识别验证集整码准确率 **99.67%**（CER 0.0005）
- 端到端（400 张整图，det→crop→rec）：精确率 **72.0%**，宽松率 73.25%；det 召回 398/400
- 模型下载：[本仓库 Release ocr-models-v1](https://github.com/bcggxx/parcel-model/releases/tag/ocr-models-v1)（含 SHA256）
- fp32 未量化：合计 <20 MB，2018 年后的 arm64 手机可流畅运行，量化收益小于精度风险

## 流水线

```
synth_labels.py          # 生成合成快递单（检测集：整图+YOLO 标签；识别集：裁剪行+labels.txt）
prep_det.py              # 检测集 95/5 划分成 data/det_yolo，写 det.yaml
train_det.py             # YOLO11n CPU 微调，导出 export/det_run/weights/best.onnx
train_rec.py             # CRNN+CTC 训练（含 ±6° 旋转增强），导出 export/rec_model.onnx
e2e_onnx.py <整图目录>    # 纯 ONNX 端到端验证，与 Android 端 OnnxOcrEngine 逻辑一致
eval_rec_onnx.py         # 仅识别模型的离线评估
```

## 本地复现

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python scripts/synth_labels.py --n-det 6000 --n-rec 30000   # data/synth
python scripts/prep_det.py                                   # data/det_yolo
python scripts/train_det.py --epochs 25                      # 约 3h @ 12 线程 CPU
python scripts/train_rec.py --epochs 15                      # 约 1.5h @ 12 线程 CPU

python scripts/synth_labels.py --n-det 400 --out data/synth_e2e --seed 777
python scripts/e2e_onnx.py data/synth_e2e
```

## 云端训练（GitHub Actions）

无需本地算力，Actions 页 → `train-models` → Run workflow 即可，参数（图片数/轮数/种子）均可调：

1. `data`：合成数据并打包为 artifact（≤3h）
2. `train-det` / `train-rec`：并行训练（公开仓库 runner 为 4 vCPU，单 job 上限 6h，默认轮数已留余量）
3. `release`：自动跑 400 张端到端验证（种子 777，与训练集独立），把两个 ONNX + 校验和 + 验证报告发到本仓库 Release

仓库为公开仓库：GitHub 标准 runner 对公开仓库免费且不限量，无需关心每月额度；单 job 上限 6 小时，若加大数据量导致超限，优先减少 `det_epochs`（检测收敛快，15 轮足够）。

## 数据说明

- 全部为程序合成的快递面单图：字体来自 `fonts/`（BarlowCondensed / JetBrainsMono / NotoSansSC / Oswald / RobotoMono），随机背景、噪点、模糊、透视与 ±6° 旋转
- 取件码格式覆盖：`4-5-1283`、`M10-662-75`、`S598271`、`U98-34-29` 等常见驿站编码模式
- `data/`、`export/` 不入库（见 .gitignore），可随时重新生成

## 目录

```
scripts/    训练与验证脚本
fonts/      合成用字体
.github/    云端训练流水线
```
