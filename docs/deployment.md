# ESP32-P4 部署流程

论文将 EFW-YOLO 导出为 ONNX，针对微控制器推理调整计算图，再使用 ESP-PPQ 和 ESP-DL 进行 INT8 量化与部署。本页说明仓库中提供的 ONNX 准备入口。

## 导出 ONNX

使用训练得到的权重导出模型：

```powershell
python scripts/export_onnx.py --weights path/to/best.pt --output artifacts/efw_yolo.onnx --imgsz 512 --device cpu
```

导出图包含三个尺度的 box 和 score 输出。后处理需要完成边界框解码、分类分数处理和 NMS。没有训练权重时，只有显式传入 `--random-init` 才会使用随机初始化模型；此模型用于检查导出流程。

## 通道置换优化

```powershell
python scripts/optimize_shuffle.py --input artifacts/efw_yolo.onnx --output artifacts/efw_yolo_optimized.onnx
```

脚本会处理静态图中的 ShuffleNet 通道置换：符合条件的置换可并入后续卷积权重，其他置换以通道 Gather 表示。优化后的 ONNX 可作为后续量化流程的输入。

## INT8 量化与板端运行

按论文流程，使用 ESP-PPQ 执行 PTQ 与 TQT 量化，再由 ESP-DL 生成并运行 ESP32-P4 所需的 `.espdl` 模型。校准数据、量化配置、ESP-IDF 工程和板端固件需在对应工具环境中准备；本仓库的上述脚本负责 ONNX 导出与图优化。
