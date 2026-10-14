# EFW-YOLO

**A lightweight early fire detection network with a channel-permutation backbone and large-kernel attention for resource-constrained microcontrollers**

面向火灾早期预警的终端设备往往受限于算力、存储和内存。本文提出 EFW-YOLO，研究如何在压缩检测网络的同时维持火焰检测性能，并将模型部署到微控制器。网络以 YOLO26n 为基础，采用 0.5× ShuffleNetV2 主干、SRSM 与 MFMN 特征融合结构和大核平均注意力；部署阶段结合通道置换优化与 INT8 量化。

## 实验设置

实验使用单类火焰数据集，共 2,384 张图像，按 7:2:1 划分为训练集 1,669 张、验证集 477 张和测试集 238 张。输入尺寸为 512×512。训练采用 YOLO26n、one-to-many 检测头、batch size 48、1,000 个 epoch、MuSGD 优化器，初始学习率为 0.01，weight decay 为 0.0005。

论文实验在远端 Windows 服务器上使用 PyTorch 2.8.0 和 NVIDIA RTX 5060 Ti 训练。部署目标为 360 MHz 双核 RISC-V、32 MB PSRAM 且没有 NPU 的 ESP32-P4。

## 论文报告结果

| 指标 | YOLO26n 基线 | EFW-YOLO |
|---|---:|---:|
| 参数量 | 2.38 M | 0.34 M |
| 计算量 | 3.4 GFLOPs | 0.9 GFLOPs |
| Precision | 95.87% | 95.18% |
| Recall | 82.95% | 85.16% |
| mAP@0.5 | 90.12% | 89.05% |

论文还报告了 ESP32-P4 上 0.63 MB 的 INT8 模型文件、3.87 MB 峰值运行内存和 1,859.87 ms 网络推理延迟。以上数值均为论文报告值。

## 环境建议

复现实验建议使用 PyTorch 2.8.0 和 CUDA GPU；NVIDIA RTX 5060 Ti 与论文训练环境一致，使用同一型号便于对照实验设置。若更换显卡，应记录硬件与软件版本，并保持数据划分和训练参数一致。

板端部署面向 ESP32-P4，论文使用 ESP-PPQ 和 ESP-DL 工具链。运行训练脚本时，需在命令中指定数据配置和输出目录。

## 代码目录

- `configs/models/`：基线、EFW-YOLO、消融和对比模型配置。
- `efw_yolo/`：模型模块与 ONNX 图优化实现。
- `scripts/`：训练、评估、模型检查、ONNX 导出和图优化入口。
- `tests/`：模型结构、注意力模块、导出与训练流程测试。
