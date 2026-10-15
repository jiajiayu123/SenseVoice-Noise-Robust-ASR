# 模型训练说明

## 设置环境变量

export ASR_PROJECT_DIR="$(pwd)"
export SENSEVOICE_MODEL_DIR="/path/to/SenseVoiceSmall"

## Clean-3835

python train_awgn.py --config-path ./configs --config-name sensevoice_clean_multi

## AWGN-3835

python train_awgn.py --config-path ./configs --config-name sensevoice_awgn

## Low-SNR

python train_awgn.py --config-path ./configs --config-name sensevoice_awgn_lowsnr

## 说明

训练前需要准备 AISHELL-1 数据集、训练和验证 JSONL 清单、SenseVoiceSmall 预训练模型及分词文件。

以上命令参考已完成的原始实验。GitHub 版本尚未重新执行完整训练验证。
