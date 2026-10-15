import runpy

# 必须先导入自定义模块，完成 FunASR 注册
import preprocessors.awgn_preprocessor
import preprocessors.awgn_dataset

from funasr.register import tables

assert "SpeechPreprocessAWGN" in tables.preprocessor_classes
assert "SenseVoiceAWGNDataset" in tables.dataset_classes

print("AWGN 预处理器与数据集注册成功", flush=True)

# 调用 FunASR 原有训练入口
runpy.run_module(
    "funasr.bin.train_ds",
    run_name="__main__",
)
