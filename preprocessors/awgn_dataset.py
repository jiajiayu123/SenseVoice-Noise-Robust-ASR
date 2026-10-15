from funasr.register import tables
from funasr.datasets.sense_voice_datasets.datasets import SenseVoiceCTCDataset


@tables.register("dataset_classes", "SenseVoiceAWGNDataset")
class SenseVoiceAWGNDataset(SenseVoiceCTCDataset):

    def __init__(self, *args, is_training=True, **kwargs):

        if not is_training:
            # 验证集禁用 AWGN，防止验证数据被随机加噪
            kwargs.pop("preprocessor_speech", None)
            kwargs.pop("preprocessor_speech_conf", None)

        super().__init__(
            *args,
            is_training=is_training,
            **kwargs
        )
