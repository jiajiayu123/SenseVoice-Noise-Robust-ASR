import random

import torch
from funasr.register import tables


@tables.register("preprocessor_classes", "SpeechPreprocessAWGN")
class SpeechPreprocessAWGN:
    def __init__(
        self,
        snr_min=-5,
        snr_max=20,
        clean_prob=0.2,
    ):
        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)
        self.clean_prob = float(clean_prob)

        if self.snr_min > self.snr_max:
            raise ValueError("snr_min 不能大于 snr_max")

        if not 0 <= self.clean_prob <= 1:
            raise ValueError("clean_prob 必须在 0 到 1 之间")

    def __call__(self, waveform, fs=None):
        if random.random() < self.clean_prob:
            return waveform

        is_tensor = isinstance(waveform, torch.Tensor)
        signal = waveform if is_tensor else torch.as_tensor(waveform)
        signal = signal.float()

        snr_db = random.uniform(self.snr_min, self.snr_max)

        noise = torch.randn_like(signal)
        noise_power = noise.square().mean().clamp_min(1e-12)
        noise = noise / noise_power.sqrt()

        signal_power = signal.square().mean()

        target_noise_power = signal_power / (10 ** (snr_db / 10))

        noisy_signal = signal + noise * target_noise_power.sqrt()

        if is_tensor:
            return noisy_signal

        return noisy_signal.numpy()
