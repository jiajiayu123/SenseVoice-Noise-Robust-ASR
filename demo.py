import argparse
import csv
import json
import re
from pathlib import Path

import torch
import torchaudio
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


BASE_DIR = Path(__file__).resolve().parent
BASE_MODEL = (
    Path.home()
    / ".cache/modelscope/models/iic--SenseVoiceSmall/snapshots/master"
)
FINETUNED_MODEL = (
    BASE_DIR
    / "exp/sensevoice_awgn_lowsnr/model_lowsnr_finetuned.pt"
)


def normalize_text(text):
    return re.sub(
        r"[^\u4e00-\u9fffA-Za-z0-9]", "", text
    ).lower()


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (char_a != char_b)
            ))
        previous = current
    return previous[-1]


def add_awgn(waveform, snr_db, seed):
    generator = torch.Generator().manual_seed(seed)

    noise = torch.randn(
        waveform.shape,
        generator=generator,
        dtype=waveform.dtype
    )
    noise = noise / noise.pow(2).mean().sqrt().clamp_min(1e-12)

    signal_power = waveform.pow(2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10))

    return waveform + noise * noise_power.sqrt()


def recognize(model, audio):
    result = model.generate(
        input=audio,
        language="zh",
        use_itn=False,
        batch_size=1
    )
    return rich_transcription_postprocess(result[0]["text"])


def main():
    parser = argparse.ArgumentParser(
        description="SenseVoiceSmall 噪声鲁棒语音识别 Demo"
    )
    parser.add_argument("--audio", required=True, help="WAV 音频路径")
    parser.add_argument(
        "--snr", type=float, default=None,
        help="AWGN 信噪比，省略时不加噪"
    )
    parser.add_argument(
        "--reference", type=str, default=None,
        help="正确文本，可选；提供后计算单条 CER"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=str(BASE_MODEL),
        help="SenseVoiceSmall 基础模型目录"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(FINETUNED_MODEL),
        help="微调模型权重路径"
    )
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()

    if not audio_path.is_file():
        raise FileNotFoundError(f"音频不存在: {audio_path}")

    base_model_path = Path(args.base_model).expanduser().resolve()
    if not base_model_path.is_dir():
        raise FileNotFoundError(
            f"基础模型不存在: {base_model_path}"
        )

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Low-SNR 权重不存在: {checkpoint}"
        )

    waveform, sample_rate = torchaudio.load(str(audio_path))

    if sample_rate != 16000 or waveform.shape[0] != 1:
        raise ValueError("当前 Demo 要求 16 kHz、单声道 WAV")

    if args.snr is not None:
        waveform = add_awgn(waveform, args.snr, args.seed)

    audio = waveform.squeeze(0).numpy()

    output_dir = BASE_DIR / "demo_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_audio = output_dir / "input_processed.wav"
    torchaudio.save(
        str(output_audio), waveform, sample_rate
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("推理设备:", device)
    print("输入音频:", audio_path)
    print("噪声条件:", "Clean" if args.snr is None else f"{args.snr} dB")

    print("\n[1/2] 加载原始 SenseVoiceSmall...")
    base_model = AutoModel(
        model=str(base_model_path),
        device=device,
        trust_remote_code=False,
        disable_update=True
    )
    base_text = recognize(base_model, audio)

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n[2/2] 加载 Low-SNR 微调模型...")
    low_model = AutoModel(
        model=str(base_model_path),
        init_param=str(checkpoint),
        device=device,
        trust_remote_code=False,
        disable_update=True
    )
    low_text = recognize(low_model, audio)

    report = {
        "audio": str(audio_path),
        "snr_db": args.snr,
        "seed": args.seed,
        "base_prediction": base_text,
        "lowsnr_prediction": low_text,
    }

    print("\n========== 推理结果 ==========")
    print("原始模型:", base_text)
    print("Low-SNR :", low_text)

    if args.reference is not None:
        reference = normalize_text(args.reference)
        base_pred = normalize_text(base_text)
        low_pred = normalize_text(low_text)

        if not reference:
            raise ValueError("正确文本归一化后为空")

        base_cer = edit_distance(reference, base_pred) / len(reference)
        low_cer = edit_distance(reference, low_pred) / len(reference)

        report["reference"] = args.reference
        report["base_cer"] = base_cer
        report["lowsnr_cer"] = low_cer

        print("正确文本:", args.reference)
        print(f"原始 CER: {base_cer:.2%}")
        print(f"Low-SNR CER: {low_cer:.2%}")

    output_json = output_dir / "demo_result.json"
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\n处理后音频:", output_audio)
    print("结果文件:", output_json)


if __name__ == "__main__":
    main()
