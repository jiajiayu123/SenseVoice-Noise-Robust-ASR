import argparse
import csv
import re
from pathlib import Path

import torch
import torchaudio
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


def normalize_text(text):
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text).lower()


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
    target_noise_power = signal_power / (10 ** (snr_db / 10))

    return waveform + noise * target_noise_power.sqrt()


def main():
    parser = argparse.ArgumentParser(
        description="SenseVoiceSmall 多信噪比 CER 评估"
    )
    parser.add_argument("--manifest", required=True, help="测试集 CSV")
    parser.add_argument(
        "--base-model",
        default="iic/SenseVoiceSmall",
        help="基础模型目录或 ModelScope 模型名称"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="微调模型权重；不提供则评估原始模型"
    )
    parser.add_argument(
        "--output",
        default="results/evaluation",
        help="输出文件前缀"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = Path(args.manifest).expanduser().resolve()

    with manifest.open(encoding="utf-8-sig", newline="") as f:
        samples = list(csv.DictReader(f))

    required = {"audio_path", "reference"}
    if not samples or not required.issubset(samples[0]):
        raise ValueError("CSV 至少需要 audio_path 和 reference 两列")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_kwargs = {
        "model": args.base_model,
        "device": device,
        "trust_remote_code": False,
        "disable_update": True,
    }

    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model_kwargs["init_param"] = str(checkpoint)

    print("当前设备:", device)
    print("测试语音数量:", len(samples))
    print("模型权重:", args.checkpoint or "原始预训练模型")

    model = AutoModel(**model_kwargs)

    output_prefix = Path(args.output)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    details_file = Path(str(output_prefix) + "_details.csv")
    summary_file = Path(str(output_prefix) + "_summary.csv")

    snr_list = [None, 20, 10, 5, 0, -5]
    summary_rows = []

    with details_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "snr", "audio_id", "speaker", "reference",
            "prediction", "errors", "reference_chars", "cer"
        ])

        for snr_db in snr_list:
            snr_label = "Clean" if snr_db is None else str(snr_db)
            total_errors = 0
            total_chars = 0

            print(f"\n========== {snr_label} ==========")

            for index, sample in enumerate(samples):
                audio_path = Path(sample["audio_path"]).expanduser()

                if not audio_path.is_absolute():
                    audio_path = manifest.parent / audio_path
                    if not audio_path.is_file():
                        audio_path = Path.cwd() / sample["audio_path"]

                waveform, sample_rate = torchaudio.load(str(audio_path))

                if sample_rate != 16000 or waveform.shape[0] != 1:
                    raise ValueError(
                        f"音频必须为 16kHz 单声道: {audio_path}"
                    )

                if snr_db is not None:
                    waveform = add_awgn(
                        waveform, snr_db, args.seed + index
                    )

                result = model.generate(
                    input=waveform.squeeze(0).numpy(),
                    language="zh",
                    use_itn=False,
                    batch_size=1
                )

                prediction = normalize_text(
                    rich_transcription_postprocess(result[0]["text"])
                )
                reference = normalize_text(sample["reference"])

                errors = edit_distance(reference, prediction)
                total_errors += errors
                total_chars += len(reference)

                audio_id = sample.get("audio_id") or audio_path.stem
                speaker = sample.get("speaker") or audio_path.parent.name

                cer = errors / len(reference) if reference else 0.0

                writer.writerow([
                    snr_label, audio_id, speaker, reference,
                    prediction, errors, len(reference), f"{cer:.6f}"
                ])

                if (index + 1) % 25 == 0:
                    print(f"已完成 {index + 1}/{len(samples)} 条")

            overall_cer = (
                total_errors / total_chars * 100
                if total_chars else 0.0
            )

            summary_rows.append({
                "snr": snr_label,
                "samples": len(samples),
                "total_errors": total_errors,
                "total_chars": total_chars,
                "cer_percent": f"{overall_cer:.4f}"
            })

            print(f"{snr_label}: CER = {overall_cer:.4f}%")

    with summary_file.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "snr", "samples", "total_errors",
                "total_chars", "cer_percent"
            ]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n评估完成")
    print("详细结果:", details_file)
    print("汇总结果:", summary_file)


if __name__ == "__main__":
    main()
