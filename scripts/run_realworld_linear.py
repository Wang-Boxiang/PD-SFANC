import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from scipy.signal import lfilter, welch

from pd_sfanc import CRNNAzimuthClassifier, PredictiveFixedFilterController
from pd_sfanc.features import compute_stft_for_all_channels
from pd_sfanc.geometry import (
    azimuth_centers,
    azimuth_to_class,
    circular_error_deg,
    make_linear_trajectory,
)
from pd_sfanc.signal_utils import load_demo_noise, segment_noise_reduction_db
from pd_sfanc.simulation import simulate_moving_source


def load_state_dict(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def predict_filter_sequence(reference, segments, model, device, frame_length, context_frames, n_fft, hop):
    centers = azimuth_centers(36)
    pred_classes = []
    true_classes = []
    pred_azimuths = []
    errors = []
    for frame_index in range(context_frames, len(segments)):
        frames_per_channel = [[], [], [], []]
        for source_frame in range(frame_index - context_frames, frame_index):
            frame = reference[source_frame * frame_length : (source_frame + 1) * frame_length, :]
            for channel in range(4):
                frames_per_channel[channel].append(torch.from_numpy(frame[:, channel].astype(np.float32)))
        features = compute_stft_for_all_channels(
            frames_per_channel,
            n_fft=n_fft,
            hop_length=hop,
            center=False,
        )
        with torch.no_grad():
            logits = model(features.unsqueeze(0).to(device))
            pred_class = int(torch.argmax(logits, dim=1).item())
        pred_deg = float(centers[pred_class])
        true_deg = float(segments[frame_index]["azimuth"])
        true_class = azimuth_to_class(true_deg, centers)
        pred_classes.append(pred_class)
        true_classes.append(true_class)
        pred_azimuths.append(pred_deg)
        errors.append(circular_error_deg(pred_deg, true_deg))
    warmup_prefix = [pred_azimuths[0]] * context_frames
    return {
        "pred_classes": np.asarray(pred_classes, dtype=int),
        "true_classes": np.asarray(true_classes, dtype=int),
        "pred_azimuths": np.asarray(warmup_prefix + pred_azimuths, dtype=float),
        "errors_deg": np.asarray(errors, dtype=float),
    }


def load_control_filters(pred_azimuths, control_filter_dir: Path):
    filters = []
    for azimuth in pred_azimuths:
        path = control_filter_dir / f"CF_0.4_0_{azimuth:.0f}.mat"
        filters.append(sio.loadmat(path)["W"])
    return np.stack(filters, axis=0)


def plot_results(output_path, fs, disturbance, error, segments, pred_azimuths, noise_reduction):
    time = np.arange(len(disturbance)) / fs
    frame_time = np.arange(len(segments)) * 0.5
    true_azimuths = np.asarray([segment["azimuth"] for segment in segments], dtype=float)
    selected = pred_azimuths.copy()
    selected[:4] = np.nan
    f_dist, p_dist = welch(disturbance, fs, nperseg=4096)
    f_err, p_err = welch(error, fs, nperseg=4096)
    p_dist_db = 10.0 * np.log10(p_dist + 1e-12)
    p_err_db = 10.0 * np.log10(p_err + 1e-12)

    fig = plt.figure(figsize=(14, 9), dpi=200)
    gs = fig.add_gridspec(2, 2)
    ax_time = fig.add_subplot(gs[0, 0])
    ax_psd = fig.add_subplot(gs[0, 1])
    ax_doa = fig.add_subplot(gs[1, 0])
    ax_nr = fig.add_subplot(gs[1, 1])

    ax_time.plot(time, disturbance, label="ANC off", linewidth=1.0)
    ax_time.plot(time, error, label="PD-SFANC", linewidth=1.0)
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Error signal")
    ax_time.set_xlim(0, time[-1])
    ax_time.grid(True, linestyle="--", linewidth=0.5)
    ax_time.legend()

    ax_psd.plot(f_dist, p_dist_db, label="ANC off", linewidth=1.2)
    ax_psd.plot(f_err, p_err_db, label="PD-SFANC", linewidth=1.2)
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD (dB/Hz)")
    ax_psd.set_xlim(0, 1000)
    ax_psd.grid(True, linestyle="--", linewidth=0.5)
    ax_psd.legend()

    ax_doa.step(frame_time, selected, where="post", label="Selected filter DoA", linewidth=1.5)
    ax_doa.scatter(frame_time + 0.25, true_azimuths, marker="x", s=25, label="True DoA")
    ax_doa.set_xlabel("Time (s)")
    ax_doa.set_ylabel("Azimuth (deg)")
    ax_doa.set_xlim(0, time[-1])
    ax_doa.grid(True, linestyle="--", linewidth=0.5)
    ax_doa.legend()

    ax_nr.step(frame_time, noise_reduction, where="post", label="PD-SFANC", linewidth=1.5)
    ax_nr.set_xlabel("Time (s)")
    ax_nr.set_ylabel("Noise reduction (dB)")
    ax_nr.set_xlim(0, time[-1])
    ax_nr.grid(True, linestyle="--", linewidth=0.5)
    ax_nr.legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_demo(root=None, output_dir=None, device=None):
    root = (Path(__file__).resolve().parents[1] if root is None else Path(root)).resolve()
    output_dir = Path(output_dir) if output_dir is not None else (root / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    fs = 16000
    duration_per_segment = 0.5
    total_duration = 20.0
    frame_length = int(duration_per_segment * fs)
    total_samples = int(total_duration * fs)
    context_frames = 4
    n_fft = 1024
    hop = 64

    noise = load_demo_noise(
        root / "data" / "finaltest_noises" / "1-19840-A-36.wav",
        target_fs=fs,
        target_length=total_samples,
        cutoff_hz=1000,
        target_rms=0.3,
    )
    segments = make_linear_trajectory(
        n_frames=int(total_duration / duration_per_segment),
        start_angle=2.0,
        angular_step_deg=10.0,
        distance=0.4,
        elevation=0.0,
    )
    reference, disturbance = simulate_moving_source(
        noise,
        segments,
        fs=fs,
        duration_per_segment=duration_per_segment,
        snr_db=30.0,
        target_rms=0.3,
        rng=np.random.default_rng(1234),
    )

    model = CRNNAzimuthClassifier(
        in_channels=8,
        num_azimuth=36,
        freq_bins=n_fft // 2 + 1,
        cnn_channels=(16, 32, 64),
        gru_hidden=64,
        gru_layers=1,
        bidirectional=False,
    )
    model.load_state_dict(load_state_dict(root / "models" / "CRNN_interspeech.pth", device))
    model.to(device).eval()

    prediction = predict_filter_sequence(
        reference,
        segments,
        model,
        device,
        frame_length=frame_length,
        context_frames=context_frames,
        n_fft=n_fft,
        hop=hop,
    )
    control_filters = load_control_filters(
        prediction["pred_azimuths"],
        root / "data" / "Pre_trained_CFs_2khz_causal",
    )
    secondary_path = np.squeeze(sio.loadmat(root / "data" / "SecondaryPath_final_correct_casual.mat")["Sec_path_cal"])
    filtered_reference = np.zeros_like(reference)
    for channel in range(4):
        filtered_reference[:, channel] = lfilter(secondary_path, 1, reference[:, channel])

    controller = PredictiveFixedFilterController(
        frame_length=frame_length,
        control_filters=control_filters,
        num_channels=4,
        filter_length=1024,
        warmup_frames=context_frames,
    )
    error = controller.noise_cancellation(disturbance[:, 0], filtered_reference)
    noise_reduction = segment_noise_reduction_db(disturbance[:, 0], error, frame_length)

    top1 = float(np.mean(prediction["pred_classes"] == prediction["true_classes"]))
    mae = float(np.mean(prediction["errors_deg"]))
    summary = {
        "segments_evaluated": int(len(prediction["pred_classes"])),
        "warmup_frames": context_frames,
        "top1_accuracy": top1,
        "angular_mae_deg": mae,
        "mean_noise_reduction_db_all": float(np.mean(noise_reduction)),
        "mean_noise_reduction_db_after_warmup": float(np.mean(noise_reduction[context_frames:])),
        "min_noise_reduction_db_after_warmup": float(np.min(noise_reduction[context_frames:])),
    }

    per_frame = pd.DataFrame(
        {
            "frame": np.arange(len(segments)),
            "time_s": np.arange(len(segments)) * duration_per_segment,
            "true_azimuth_deg": [segment["azimuth"] for segment in segments],
            "selected_filter_azimuth_deg": prediction["pred_azimuths"],
            "noise_reduction_db": noise_reduction,
        }
    )
    per_frame.to_csv(output_dir / "realworld_linear_per_frame.csv", index=False)
    with open(output_dir / "realworld_linear_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    figure_path = output_dir / "realworld_linear_pd_sfanc.png"
    plot_results(
        figure_path,
        fs,
        disturbance[:, 0],
        error,
        segments,
        prediction["pred_azimuths"],
        noise_reduction,
    )
    return {
        "summary": summary,
        "summary_path": output_dir / "realworld_linear_summary.json",
        "per_frame_path": output_dir / "realworld_linear_per_frame.csv",
        "figure_path": figure_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    result = run_demo(args.root, args.output_dir, args.device)

    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
