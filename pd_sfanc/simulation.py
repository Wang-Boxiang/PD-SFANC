import numpy as np
import gpuRIR
from scipy.signal import fftconvolve

from .geometry import ambeo_vr_mic_positions
from .signal_utils import add_awgn, normalize_rms


def simulate_moving_source(
    noise,
    segments,
    fs: int = 16000,
    duration_per_segment: float = 0.5,
    mic_center=(6.0, 4.0, 2.2),
    error_mic=(4.6, 4.0, 2.2),
    room_size=(11.0, 9.0, 3.2),
    t60: float = 0.48,
    snr_db: float = 30.0,
    target_rms: float = 0.3,
    rng=None,
):
    rng = np.random.default_rng() if rng is None else rng
    mic_center = np.asarray(mic_center, dtype=float)
    error_mic = np.asarray(error_mic, dtype=float).reshape(1, 3)
    samples_per_segment = int(round(duration_per_segment * fs))
    total_samples = samples_per_segment * len(segments)
    beta = gpuRIR.beta_SabineEstimation(room_size, float(t60))
    t_max = gpuRIR.att2t_SabineEstimator(60.0, float(t60))
    nb_img = gpuRIR.t2n(t_max, room_size)
    reference = None
    disturbance = None
    for index, segment in enumerate(segments):
        start = index * samples_per_segment
        stop = start + samples_per_segment
        current_noise = normalize_rms(noise[start:stop], target_rms)
        azimuth = np.deg2rad(segment["azimuth"])
        elevation = np.deg2rad(segment["elevation"])
        distance = segment["distance"]
        offset = np.array(
            [
                distance * np.cos(elevation) * np.cos(azimuth),
                distance * np.cos(elevation) * np.sin(azimuth),
                distance * np.sin(elevation),
            ]
        )
        source_position = mic_center + offset
        reference_rir = gpuRIR.simulateRIR(
            room_size,
            beta,
            source_position,
            ambeo_vr_mic_positions(mic_center),
            nb_img,
            t_max,
            fs,
            mic_pattern="card",
            orV_rcv=np.asarray([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]),
        )
        error_rir = gpuRIR.simulateRIR(room_size, beta, source_position, error_mic, nb_img, t_max, fs)
        if reference is None:
            reference = np.zeros((total_samples + reference_rir.shape[-1] - 1, 4), dtype=np.float64)
            disturbance = np.zeros((total_samples + error_rir.shape[-1] - 1, 1), dtype=np.float64)
        for channel in range(4):
            y = fftconvolve(current_noise, reference_rir[0, channel, :], mode="full")
            reference[start : start + len(y), channel] += y
        y_error = fftconvolve(current_noise, error_rir[0, 0, :], mode="full")
        disturbance[start : start + len(y_error), 0] += y_error
    reference = add_awgn(reference[:total_samples, :], snr_db, rng=rng)
    disturbance = disturbance[:total_samples, :]
    return reference, disturbance
