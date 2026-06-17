import numpy as np
import soundfile as sf
from scipy.signal import firwin, lfilter, resample


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2) + 1e-12))


def normalize_rms(x, target_rms: float):
    return np.asarray(x) / rms(x) * target_rms


def add_awgn(signal, snr_db: float, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    signal = np.asarray(signal)
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power) * rng.standard_normal(signal.shape)
    return signal + noise


def load_demo_noise(file_path, target_fs: int, target_length: int, cutoff_hz: float, target_rms: float):
    noise, sample_rate = sf.read(file_path)
    if noise.ndim > 1:
        noise = noise[:, 0]
    if sample_rate != target_fs:
        num_samples = int(len(noise) * target_fs / sample_rate)
        noise = resample(noise, num_samples)
    lowpass = firwin(1024, cutoff_hz, fs=target_fs, pass_zero=True)
    noise = lfilter(lowpass, [1.0], noise)
    noise = normalize_rms(noise, target_rms)
    if len(noise) < target_length:
        repeats = int(np.ceil(target_length / len(noise)))
        noise = np.tile(noise, repeats)
    return noise[:target_length]


def segment_noise_reduction_db(reference, error, segment_samples: int):
    reference = np.asarray(reference).reshape(-1)
    error = np.asarray(error).reshape(-1)
    n_segments = int(np.ceil(len(reference) / segment_samples))
    values = []
    for index in range(n_segments):
        start = index * segment_samples
        stop = min((index + 1) * segment_samples, len(reference))
        values.append(20.0 * np.log10(rms(reference[start:stop]) / rms(error[start:stop])))
    return np.asarray(values)
