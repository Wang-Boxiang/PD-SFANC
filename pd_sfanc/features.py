import torch


def compute_stft_for_all_channels(
    frames_per_channel,
    n_fft: int = 1024,
    hop_length: int = 64,
    center: bool = False,
):
    if len(frames_per_channel) != 4:
        raise ValueError("frames_per_channel must contain four channel frame lists")
    first_frame = next((frames[0] for frames in frames_per_channel if frames), None)
    if first_frame is None:
        raise ValueError("each channel must contain at least one frame")
    window = torch.hann_window(n_fft, device=first_frame.device, dtype=first_frame.dtype)
    mag_channels = []
    phase_channels = []
    for channel_frames in frames_per_channel:
        if not channel_frames:
            raise ValueError("each channel must contain at least one frame")
        mags = []
        phases = []
        for frame in channel_frames:
            spectrum = torch.stft(
                frame,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                return_complex=True,
                center=center,
            )
            mags.append(torch.abs(spectrum))
            phases.append(torch.angle(spectrum))
        mag_channels.append(torch.cat(mags, dim=-1).unsqueeze(0))
        phase_channels.append(torch.cat(phases, dim=-1).unsqueeze(0))
    return torch.cat([torch.cat(mag_channels, dim=0), torch.cat(phase_channels, dim=0)], dim=0)
