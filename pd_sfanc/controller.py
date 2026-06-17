import torch


class PredictiveFixedFilterController:
    def __init__(
        self,
        frame_length: int,
        control_filters,
        num_channels: int = 4,
        filter_length: int = 1024,
        warmup_frames: int = 4,
        device=None,
        dtype=torch.float32,
    ):
        self.frame_length = int(frame_length)
        self.num_channels = int(num_channels)
        self.filter_length = int(filter_length)
        self.warmup_frames = int(warmup_frames)
        self.device = device or torch.device("cpu")
        self.control_filters = torch.as_tensor(control_filters, dtype=dtype, device=self.device).contiguous()
        if self.control_filters.ndim != 3:
            raise ValueError("control_filters must have shape [n_frames, num_channels, filter_length]")
        if self.control_filters.shape[1] != self.num_channels:
            raise ValueError("control_filters channel count does not match num_channels")
        if self.control_filters.shape[2] != self.filter_length:
            raise ValueError("control_filters length does not match filter_length")
        self.delay_lines = torch.zeros(self.num_channels, self.filter_length, dtype=dtype, device=self.device)
        self.current_filter = torch.zeros(self.num_channels, self.filter_length, dtype=dtype, device=self.device)

    def reset_state(self):
        self.delay_lines.zero_()
        self.current_filter.zero_()

    @torch.no_grad()
    def noise_cancellation(self, disturbance, filtered_reference):
        disturbance = torch.as_tensor(disturbance, dtype=self.current_filter.dtype, device=self.device).flatten()
        filtered_reference = torch.as_tensor(
            filtered_reference,
            dtype=self.current_filter.dtype,
            device=self.device,
        )
        if filtered_reference.shape != (disturbance.numel(), self.num_channels):
            raise ValueError("filtered_reference must have shape [n_samples, num_channels]")
        if disturbance.numel() % self.frame_length != 0:
            raise ValueError("signal length must be divisible by frame_length")
        output = torch.zeros_like(disturbance)
        for sample_index in range(disturbance.numel()):
            if sample_index % self.frame_length == 0:
                frame_index = sample_index // self.frame_length
                if frame_index < self.warmup_frames:
                    self.current_filter.zero_()
                else:
                    control_index = min(frame_index, self.control_filters.shape[0] - 1)
                    self.current_filter.copy_(self.control_filters[control_index])
            for channel in range(self.num_channels):
                self.delay_lines[channel] = torch.roll(self.delay_lines[channel], 1, 0)
                self.delay_lines[channel, 0] = filtered_reference[sample_index, channel]
            control_signal = torch.sum(self.current_filter * self.delay_lines)
            output[sample_index] = disturbance[sample_index] - control_signal
        return output.cpu().numpy()
