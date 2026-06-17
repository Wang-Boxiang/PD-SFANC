import numpy as np


def ambeo_vr_mic_positions(
    center=(0.0, 0.0, 0.0),
    radius: float = 0.0125,
    azimuths_deg=(0.0, 180.0, 90.0, 270.0),
    elevations_deg=(35.264, 35.264, -35.264, -35.264),
):
    azimuths = np.deg2rad(azimuths_deg)
    elevations = np.deg2rad(elevations_deg)
    points = []
    for azimuth, elevation in zip(azimuths, elevations):
        points.append(
            [
                radius * np.cos(elevation) * np.cos(azimuth),
                radius * np.cos(elevation) * np.sin(azimuth),
                radius * np.sin(elevation),
            ]
        )
    return np.asarray(points) + np.asarray(center)


def make_linear_trajectory(
    n_frames: int,
    start_angle: float,
    angular_step_deg: float,
    distance: float = 0.4,
    elevation: float = 0.0,
):
    segments = []
    for index in range(n_frames):
        segments.append(
            {
                "distance": float(distance),
                "elevation": float(elevation),
                "azimuth": float((start_angle + angular_step_deg * index) % 360.0),
            }
        )
    return segments


def azimuth_centers(num_classes: int = 36):
    return np.linspace(0.0, 360.0, num_classes, endpoint=False)


def azimuth_to_class(azimuth_deg: float, centers=None):
    centers = azimuth_centers() if centers is None else np.asarray(centers)
    azimuth = azimuth_deg % 360.0
    distance = np.abs(azimuth - centers)
    distance = np.minimum(distance, 360.0 - distance)
    return int(np.argmin(distance))


def circular_error_deg(a: float, b: float):
    error = abs(a - b) % 360.0
    return float(min(error, 360.0 - error))
