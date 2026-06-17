# PD-SFANC

This is the code repository for the paper: **"Predictive Directional Selective Fixed-Filter Active Noise Control for Moving Sources via a Convolutional Recurrent Neural Network"**, accepted by *Interspeech 2026*. The paper is available on [https://arxiv.org/abs/2409.05470](https://arxiv.org/abs/2604.23144).


## Highlights

- **Predictive directional ANC for moving sources.** PD-SFANC estimates the incoming noise direction before selecting the control filter, allowing the controller to respond to source motion rather than only to the current error signal.
- **CRNN-based direction prediction.** A convolutional recurrent neural network is used to infer the source azimuth from multichannel microphone observations, providing frame-level directional information for filter selection.
- **Selective fixed-filter control.** Instead of adapting filters online, PD-SFANC selects from a bank of pretrained directional control filters, reducing online computational cost while preserving directional control capability.
- **End-to-end real-world simulation.** The released notebook demonstrates the full inference pipeline, including direction prediction, filter selection, secondary-path simulation, and denoising result visualization.

## Overview

<p align="center">
  <img src="data/main_control.jpg" alt="PD-SFANC control diagram" width="760">
</p>

<p align="center">
  <img src="data/doa_prediction.png" alt="DoA prediction overview" width="760">
</p>

## Release Contents

```text
notebooks/07a_end2end_realworld_linear.ipynb
    Runnable PD-SFANC denoising demo with displayed results.

pd_sfanc/
    Core PD-SFANC inference modules.

scripts/run_realworld_linear.py
    Backend used by the notebook.

models/CRNN_interspeech.pth
    Pretrained CRNN azimuth classifier.

data/
    Demo noise, pretrained control filters, and secondary path.
```


## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

The demo uses `gpuRIR` for room impulse response simulation. Please install a CUDA-enabled `gpuRIR` build before running the notebook.

## Run the Demo

Open and run:

```text
notebooks/07a_end2end_realworld_linear.ipynb
```

The notebook runs the complete inference-only PD-SFANC pipeline and displays the generated result figure. It also writes the following files to `outputs/`:

```text
realworld_linear_summary.json
realworld_linear_per_frame.csv
realworld_linear_pd_sfanc.png
```

For command-line execution, the same demo can be run with:

```bash
python scripts/run_realworld_linear.py
```



## Citation

If this code is useful for your research, please cite the paper:

```text
@article{wang2026predictive,
  title={Predictive Directional Selective Fixed-Filter Active Noise Control for Moving Sources via a Convolutional Recurrent Neural Network},
  author={Wang, Boxiang and Luo, Zhengding and Shi, Dongyuan and Ji, Junwei and Su, Xiruo and Gan, Woon-Seng},
  journal={arXiv preprint arXiv:2604.23144},
  year={2026}
}
```
