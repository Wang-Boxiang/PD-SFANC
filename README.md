# PD-SFANC

This repository provides a minimal open-source implementation of Prediction-Driven Spatial-Filtering Active Noise Control (PD-SFANC).

The current release focuses on the inference-only real-world linear trajectory demo used in our paper. It includes the pretrained azimuth classifier, pretrained control filters, the secondary path, and a runnable Jupyter notebook for the final denoising simulation.

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

This release does not include data generation scripts, training scripts, or baseline implementations.

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

## Expected Demo Output

With the bundled assets, the real-world linear demo should report:

```text
Top-1 azimuth accuracy: 1.0000
Angular MAE: 2.00 deg
Mean noise reduction after warmup: about 15 dB
```

Small numerical differences may occur across CUDA, PyTorch, and `gpuRIR` versions.

## Citation

If this code is useful for your research, please cite our associated PD-SFANC paper.
