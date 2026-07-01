# Graduation Project - Sports Tracking

This project processes sports videos (basketball, football, tennis) using computer vision to track players, the ball, passes, interceptions, and camera movement. It maps the action to a 2D tactical view (mini-map) and can generate team-specific heatmaps.

## Running Locally

To run the pipeline locally on your machine (without relying on Modal), use the `local_main.py` script:

```bash
python local_main.py --sport basketball --input basketball_test2.mp4
```

### Arguments

- `--sport`: The sport to track. Supported options: `basketball`, `football`, `tennis`. Default is `basketball`.
- `--input`: The input video filename. This file must be placed in the `../input_folder/` directory relative to this script.
- `--no-teams`: Include this flag to disable team assignment (which otherwise loads a SigLIP2 model to cluster teams by apparel).

### Outputs

The script will save the processed video and any generated team heatmaps (for basketball/football) in the `outputs/` directory.

### Models and Data Structure

For `local_main.py` to work correctly, ensure your repository has the following structure:
- `input_folder/`: Place your input `.mp4` videos here.
- `grad_proj/models/`: Contains the required YOLO and SigLIP2 model weights (e.g., `yolo11x_v2_best.pt`, `basketball_court_yolo11lv2.pt`).
- `grad_proj/images/`: Contains the 2D pitch images used for the mini-map projection.

## Running on Modal

The project also supports remote execution via Modal, which can be run using the `modal_main.py` script:

```bash
modal run modal_main.py --sport basketball --input-filename basketball_test2.mp4
```