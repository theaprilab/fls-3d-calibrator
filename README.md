# FLS-3D Calibrator
This repository contains the code for calibrating the FLS-3D sonar sensor. The calibration process involves determining the intrinsic and extrinsic parameters of the sonar sensor to improve its accuracy and performance in various applications such as underwater mapping, object detection, and navigation.

## Installation
To install the necessary dependencies for the FLS-3D Calibrator, follow these steps:
1. Clone the repository to your local machine using the following command:

   ```bash
   git clone https://github.com/theaprilab/fls-3d-calibrator.git
    ```

2. Install the `uv` environment by running the following command:

   ```bash
   uv sync
    ```

## Running the calibrator

Run the calibrator from the repository root:

```bash
uv run python calibrator.py [flags]
```

The input directory, collection name, snapshot number, and initial calibration
parameters are read from `config.toml`.

### Command-line flags

| Short flag | Long flag | Description |
| --- | --- | --- |
| `-cal` | `--calibration` | Enables calibration mode. Without this flag, the program evaluates the existing calibration from `calibration.toml` against the snapshots in the configured collection. |
| `-cd` | `--calibrate_all` | In calibration mode, calibrates every snapshot directory in the configured collection, consolidates the results, and writes the resulting parameters to `calibration.toml`. |
| `-sc` | `--single_calibration` | In calibration mode, calibrates the single snapshot identified by `input_params.snapshot_number` in `config.toml`. |
| `-dc` | `--display_calibration` | When used with `--single_calibration` or `--single_display`, displays the selected snapshot's calibration frames and saves generated sonar, collapsed-cloud, and mask images under `test_images/`. |
| `-sd` | `--single_display` | Evaluates and optionally displays the single snapshot identified by `input_params.snapshot_number`, using the existing parameters from `calibration.toml` rather than recalibrating it. |

These options are declared as boolean-valued arguments. Pass `True` to enable
an option; omit it to leave the option disabled. (With the current
`argparse` configuration, passing the string `False` is not equivalent to
omitting the option.)

```bash
# Calibrate all snapshots and save the consolidated calibration.
uv run python calibrator.py --calibration True --calibrate_all True

# Calibrate and display one snapshot.
uv run python calibrator.py --calibration True --single_calibration True --display_calibration True

# Display one snapshot using the saved calibration.
uv run python calibrator.py --single_display True --display_calibration True
```

If no flags are provided, the program evaluates the saved calibration across
all snapshots in the configured collection.

## Viewing denoised point clouds

When denoised point-cloud files have been generated for a snapshot, use the
viewer from the repository root:

```bash
uv run python denoised_viewer.py
```

By default, it reads `config.toml`, loads every `.npy` file in the configured
snapshot's `denoised/` directory, and opens the plots in Matplotlib. The
viewer also supports:

| Option | Description |
| --- | --- |
| `--config PATH` | Use a different TOML configuration file. |
| `--snapshot-number NUMBER` | Override `input_params.snapshot_number` from the config. |
| `--file NAME.npy` | Display only one file from the snapshot's `denoised/` directory. |
| `--point-size SIZE` | Set the plotted marker size; the default is `1.0`. |
| `--save-dir PATH` | Save each plot as a PNG in the specified directory. |
| `--no-show` | Do not open plot windows; use with `--save-dir` for headless systems. |

Examples:

```bash
# Display only snapshot 0037.
uv run python denoised_viewer.py --snapshot-number 0037

# Save all plots without opening GUI windows.
uv run python denoised_viewer.py --save-dir viewer_output --no-show

# Save one point cloud with larger markers.
uv run python denoised_viewer.py \
  --file denoised_points_0001.npy \
  --point-size 3 \
  --save-dir viewer_output \
  --no-show
```

The viewer expects files shaped like an `N x 3` NumPy array, where columns
represent `X`, `Y`, and `Z`. The calibration code must save those arrays into
the snapshot's `denoised/` directory before they can be viewed.
