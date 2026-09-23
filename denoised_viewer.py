"""Display denoised point-cloud files saved by the calibrator."""

import argparse
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_config(config_path: Path) -> dict:
    """Load the TOML configuration used to locate the snapshot."""
    try:
        with config_path.open("rb") as config_file:
            return tomllib.load(config_file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        ) from exc


def snapshot_path_from_config(
    config: dict, snapshot_number: str | None = None
) -> Path:
    """Build the configured snapshot path, optionally overriding its number."""
    try:
        input_params = config["input_params"]
        snapshot_directory = Path(input_params["snapshot_dir"]).expanduser()
        collection_name = input_params["collection_name"]
        configured_number = input_params["snapshot_number"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "config.toml must define input_params.snapshot_dir, "
            "input_params.collection_name, and input_params.snapshot_number"
        ) from exc

    number = configured_number if snapshot_number is None else snapshot_number
    return snapshot_directory / collection_name / f"snapshot_{number}"


def point_cloud_files(denoised_path: Path, filename: str | None) -> list[Path]:
    """Return sorted NumPy point-cloud files, optionally selecting one file."""
    if not denoised_path.is_dir():
        raise FileNotFoundError(
            f"Denoised point-cloud directory not found: {denoised_path}"
        )

    if filename is not None:
        selected = denoised_path / filename
        if selected.suffix != ".npy" or not selected.is_file():
            raise FileNotFoundError(
                f"NumPy point-cloud file not found: {selected}"
            )
        return [selected]

    files = sorted(denoised_path.glob("*.npy"))
    if not files:
        raise FileNotFoundError(
            f"No .npy point-cloud files found in {denoised_path}"
        )
    return files


def display_point_cloud(file_path: Path, point_size: float) -> None:
    """Display one point cloud in a Matplotlib 3D figure."""
    data = np.load(file_path)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            f"{file_path} must contain an array shaped (N, 3+) "
            f"(received {data.shape})"
        )

    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(data[:, 0], data[:, 1], data[:, 2], s=point_size)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_title(f"Denoised Point Cloud: {file_path.name}")
    figure.tight_layout()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display denoised point clouds saved by calibrator.py."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="Path to the TOML configuration file (default: config.toml).",
    )
    parser.add_argument(
        "--snapshot-number",
        help="Snapshot number to display instead of the value in config.toml.",
    )
    parser.add_argument(
        "--file",
        help="Display one .npy file from the snapshot's denoised directory.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="Save rendered plots as PNG files instead of only displaying them.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.0,
        help="Marker size for plotted points (default: 1.0).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open plot windows; useful with --save-dir on headless systems.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.point_size <= 0:
        raise ValueError("--point-size must be greater than zero")
    if args.no_show and args.save_dir is None:
        raise ValueError("--no-show requires --save-dir")

    config = load_config(args.config)
    snapshot_path = snapshot_path_from_config(config, args.snapshot_number)
    denoised_path = snapshot_path / "denoised"
    files = point_cloud_files(denoised_path, args.file)
    print(f"Loading {len(files)} point cloud(s) from {denoised_path}")

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        display_point_cloud(file_path, args.point_size)
        if args.save_dir is not None:
            output_path = args.save_dir / f"{file_path.stem}.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved {output_path}")
        if args.no_show:
            plt.close()

    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
