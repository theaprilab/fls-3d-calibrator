import argparse
import json
import tomllib
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree, cKDTree
from tqdm import tqdm

from sonar_calibrator import SonarCalibrator


def load_snapshot_data(snapshot_path: Path) -> dict:
    """
    Load data from a snapshot directory.

    :param snapshot_path: Path to the snapshot directory.
    :return: A dictionary containing calibration data.
    """
    camera_data = np.load(
        snapshot_path / "camera.npy", mmap_mode="r", allow_pickle=True
    )
    camera_times = np.load(
        snapshot_path / "camera_times.npy", mmap_mode="r", allow_pickle=True
    )
    camera_data = np.load(
        snapshot_path / "camera.npy", mmap_mode="r", allow_pickle=True
    )
    camera_times = np.load(
        snapshot_path / "camera_times.npy", mmap_mode="r", allow_pickle=True
    )
    pcloud_data = np.load(snapshot_path / "points.npy", allow_pickle=True)
    pcloud_times = np.load(
        snapshot_path / "points_times.npy", mmap_mode="r", allow_pickle=True
    )
    sonar_data = np.load(
        snapshot_path / "sonar.npy", mmap_mode="r", allow_pickle=True
    )
    sonar_times = np.load(
        snapshot_path / "sonar_times.npy", mmap_mode="r", allow_pickle=True
    )
    pcloud_times = np.load(
        snapshot_path / "points_times.npy", mmap_mode="r", allow_pickle=True
    )
    sonar_data = np.load(
        snapshot_path / "sonar.npy", mmap_mode="r", allow_pickle=True
    )
    sonar_times = np.load(
        snapshot_path / "sonar_times.npy", mmap_mode="r", allow_pickle=True
    )

    return {
        "camera_data": camera_data,
        "camera_times": camera_times,
        "pcloud_data": pcloud_data,
        "pcloud_times": pcloud_times,
        "sonar_data": sonar_data,
        "sonar_times": sonar_times,
    }


def prepare_frame(calibrator, points_idx, time_offset=0.0):
    """
    Pair the sonar frame to `points_idx`,
    threshold the sonar image to a mask, and pull out the largest target contour.

    This function respects an optional `time_offset` (in seconds) stored in
    `calibrator.calibration_params['time_offset']`. The point-cloud timestamp is
    shifted by that offset before finding the closest sonar frame. Positive
    offsets advance the point-cloud time (i.e., assume pcl lags behind sonar).

    :param calibrator: An instance of SonarCalibrator containing calibration data.
    :param camera_idx: Index of the camera frame to prepare.
    :return: A tuple containing the point cloud, target contour, mask, grayscale
            sonar image, image shape, and target center, or None if preparation fails.
    """
    points_time = calibrator.calibration_data["pcloud_times"][points_idx]
    # Apply optional time offset (seconds -> nanoseconds)
    tau_s = float(time_offset)
    try:
        points_time_adj = int(points_time + tau_s * 1e9)
    except Exception:
        # Fallback: if types don't support direct arithmetic, coerce via numpy
        points_time_adj = int(
            np.asarray(points_time, dtype=np.int64) + int(tau_s * 1e9)
        )

    sonar_idx = calibrator.closest_frame(
        calibrator.calibration_data["sonar_times"], points_time_adj
    )
    cloud_idx = points_idx
    if sonar_idx is None or cloud_idx is None:
        return None

    sonar_image = calibrator.calibration_data["sonar_data"][sonar_idx][:, ::-1]
    pcloud = calibrator.calibration_data["pcloud_data"][cloud_idx]
    mask = calibrator.generate_mask(sonar_image)
    sonar_contours, _ = calibrator.find_countours(mask)
    if len(sonar_contours) == 0:
        return None

    target = max(sonar_contours, key=cv2.contourArea)
    gray = cv2.cvtColor(sonar_image, cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.boundingRect(target)
    center = np.array([x + w / 2, y + h / 2])
    return pcloud, target, mask, gray, sonar_image.shape[:2], center


def consolidate(rows, labels):
    """
    Combine per-frame parameter estimates into one calibration. Reject outliers with a
    MAD (median-absolute-deviation) test, then take the median of the inliers.

    :param rows: A list of parameter estimates from each frame.
    :param labels: A list of parameter names corresponding to the estimates in `rows`.
    :return: A tuple containing the consolidated parameter estimates, their spread, the number of in
    """
    if isinstance(rows[0], dict):
        rows = np.array(
            [[row[label] for label in labels] for row in rows],
            dtype=float,
        )
    else:
        rows = np.asarray(rows, dtype=float)
    med = np.median(rows, axis=0)
    mad = np.median(np.abs(rows - med), axis=0) + 1e-9
    inlier = np.all(np.abs(rows - med) <= 3 * 1.4826 * mad, axis=1)
    final = np.median(rows[inlier], axis=0)
    spread = rows[inlier].std(axis=0)
    return dict(zip(labels, final)), spread, int(inlier.sum()), len(rows)


def consolidate_calibrations(records, labels):
    """
    Reject high-loss fits and return the consensus medoid.

    records:
        [{"params": {...}, "loss": float}, ...]
    """
    valid = [
        r
        for r in records
        if np.isfinite(r["loss"])
        and all(np.isfinite(r["params"][key]) for key in labels)
    ]

    if len(valid) < 2:
        return None, None, 0, len(records)

    losses = np.asarray([r["loss"] for r in valid], dtype=float)

    # Reject unusually poor objective values.
    loss_median = np.median(losses)
    loss_mad = 1.4826 * np.median(np.abs(losses - loss_median))
    loss_limit = loss_median + 2.5 * max(loss_mad, 1e-9)

    inliers = [r for r in valid if r["loss"] <= loss_limit]

    if len(inliers) < 2:
        return None, None, len(inliers), len(records)

    parameters = np.asarray(
        [[r["params"][key] for key in labels] for r in inliers]
    )

    # Characteristic parameter scales used only when comparing solutions.
    parameter_scales = np.asarray(
        [
            0.10,  # dy: 10 cm
            np.deg2rad(5.0),  # psi: 5 degrees
            0.10,  # radius_scale
            0.10,  # az_scale
        ]
    )

    normalized = parameters / parameter_scales

    pairwise_distances = np.linalg.norm(
        normalized[:, None, :] - normalized[None, :, :],
        axis=2,
    )

    # Choose an actual solution nearest to the consensus.
    medoid_index = np.argmin(np.median(pairwise_distances, axis=1))
    final_array = parameters[medoid_index]

    center = np.median(parameters, axis=0)
    spread = 1.4826 * np.median(
        np.abs(parameters - center),
        axis=0,
    )

    final = dict(zip(labels, final_array))

    return final, spread, len(inliers), len(records)


# =============================================================================
#  MANUAL VALUE
# =============================================================================
MANUAL_MEASURED = {
    "dy": -0.2667,
    "psi": 0.0,
    "radius_scale": 1.0,
    "az_scale": 1.0,
}


def _px_to_m(image_shape):

    return 6.0 / image_shape[0]


def compare_against_manual(calibrator, calibrated, start_params, step=1):
    """
    Compare `calibrated` params against MANUAL_MEASURED across all frames. Output:
      1) parameter deltas (calibrated - manual),
      2) mean alignment quality (IoU / coverage / MI / chamfer) at each calibration,
      3) alignment distances: centroid distance and contour distance
    :param calibrator: An instance of SonarCalibrator containing calibration data.
    :param calibrated: A dictionary of calibrated parameters to compare against MANUAL_MEASURED.
    :param start_params: A dictionary of starting parameters for calibration.
    :param step: Step size for iterating through frames (default is 1).
    :return: None
    """
    n_cam = len(calibrator.calibration_data["pcloud_data"])
    px_to_m = _px_to_m(calibrator.calibration_data["sonar_data"][0].shape[:2])

    print("\nparameter comparison:")
    print(
        f"  {'param':13s} {'manual':>9s} {'calibrated':>11s} {
            'cal-manual':>11s}"
    )
    for k in ["dy", "psi", "radius_scale", "az_scale"]:
        m, c = MANUAL_MEASURED[k], calibrated[k]
        print(f"  {k:13s} {m:>9.4f} {c:>11.4f} {c - m:>+11.4f}")

    def mean_quality(params):
        """
        Compute the mean alignment quality metrics (IoU, coverage, MI, chamfer) across all frames for the given parameters.
        :param params: A dictionary of parameters to evaluate.
        :return: A dictionary containing the mean values of the alignment quality metrics.
        """
        acc = {"iou": [], "coverage": [], "mi": [], "chamfer": []}
        for i in range(0, n_cam, step):
            prep = prepare_frame(calibrator, i)
            if prep is None:
                continue
            pcl, target, mask, gray, shape, center = prep
            cpts = target.squeeze(axis=1).astype(np.float32)
            L = calibrator.all_losses(
                [
                    params["dy"],
                    params["psi"],
                    params["radius_scale"],
                    params["az_scale"],
                ],
                pcl,
                shape,
                cpts,
                center,
                mask,
                gray,
            )
            for key in acc:
                if np.isfinite(L[key]):
                    acc[key].append(L[key])
        return {k: (np.mean(v) if v else float("nan")) for k, v in acc.items()}

    qm, qc = mean_quality(MANUAL_MEASURED), mean_quality(calibrated)
    print("\nmean alignment quality:")
    print(f"  {'':11s} {'IoU':>7s} {'cov':>7s} {'MI':>7s} {'chamfer':>9s}")
    print(
        f"  {'manual':11s} {1 - qm['iou']:>7.3f} {1 - qm['coverage']:>7.3f} {-qm['mi']:>7.3f} {qm['chamfer']:>9.1f}"
    )
    print(
        f"  {'calibrated':11s} {1 - qc['iou']:>7.3f} {
            1 - qc['coverage']:>7.3f} {-qc['mi']:>7.3f} {qc['chamfer']:>9.1f}"
    )

    def distances(params):
        pd = [
            params["dy"],
            params["psi"],
            params["radius_scale"],
            params["az_scale"],
        ]
        cen, cham, haus = [], [], []
        for i in range(0, n_cam, step):
            prep = prepare_frame(calibrator, i)
            if prep is None:
                continue
            pcl, target, mask, gray, shape, center = prep
            Ms = cv2.moments(target)
            if Ms["m00"] == 0:
                continue
            cs = np.array(
                [Ms["m10"] / Ms["m00"], Ms["m01"] / Ms["m00"]]
            )  # sonar target centroid
            best = calibrator._nearest_pcl_contour(pd, pcl, shape, center)
            if best is None:
                continue
            Mc = cv2.moments(best)
            if Mc["m00"] != 0:
                cbl = np.array([Mc["m10"] / Mc["m00"], Mc["m01"] / Mc["m00"]])
                cen.append(np.linalg.norm(cbl - cs))  # centroid distance
            A = target.reshape(-1, 2).astype(np.float32)
            B = best.reshape(-1, 2).astype(np.float32)
            if len(A) and len(B):
                dAB, _ = cKDTree(B).query(A, k=1)
                dBA, _ = cKDTree(A).query(B, k=1)
                cham.append(
                    0.5 * (dAB.mean() + dBA.mean())
                )  # symmetric mean NN (px)
                haus.append(
                    max(dAB.max(), dBA.max())
                )  # Hausdorff / worst case (px)
        return np.array(cen), np.array(cham), np.array(haus)

    print("\nalignment distances:")
    print(
        f"  {'params':11s} {'centroid_px':>12s} {'~cm':>6s} {
            'contour_px':>11s} {'Hausd_px':>9s} {'frames':>7s}"
    )
    for name, P in [("manual", MANUAL_MEASURED), ("calibrated", calibrated)]:
        cen, cham, haus = distances(P)
        if len(cen) and len(cham):
            print(
                f"  {name:11s} {cen.mean():>12.1f} {
                    cen.mean() * px_to_m * 100:>6.1f} "
                f"{cham.mean():>11.2f} {haus.mean():>9.2f} {len(cham):>7d}"
            )
        else:
            print(f"  {name:11s} {'no valid blob pairs':>36s}")

    m_distances = distances(MANUAL_MEASURED)
    auto_distances = distances(calibrated)
    manual_metrics = {
        "qm": qm,
        "qc": qc,
        "manual_distances": m_distances,
        "auto_distances": auto_distances,
    }
    return manual_metrics


def load_labels(label_path):
    """
    Load labelled data from a JSON file.

    :param label_path: Path to the JSON file containing labelled data.
    :return: A dictionary containing the labelled data.
    """
    with open(label_path, "r") as f:
        labels = json.load(f)
    return labels


def compare_against_labels(
    calibrator, calibrated, start_params, labels, step=1
):
    """
    Compare the calibrated parameters using the labelled data. This function
    evaluate the alignment quality metrics (IoU, coverage, MI, chamfer) across
    all frames for the given parameters.
    :param calibrator: An instance of SonarCalibrator containing calibration data.
    :param calibrated: A dictionary of calibrated parameters to compare against labelled data.
    :param start_params: A dictionary of starting parameters for calibration.
    :param step: Step size for iterating through frames (default is 1).

    :return: None
    """
    reproj_error = []
    iou_score = []
    f1_score = []
    chamfer_error = []
    for label in labels:
        point_frame_idx = label["points_frame"]
        sonar_frame_idx = label["sonar_frame"]
        fls_points = np.array(label["fls_uv"])
        fls_points[[-2, -1]] = fls_points[[-1, -2]]
        pcl_points = np.array(label["pcl_points"])
        pcl_points[[-2, -1]] = pcl_points[[-1, -2]]
        # transform the pcl_labels to sonar image coordinates using the calibrated parameters
        image_shape = calibrator.calibration_data["sonar_data"][
            sonar_frame_idx
        ].shape[:2]
        pcl_transformed, _ = calibrator._project_pcl_to_image(
            pcl_points,
            image_shape,
            calibrated,
        )

        # compute the alignment quality metrics (IoU, coverage, MI, chamfer) for the transformed pcl points and the labelled sonar points
        cpts = np.array(fls_points).reshape(-1, 2).astype(np.float32)
        pcl_cpts = np.array(pcl_transformed).reshape(-1, 2).astype(np.float32)
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [cpts.astype(np.int32)], color=255)
        pcl_mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillPoly(pcl_mask, [pcl_cpts.astype(np.int32)], 255)

        intersection = cv2.bitwise_and(mask, pcl_mask)
        union = cv2.bitwise_or(mask, pcl_mask)
        area_intersection = np.sum(intersection == 255)
        area_union = np.sum(union == 255)

        area1 = np.sum(mask == 255)
        area2 = np.sum(pcl_mask == 255)

        iou = area_intersection / area_union if area_union > 0.0 else 0.0
        dice = (
            (2 * area_intersection) / (area1 + area2)
            if (area1 + area2) > 0
            else 0.0
        )
        # Calculating reprojection Error
        errors = [
            cv2.norm(cpts[i], pcl_cpts[i], cv2.NORM_L2)
            for i in range(len(cpts))
        ]
        mean_error = np.mean(errors)

        def boundary_chamfer(mask_a, mask_b):
            contour_a, _ = cv2.findContours(
                mask_a,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            contour_b, _ = cv2.findContours(
                mask_b,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )

            if not contour_a or not contour_b:
                return np.inf

            points_a = max(contour_a, key=cv2.contourArea).reshape(-1, 2)
            points_b = max(contour_b, key=cv2.contourArea).reshape(-1, 2)

            tree_a = KDTree(points_a)
            tree_b = KDTree(points_b)

            distance_a_to_b, _ = tree_b.query(points_a, k=1)
            distance_b_to_a, _ = tree_a.query(points_b, k=1)

            # Symmetric mean boundary distance in pixels.
            return 0.5 * (np.mean(distance_a_to_b) + np.mean(distance_b_to_a))

        chamfer = boundary_chamfer(mask, pcl_mask)
        f1_score.append(dice)
        reproj_error.append(mean_error)
        iou_score.append(iou)
        chamfer_error.append(chamfer)

    mean_f1_score = np.mean(np.array(f1_score))
    mean_reproj = np.mean(np.array(reproj_error))
    mean_iou = np.mean(np.array(iou_score))
    mean_chamfer = np.mean(np.array(chamfer_error))

    return {
        "f1_score": mean_f1_score,
        "reproj_error": mean_reproj,
        "iou": mean_iou,
        "chamfer": mean_chamfer,
    }


def estimate_time_offset_gridsearch(
    calibrator,
    start_params,
    min_s: float = -0.5,
    max_s: float = 0.5,
    n: int = 41,
    sample_step: int = 5,
):
    """
    Simple grid-search over a time offset (seconds) that shifts point-cloud
    timestamps before finding the closest sonar frame. The search evaluates
    the mean chamfer loss across a subset of frames using `start_params` and
    returns the tau (seconds) with minimal mean chamfer.

    :param calibrator: SonarCalibrator instance.
    :param start_params: Dictionary with starting calibration parameters.
    :param min_s: Minimum offset in seconds.
    :param max_s: Maximum offset in seconds.
    :param n: Number of grid samples.
    :param sample_step: Frame step for evaluation (keeps the search fast).
    :return: (best_tau, best_score) or (None, None) if no valid scores found.
    """
    taus = np.linspace(min_s, max_s, n)
    best_tau = None
    best_score = np.inf
    n_cam = len(calibrator.calibration_data.get("pcloud_data", []))
    if n_cam == 0:
        return None, None
    idxs = list(range(0, n_cam, max(1, sample_step)))
    # Ensure we always evaluate at least one frame
    if len(idxs) == 0:
        idxs = [0]

    for tau in taus:
        calibrator.calibration_params = dict(start_params)
        scores = []
        for i in idxs:
            prep = prepare_frame(calibrator, i, tau)
            if prep is None:
                continue
            pcl, target, mask, gray, shape, center = prep
            cpts = target.squeeze(axis=1).astype(np.float32)
            L = calibrator.all_losses(
                [
                    start_params.get("dy", 0.0),
                    start_params.get("psi", 0.0),
                    start_params.get("radius_scale", 1.0),
                    start_params.get("az_scale", 1.0),
                ],
                pcl,
                shape,
                cpts,
                center,
                mask,
                gray,
            )
            cham = L.get("chamfer", np.inf)
            if np.isfinite(cham):
                scores.append(cham)
        if len(scores) == 0:
            continue
        mean_score = float(np.mean(scores))
        # lower chamfer == better
        if mean_score < best_score:
            best_score = mean_score
            best_tau = float(tau)
    if best_tau is None:
        return None, None
    return best_tau, best_score


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate sonar to camera using labelled data."
    )
    parser.add_argument(
        "-dc",
        "--display_calibration",
        type=bool,
        default=False,
        help="Whether to display calibration results.",
    )
    parser.add_argument(
        "-cal",
        "--calibration",
        type=bool,
        default=False,
        help=" Whether to perform calibration.",
    )
    parser.add_argument(
        "-cd",
        "--calibrate_all",
        type=bool,
        default=False,
        help="Whether to calibrate over all snapshots and save calibration parameters..",
    )
    parser.add_argument(
        "-sc",
        "--single_calibration",
        type=bool,
        default=False,
        help="Whether to calibrate over all snapshots and save calibration parameters..",
    )
    parser.add_argument(
        "-sd",
        "--single_display",
        type=bool,
        default=False,
        help="Whether to display calibration results for a single snapshot.",
    )
    args = parser.parse_args()
    print(f"Arguments: {args}")
    # Load the configuration from the TOML file
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    snapshot_directory = config["input_params"]["snapshot_dir"]
    collection_name = config["input_params"]["collection_name"]
    collection_dir = Path(snapshot_directory).expanduser() / collection_name
    cal_params = []
    if args.calibration:
        print("Starting calibration process...")
        if args.calibrate_all:
            print("Calibrating over all snapshots in the directory...")
            snapshot_folders_list = [
                f for f in collection_dir.iterdir() if f.is_dir()
            ]
            print(f"Found {len(snapshot_folders_list)} snapshot folders.")
            for snapshot_path in tqdm(snapshot_folders_list):
                snapshot_number = snapshot_path.name.split("_")[-1]
                print(
                    f"Loading snapshot from: {snapshot_directory}/{collection_name}/snapshot_{snapshot_number}"
                )
                gt_labels_path = snapshot_path / "correspondences.json"
                # Check if the ground truth labels file exists
                if not gt_labels_path.exists():
                    print(
                        f"Error: Ground truth labels file not found at {gt_labels_path}"
                    )
                    gt_labels = None
                else:
                    gt_labels = load_labels(gt_labels_path)
                calibration_data = load_snapshot_data(snapshot_path)
                start_params = dict(
                    config["calibration_params"]
                )  # e.g. identity: dy=0, psi=0, rs=1, az=1
                # METHOD = "chamfer"
                METHOD = "combined"
                labels = ["dy", "psi", "radius_scale", "az_scale"]
                if METHOD == "chamfer":
                    # Original behaviour: keep the built-in Chamfer regularization penalty.
                    calibrator = SonarCalibrator(
                        calibration_data, start_params
                    )
                else:
                    # Combined behaviour: no start-anchored penalty (it is mis-scaled for the
                    # normalized combined loss); a light pull of radius_scale/az_scale toward 1.0
                    # is applied inside combined_loss via scale_reg instead.
                    calibrator = SonarCalibrator(
                        calibration_data, start_params, penalty_weight=0.0
                    )
                    calibrator.scale_reg = 0.3
                print(
                    f"Sonar Data Shape: {calibrator.calibration_data['sonar_data'][0].shape[:2]}"
                )
                # Estimate time offset (seconds) by simple grid-search before per-frame calibration.
                print("Estimating time offset (grid search)...")
                # sample_step controls how many frames are evaluated during the grid search;
                # use up to ~50 frames evenly spaced to keep runtime reasonable.
                sample_step = max(
                    1,
                    int(
                        len(calibrator.calibration_data.get("pcloud_data", []))
                        / 50
                    ),
                )
                best_tau, best_score = estimate_time_offset_gridsearch(
                    calibrator,
                    start_params,
                    min_s=-0.5,
                    max_s=0.5,
                    n=41,
                    sample_step=sample_step,
                )
                if best_tau is not None:
                    print(
                        f"Estimated time_offset = {best_tau:.3f}s (score {best_score:.3f}), applying to calibrator and start_params"
                    )
                else:
                    print(
                        "Time offset grid-search found no valid alignment; leaving time_offset=0.0"
                    )
                    best_tau = 0.0

                n_cam = len(calibrator.calibration_data["pcloud_data"])

                weights = None
                if METHOD == "combined":
                    for i in range(n_cam):
                        prep = prepare_frame(calibrator, i, best_tau)
                        if prep is None:
                            continue
                        pcl, target, mask, gray, shape, center = prep
                        calibrator.calibration_params = dict(start_params)
                        calibrator.multi_variable_optimization(
                            pcl, target, shape
                        )  # reference alignment
                        base = dict(calibrator.calibration_params)
                        weights = calibrator.weight_by_prediction(
                            pcl,
                            shape,
                            target,
                            center,
                            mask,
                            gray,
                            base_params=base,
                            n_samples=200,
                        )["weights"]
                        break
                    if weights is None:
                        print(
                            "Error: no frame with a valid target contour; cannot weight metrics."
                        )
                        return

                # -------------------------------------------------------------------------
                # Per-frame calibration. Each frame is fit from the same start then all estimates are consolidated by an outlier-rejected median.
                # -------------------------------------------------------------------------
                rows = []
                for i in range(n_cam):
                    prep = prepare_frame(calibrator, i, best_tau)
                    if prep is None:
                        continue
                    pcl, target, mask, gray, shape, center = prep
                    calibrator.calibration_params = dict(
                        start_params
                    )  # reset each frame

                    if METHOD == "chamfer":
                        calibrator.multi_variable_optimization(
                            pcl, target, shape
                        )
                        p = calibrator.calibration_params

                        rows.append(
                            [
                                p["dy"],
                                p["psi"],
                                p["radius_scale"],
                                p["az_scale"],
                            ]
                        )
                    else:
                        res = calibrator.optimize_weighted(
                            pcl,
                            shape,
                            target,
                            center,
                            mask,
                            gray,
                            weights,
                            maxiter=500,
                        )
                        if res is None or not getattr(res, "success", False):
                            continue
                        p = calibrator.calibration_params
                        rows.append(
                            {
                                "params": {
                                    "dy": float(p["dy"]),
                                    "psi": float(p["psi"]),
                                    "radius_scale": float(p["radius_scale"]),
                                    "az_scale": float(p["az_scale"]),
                                },
                                "loss": float(res.fun),
                            }
                        )

                if len(rows) < 2:
                    print("Error: too few frames produced a calibration.")
                    return

                final, spread, ninl, ntot = consolidate_calibrations(
                    rows, labels
                )
                print(f"\nMethod: {METHOD}")
                print(
                    f"Consolidated calibration (median of {ninl}/{ntot} inlier frames):"
                )
                for i, l in enumerate(labels):
                    print(
                        f"  {l:13s} = {
                            final[i]
                            if isinstance(final, np.ndarray)
                            else final[l]:+.4f}  +/- {spread[i]:.4f}"
                    )
                calibrator.calibration_params = dict(final)
                snapshot_loss = np.median(
                    [
                        row["loss"]
                        for row in rows
                        if row["loss"]
                        <= np.median([r["loss"] for r in rows])
                        + 2.5
                        * max(
                            1.4826
                            * np.median(
                                np.abs(
                                    np.asarray([r["loss"] for r in rows])
                                    - np.median([r["loss"] for r in rows])
                                )
                            ),
                            1e-9,
                        )
                    ]
                )
                cal_params.append(
                    {"params": final.copy(), "loss": snapshot_loss}
                )
                compare_against_manual(calibrator, dict(final), start_params)
                if gt_labels_path.exists():
                    metrics = compare_against_labels(
                        calibrator,
                        dict(final),
                        start_params,
                        gt_labels,
                        step=1,
                    )
                    print(f"Metrics from labeled data: {metrics}")
                    calibrator.calibration_params = MANUAL_MEASURED
                    metrics_manual = compare_against_labels(
                        calibrator,
                        MANUAL_MEASURED,
                        start_params,
                        gt_labels,
                        step=1,
                    )
                    print(
                        f"Manual Metrics from labeled data: {metrics_manual}"
                    )

            # Find the median calibration parameters across all snapshots
            if cal_params:
                labels = ["dy", "psi", "radius_scale", "az_scale"]
                final, spread, ninl, ntot = consolidate_calibrations(
                    cal_params, labels
                )
                for key, value in dict(final).items():
                    print(f"  {key}: {value:.4f}")

                # Output to calibration.toml
                with open("calibration.toml", "w") as f:
                    f.write("[calibration_params]\n")
                    for key, value in dict(final).items():
                        f.write(f"{key} = {value:.6f}\n")

        elif args.single_calibration:
            snapshot_number = config["input_params"]["snapshot_number"]
            collection_name = config["input_params"]["collection_name"]
            print(
                f"Loading snapshot from: {snapshot_directory}/{
                    collection_name
                }/snapshot_{snapshot_number}"
            )
            snapshot_path = (
                Path(snapshot_directory).expanduser()
                / collection_name
                / f"snapshot_{snapshot_number}"
            )

            gt_labels_path = snapshot_path / "correspondences.json"
            # Check if the ground truth labels file exists
            if not gt_labels_path.exists():
                print(
                    f"Error: Ground truth labels file not found at {gt_labels_path}"
                )
                pass

            else:
                gt_labels = load_labels(gt_labels_path)

                start_params = dict(
                    config["calibration_params"]
                )  # e.g. identity: dy=0, psi=0, rs=1, az=1
            calibration_data = load_snapshot_data(snapshot_path)
            start_params = dict(
                config["calibration_params"]
            )  # e.g. identity: dy=0, psi=0, rs=1, az=1

            # =========================================================================
            #  SELECT CALIBRATION METHOD
            # -------------------------------------------------------------------------
            METHOD = "chamfer"
            # METHOD = "combined"
            # =========================================================================

            labels = ["dy", "psi", "radius_scale", "az_scale"]

            if METHOD == "chamfer":
                # Original behaviour: keep the built-in Chamfer regularization penalty.
                calibrator = SonarCalibrator(calibration_data, start_params)
            else:
                # Combined behaviour: no start-anchored penalty (it is mis-scaled for the
                # normalized combined loss); a light pull of radius_scale/az_scale toward 1.0
                # is applied inside combined_loss via scale_reg instead.
                calibrator = SonarCalibrator(
                    calibration_data, start_params, penalty_weight=0.0
                )
                calibrator.scale_reg = 0.3

            print(
                f"Sonar Data Shape: {calibrator.calibration_data['sonar_data'][0].shape[:2]}"
            )
            # Estimate time offset (seconds) by simple grid-search before per-frame calibration.
            print("Estimating time offset (grid search)...")
            # sample_step controls how many frames are evaluated during the grid search;
            # use up to ~50 frames evenly spaced to keep runtime reasonable.
            sample_step = max(
                1,
                int(
                    len(calibrator.calibration_data.get("pcloud_data", []))
                    / 50
                ),
            )
            best_tau, best_score = estimate_time_offset_gridsearch(
                calibrator,
                start_params,
                min_s=-0.5,
                max_s=0.5,
                n=41,
                sample_step=sample_step,
            )
            if best_tau is not None:
                print(
                    f"Estimated time_offset = {best_tau:.3f}s (score {best_score:.3f}), applying to calibrator and start_params"
                )
            else:
                print(
                    "Time offset grid-search found no valid alignment; leaving time_offset=0.0"
                )
                best_tau = 0.0

            n_cam = len(calibrator.calibration_data["pcloud_data"])

            # -------------------------------------------------------------------------
            # For combined method, compute the metric weights once on the
            # first frame that has a valid target, using a Chamfer fit as the reference.
            # -------------------------------------------------------------------------
            weights = None
            if METHOD == "combined":
                for i in range(n_cam):
                    prep = prepare_frame(calibrator, i, best_tau)
                    if prep is None:
                        continue
                    pcl, target, mask, gray, shape, center = prep
                    calibrator.calibration_params = dict(start_params)
                    calibrator.multi_variable_optimization(
                        pcl, target, shape
                    )  # reference alignment
                    base = dict(calibrator.calibration_params)
                    weights = calibrator.weight_by_prediction(
                        pcl,
                        shape,
                        target,
                        center,
                        mask,
                        gray,
                        base_params=base,
                        n_samples=200,
                    )["weights"]
                    break
                if weights is None:
                    print(
                        "Error: no frame with a valid target contour; cannot weight metrics."
                    )
                    return

            # -------------------------------------------------------------------------
            # Per-frame calibration. Each frame is fit from the same start then all estimates are consolidated by an outlier-rejected median.
            # -------------------------------------------------------------------------
            rows = []
            for i in range(n_cam):
                prep = prepare_frame(calibrator, i, best_tau)
                if prep is None:
                    continue
                pcl, target, mask, gray, shape, center = prep
                calibrator.calibration_params = dict(
                    start_params
                )  # reset each frame

                if METHOD == "chamfer":
                    calibrator.multi_variable_optimization(pcl, target, shape)
                    p = calibrator.calibration_params

                    rows.append(
                        [p["dy"], p["psi"], p["radius_scale"], p["az_scale"]]
                    )
                else:
                    res = calibrator.optimize_weighted(
                        pcl,
                        shape,
                        target,
                        center,
                        mask,
                        gray,
                        weights,
                        maxiter=500,
                    )
                    if res is None or not getattr(res, "success", False):
                        continue
                    p = calibrator.calibration_params
                    rows.append(
                        [p["dy"], p["psi"], p["radius_scale"], p["az_scale"]]
                    )

            if len(rows) < 2:
                print("Error: too few frames produced a calibration.")
                return

            final, spread, ninl, ntot = consolidate(rows, labels)
            print(f"\nMethod: {METHOD}")
            print(
                f"Consolidated calibration (median of {ninl}/{ntot} inlier frames):"
            )
            for i, l in enumerate(labels):
                print(
                    f"  {l:13s} = {
                        final[i]
                        if isinstance(final, np.ndarray)
                        else final[l]:+.4f}  +/- {spread[i]:.4f}"
                )
            calibrator.calibration_params = dict(final)

            compare_against_manual(calibrator, dict(final), start_params)
            if gt_labels_path.exists():
                metrics = compare_against_labels(
                    calibrator, dict(final), start_params, gt_labels, step=1
                )
                print(f"Metrics from labeled data: {metrics}")
                calibrator.calibration_params = MANUAL_MEASURED
                metrics_manual = compare_against_labels(
                    calibrator,
                    MANUAL_MEASURED,
                    start_params,
                    gt_labels,
                    step=1,
                )
                print(f"Manual Metrics from labeled data: {metrics_manual}")

            rows = np.array(rows)
            plt.figure(figsize=(12, 8))
            for j, l in enumerate(labels):
                plt.subplot(2, 2, j + 1)
                plt.plot(rows[:, j], marker="o")
                plt.axhline(final[l], color="g", linestyle="--")
                plt.title(f"{l} over frames")
                plt.xlabel("Frame")
                plt.ylabel(l)
            plt.tight_layout()
            plt.show()
            if args.display_calibration:
                sonar_images = []
                pc_masks = []
                collapsed_pcls = []
                for i in range(
                    len(calibrator.calibration_data["pcloud_data"])
                ):
                    pcloud = calibrator.calibration_data["pcloud_data"][i]
                    sonar_idx = calibrator.closest_frame(
                        calibrator.calibration_data["sonar_times"],
                        calibrator.calibration_data["pcloud_times"][i]
                        + best_tau,
                    )
                    sonar_image = calibrator.calibration_data["sonar_data"][
                        sonar_idx
                    ][:, ::-1]
                    denoised_pcloud = calibrator.denoise_pointcloud(
                        pcloud, sonar_image=sonar_image
                    )
                    sonar_mask, pc_mask, collapsed_pcl = (
                        calibrator.display_collapsed_cloud(
                            pcloud,
                            sonar_image,
                            sonar_image.shape[:2],
                            save_frames=True,
                        )
                    )
                    # Dump Denoised pcloud to npy files in denoised subfolder in Path
                    denoised_path = snapshot_path / "denoised"
                    denoised_path.mkdir(parents=True, exist_ok=True)
                    save = True
                    if save:
                        # np.save(
                        #     denoised_path / f"denoised_points_{i:04d}.npy",
                        #     denoised_pcloud,
                        # )
                        sonar_image = cv2.cvtColor(
                            sonar_image, cv2.COLOR_RGB2BGR
                        )
                        cv2.imwrite(
                            f"test_images/sonar_{i:04d}.png", sonar_image
                        )
                        cv2.imwrite(
                            f"test_images/collapsed_{i:04d}.png", collapsed_pcl
                        )
                        cv2.imwrite(
                            f"test_images/sonar_mask_{i:04d}.png", sonar_mask
                        )
                        cv2.imwrite(
                            f"test_images/pc_mask_{i:04d}.png", pc_mask
                        )

    elif args.single_display:
        snapshot_number = config["input_params"]["snapshot_number"]
        collection_name = config["input_params"]["collection_name"]
        print(
            f"Loading snapshot from: {snapshot_directory}/{
                collection_name
            }/snapshot_{snapshot_number}"
        )
        snapshot_path = (
            Path(snapshot_directory).expanduser()
            / collection_name
            / f"snapshot_{snapshot_number}"
        )
        snapshot_number = snapshot_path.name.split("_")[-1]
        print(
            f"Loading snapshot from: {snapshot_directory}/{collection_name}/snapshot_{snapshot_number}"
        )
        gt_labels_path = snapshot_path / "correspondences.json"
        start_params = dict(
            config["calibration_params"]
        )  # e.g. identity: dy=0, psi=0, rs=1, az=1
        gt_labels_path = snapshot_path / "correspondences.json"

        if not gt_labels_path.exists():
            print(
                f"Error: Ground truth labels file not found at {gt_labels_path}"
            )
            pass

        else:
            gt_labels = load_labels(gt_labels_path)

            start_params = dict(
                config["calibration_params"]
            )  # e.g. identity: dy=0, psi=0, rs=1, az=1
        # Get calibration from calibration.toml
        with open("calibration.toml", "rb") as f:
            calibration_values = tomllib.load(f)["calibration_params"]
        calibration_data = load_snapshot_data(snapshot_path)
        calibrator = SonarCalibrator(
            calibration_data, calibration_params=calibration_values
        )

        compare_against_manual(
            calibrator, dict(calibration_values), start_params
        )
        if gt_labels_path.exists():
            metrics = compare_against_labels(
                calibrator,
                dict(calibration_values),
                start_params,
                gt_labels,
                step=1,
            )
            print(f"Metrics from labeled data: {metrics}")
            calibrator.calibration_params = MANUAL_MEASURED
            metrics_manual = compare_against_labels(
                calibrator,
                MANUAL_MEASURED,
                start_params,
                gt_labels,
                step=1,
            )
            print(f"Manual Metrics from labeled data: {metrics_manual}")

        if args.display_calibration:
            sonar_images = []
            pc_masks = []
            collapsed_pcls = []
            for i in range(len(calibrator.calibration_data["pcloud_data"])):
                pcloud = calibrator.calibration_data["pcloud_data"][i]
                sonar_idx = calibrator.closest_frame(
                    calibrator.calibration_data["sonar_times"],
                    calibrator.calibration_data["pcloud_times"][i],
                )
                sonar_image = calibrator.calibration_data["sonar_data"][
                    sonar_idx
                ][:, ::-1]
                denoised_pcloud = calibrator.denoise_pointcloud(
                    pcloud, sonar_image=sonar_image
                )
                sonar_mask, pc_mask, collapsed_pcl = (
                    calibrator.display_collapsed_cloud(
                        pcloud,
                        sonar_image,
                        sonar_image.shape[:2],
                        save_frames=True,
                    )
                )
                # Dump Denoised pcloud to npy files in denoised subfolder in Path
                denoised_path = snapshot_path / "denoised"
                denoised_path.mkdir(parents=True, exist_ok=True)
                save = True
                if save:
                    # np.save(
                    #     denoised_path / f"denoised_points_{i:04d}.npy",
                    #     denoised_pcloud,
                    # )
                    sonar_image = cv2.cvtColor(sonar_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"test_images/sonar_{i:04d}.png", sonar_image)
                    cv2.imwrite(
                        f"test_images/collapsed_{i:04d}.png", collapsed_pcl
                    )
                    cv2.imwrite(
                        f"test_images/sonar_mask_{i:04d}.png", sonar_mask
                    )
                    cv2.imwrite(f"test_images/pc_mask_{i:04d}.png", pc_mask)
    else:
        snapshot_folders_list = [
            f for f in collection_dir.iterdir() if f.is_dir()
        ]
        collection_name = config["input_params"]["collection_name"]
        print(f"Found {len(snapshot_folders_list)} snapshot folders.")
        # Storing compare against manual numbers for statistical validation
        manual_metrics = []
        for snapshot_path in tqdm(snapshot_folders_list):
            snapshot_number = snapshot_path.name.split("_")[-1]
            print(
                f"Loading snapshot from: {snapshot_directory}/{collection_name}/snapshot_{snapshot_number}"
            )
            gt_labels_path = snapshot_path / "correspondences.json"
            start_params = dict(
                config["calibration_params"]
            )  # e.g. identity: dy=0, psi=0, rs=1, az=1
            gt_labels_path = snapshot_path / "correspondences.json"

            if not gt_labels_path.exists():
                print(
                    f"Error: Ground truth labels file not found at {gt_labels_path}"
                )
                pass

            else:
                gt_labels = load_labels(gt_labels_path)

                start_params = dict(
                    config["calibration_params"]
                )  # e.g. identity: dy=0, psi=0, rs=1, az=1
            # Get calibration from calibration.toml
            with open("calibration.toml", "rb") as f:
                calibration_values = tomllib.load(f)["calibration_params"]
            calibration_data = load_snapshot_data(snapshot_path)
            calibrator = SonarCalibrator(
                calibration_data, calibration_params=calibration_values
            )
            calibration_params = config["calibration_params"]
            print("\nUsing calibration parameters from calibration.toml:")
            for k, v in calibration_values.items():
                print(f"  {k}: {v:.6f}")

            sample_step = max(
                1,
                int(
                    len(calibrator.calibration_data.get("pcloud_data", []))
                    / 50
                ),
            )
            best_tau, best_score = estimate_time_offset_gridsearch(
                calibrator,
                calibration_params,
                min_s=-0.5,
                max_s=0.5,
                n=41,
                sample_step=sample_step,
            )
            print(
                f"\nEstimated time_offset = {best_tau:.3f}s (score {best_score:.3f})"
            )

            manual_metrics_snap = compare_against_manual(
                calibrator, dict(calibration_values), start_params
            )
            if gt_labels_path.exists():
                metrics = compare_against_labels(
                    calibrator,
                    dict(calibration_values),
                    start_params,
                    gt_labels,
                    step=1,
                )
                print(f"Metrics from labeled data: {metrics}")
                calibrator.calibration_params = MANUAL_MEASURED
                metrics_manual = compare_against_labels(
                    calibrator,
                    MANUAL_MEASURED,
                    start_params,
                    gt_labels,
                    step=1,
                )
                print(f"Manual Metrics from labeled data: {metrics_manual}")
            # calibrator.calibration_params = dict(calibration_params)
            save = False
            manual_metrics.append(manual_metrics_snap)
        # Calculate Statistical metrics
        save = False
        if manual_metrics and save:
            auto_chamfer = np.array(
                [i["qc"]["chamfer"] for i in manual_metrics]
            )
            auto_iou = np.array([i["qc"]["iou"] for i in manual_metrics])
            auto_coverage = np.array(
                [i["qc"]["coverage"] for i in manual_metrics]
            )
            auto_mi = np.array([i["qc"]["mi"] for i in manual_metrics])

            manual_chamfer = np.array(
                [i["qm"]["chamfer"] for i in manual_metrics]
            )
            manual_iou = np.array([i["qm"]["iou"] for i in manual_metrics])
            manual_coverage = np.array(
                [i["qm"]["coverage"] for i in manual_metrics]
            )
            manual_mi = np.array([i["qm"]["mi"] for i in manual_metrics])

            cen_manual = np.array(
                [(i["manual_distances"][0]).mean() for i in manual_metrics]
            )
            cham_manual = np.array(
                [(i["manual_distances"][1]).mean() for i in manual_metrics]
            )
            haus_manual = np.array(
                [(i["manual_distances"][2]).mean() for i in manual_metrics]
            )

            cen_auto = np.array(
                [(i["auto_distances"][0]).mean() for i in manual_metrics]
            )
            cham_auto = np.array(
                [(i["auto_distances"][1]).mean() for i in manual_metrics]
            )
            haus_auto = np.array(
                [(i["auto_distances"][2]).mean() for i in manual_metrics]
            )

            # Find Averages

            auto_chamfer_mean = np.mean(auto_chamfer)
            auto_iou_mean = np.mean(1 - auto_iou)
            auto_coverage_mean = np.mean(1 - auto_coverage)
            auto_mi_mean = np.mean(-auto_mi)

            manual_chamfer_mean = np.mean(manual_chamfer)
            manual_iou_mean = np.mean(1 - manual_iou)
            manual_coverage_mean = np.mean(1 - manual_coverage)
            manual_mi_mean = np.mean(-manual_mi)

            cen_manual_mean = np.mean(cen_manual)
            cham_manual_mean = np.mean(cham_manual)
            haus_manual_mean = np.mean(haus_manual)
            cen_auto_mean = np.mean(cen_auto)
            cham_auto_mean = np.mean(cham_auto)
            haus_auto_mean = np.mean(haus_auto)

            print("\nStatistical Metrics Across All Snapshots:")
            print(
                f"Auto Chamfer Mean: {auto_chamfer_mean:.4f}, Auto IoU Mean: {auto_iou_mean:.4f}, Auto Coverage Mean: {auto_coverage_mean:.4f}, Auto MI Mean: {auto_mi_mean:.4f}"
            )
            print(
                f"Manual Chamfer Mean: {manual_chamfer_mean:.4f}, Manual IoU Mean: {manual_iou_mean:.4f}, Manual Coverage Mean: {manual_coverage_mean:.4f}, Manual MI Mean: {manual_mi_mean:.4f}"
            )
            print(
                f"Manual Distances Mean: Cen: {cen_manual_mean:.4f}, Chamfer: {cham_manual_mean:.4f}, Hausdorff: {haus_manual_mean:.4f}"
            )
            print(
                f"Auto Distances Mean: Cen: {cen_auto_mean:.4f}, Chamfer: {cham_auto_mean:.4f}, Hausdorff: {haus_auto_mean:.4f}"
            )


if __name__ == "__main__":
    main()
