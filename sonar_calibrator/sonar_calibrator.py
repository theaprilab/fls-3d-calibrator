import cv2
import numpy as np
import scipy.optimize as opt
from scipy.spatial import KDTree
from scipy.stats import spearmanr

METRICS_LIST = ("chamfer", "iou", "coverage", "mi")
PARAM_KEYS = ("dy", "psi", "radius_scale", "az_scale")


class SonarCalibrator:
    """
    Class for calibrating sonar data with point cloud data. This class provides
    methods to perform calibration, generate confidence masks from sonar
    images, and overlay point cloud data onto sonar fan images for
    visualization. It also allows for the retrieval of the calibration matrix
    after calibration has been performed.
    """

    def __init__(
        self,
        calibration_data: dict,
        calibration_params: dict,
        penalty_weight: float = 0.3,
    ):
        """
        Initialize the SonarCalibrator with calibration data and an optional calibration matrix.

        :param calibration_data: A numpy array containing the calibration data.
        :param calibration_params: A dictionary containing the initial calibration parameters (dy, psi, radius_scale, az_scale).
        :param penalty_weight: A float representing the weight of the penalty term in the loss function
        """
        self.calibration_data = calibration_data
        self.calibration_params = calibration_params
        self.penalty_weight = penalty_weight
        self.loss_array = []
        self.calibration_array = []
        self.PARAM_KEYS = PARAM_KEYS
        self.METRICS_LIST = METRICS_LIST

    def _project_pcl_to_image(
        self,
        pcl: np.ndarray,
        image_shape: tuple,
        calibration_params: dict,
        sonar_display_range: float = 6.0,
        pcl_crop_range: float = 3.0,
    ) -> np.ndarray:
        """
        Projects the 3D point cloud data onto the 2D sonar image plane using the provided calibration matrix.

        :param pcl: A numpy array of shape (N, 3) representing the 3D point cloud data.
        :param image_shape: A tuple representing the shape of the sonar image (height, width).
        :param calibration_params: A numpy array representing the calibration matrix.
        :param sonar_display_range: A float representing the maximum range of the sonar. Default is 6.0 meters.
        :param pcl_crop_range: A float representing the maximum range of the point cloud to be considered for projection. Default is 3.0 meters.
        :return: A numpy array of shape (height, width) representing the projected points on the sonar image.
        :return cloud_idx: A numpy array of indices of the valid points in the point cloud that were projected onto the sonar image.
        """
        dy = calibration_params.get("dy", 0.0)
        psi = calibration_params.get("psi", 0.0)
        radius_scale = calibration_params.get("radius_scale", 1.0)
        az_scale = calibration_params.get("az_scale", 1.0)
        if radius_scale == 0 or az_scale == 0:
            raise ValueError(
                "radius_scale and az_scale must be non-zero values."
            )
        forward = np.cos(psi) * pcl[:, 0] - np.sin(psi) * pcl[:, 1]
        lateral = np.sin(psi) * pcl[:, 0] + np.cos(psi) * pcl[:, 1] + dy

        range = np.sqrt(forward**2 + lateral**2)

        valid_range = (range > 0.1) & (range < pcl_crop_range)
        range = range[valid_range]
        forward = forward[valid_range]
        lateral = lateral[valid_range]
        azimuth = np.arctan2(lateral, forward)
        # elevation = np.degrees(np.arctan2(vertical, range + 1e-6))
        h, w = image_shape
        cx = w * 0.5
        cy = float(h)
        radius_px = (range / sonar_display_range) * h * radius_scale

        x_px = cx + radius_px * np.sin(azimuth * az_scale)
        y_px = cy - radius_px * np.cos(azimuth * az_scale)
        good = (x_px >= 0) & (x_px < w) & (y_px >= 0) & (y_px < h)
        valid_idx = np.nonzero(valid_range)[0]
        cloud_idx = valid_idx[good]
        return np.stack((x_px, y_px), axis=-1).astype(np.int32), cloud_idx

    def _params_to_dict(self, params):
        # Helper function to convert a parameter LIST [dy, psi, radius_scale, az_scale] into the dict form
        return {k: v for k, v in zip(self.PARAM_KEYS, params)}

    def _fill(self, contour, image_shape):
        """
        Rasterize a single contour into a filled binary mask.
        Used to turn the target contour and the matched cloud contour into regions
        so IoU / coverage can be computed as pixel-area overlaps.

        :param contour: A numpy array of shape (M, 2) representing the contour points.
        :param image_shape: A tuple representing the shape of the sonar image (height, width).
        :return mask: A 2D numpy array representing the filled binary mask.
        """
        # Rasterize a single contour into a filled binary mask.
        # Used to turn the target contour and the matched cloud contour into regions
        # so IoU / coverage can be computed as pixel-area overlaps.
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        if contour is None:
            return mask
        pts = np.asarray(contour).reshape(-1, 2).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 255)
        return mask

    def _nearest_pcl_contour(self, params, pcl, image_shape, center_sonar):
        """
        Project the cloud at `params`, extract its contours, and return the one whose
        bounding-box center is closest to the sonar target center.

        :param params: A list or numpy array of calibration parameters [dy, psi, radius_scale, az_scale].
        :param pcl: A numpy array of shape (N, 3) representing the 3D point cloud data.
        :param image_shape: A tuple representing the shape of the sonar image (height, width).
        :param center_sonar: A numpy array of shape (2,) representing the center of the sonar contour.

        :return best_contour: A numpy array of shape (M, 2) representing the contour points of the nearest projected point cloud contour.
        """
        pcl_contours, _ = self.find_pcl_contours(
            pcl, image_shape, self._params_to_dict(params)
        )
        best_contour, best_distance = None, float("inf")
        for contour in pcl_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cc = np.array([x + w / 2, y + h / 2])
            distance = float(np.linalg.norm(cc - center_sonar))
            if distance < best_distance:
                best_distance, best_contour = distance, contour
        return best_contour

    def _matched_masks(
        self, params, pcl, image_shape, contour_sonar, center_sonar
    ):
        """
        Computes the filled binary masks for the sonar target contour and the nearest
        projected point cloud contour based on the provided calibration parameters.
        These masks are used to calculate overlap metrics.

        :param params: A list or numpy array of calibration parameters [dy, psi, radius_scale, az_scale].
        :param pcl: A numpy array of shape (N, 3) representing the 3D point cloud data.
        :param image_shape: A tuple representing the shape of the sonar image (height, width).
        :param contour_sonar: A numpy array of shape (M, 2) representing the contour points of the sonar image.
        :param center_sonar: A numpy array of shape (2,) representing the center of the sonar contour.

        :return: A tuple containing two 2D numpy arrays representing the
                 filled binary masks for the sonar target contour and the
                 nearest projected point cloud contour, respectively.
        """
        smask = self._fill(contour_sonar, image_shape)
        pmask = self._fill(
            self._nearest_pcl_contour(params, pcl, image_shape, center_sonar),
            image_shape,
        )
        return smask, pmask

    def chamfer_loss(
        self, params, pcl_raw, image_shape, contour_sonar, center_sonar
    ):
        """
        Computes the Chamfer loss between the contours of the point cloud and
        the sonar image. This function calculates the average distance from
        each point in one contour to the nearest point in the other contour,
        and vice versa. The Chamfer loss is a measure of similarity between
        two point sets, and it is used here to evaluate how well the point
        cloud aligns with the sonar image after projection.

        :param params: A list or numpy array of calibration parameters [dy, psi, radius_scale, az_scale].
        :param pcl_raw: A numpy array of shape (N, 3) representing the raw 3D point cloud data.
        :param image_shape: A tuple representing the shape of the sonar image (height, width).
        :param contour_sonar: A numpy array of shape (M, 2) representing the contour points of the sonar image.
        :param center_sonar: A numpy array of shape (2,) representing the center of the sonar contour.
        :return: A float representing the Chamfer loss between the two contours.
        """
        dy, psi, radius_scale, az_scale = params
        current_guess = {
            "dy": dy,
            "psi": psi,
            "radius_scale": radius_scale,
            "az_scale": az_scale,
        }

        dy0 = self.calibration_params.get("dy", 0.0)
        psi0 = self.calibration_params.get("psi", 0.0)
        r0 = self.calibration_params.get("radius_scale", 1.0)
        az0 = self.calibration_params.get("az_scale", 1.0)

        pcl_contours, _ = self.find_pcl_contours(
            pcl_raw, image_shape, current_guess
        )
        if len(pcl_contours) == 0:
            return float("inf")
        best_pcl_contour = None
        min_distance = float("inf")
        for contour in pcl_contours:
            x_p, y_p, w_p, h_p = cv2.boundingRect(contour)
            center_pcl = np.array([x_p + w_p / 2, y_p + h_p / 2])
            distance = np.linalg.norm(center_pcl - center_sonar)
            if distance < min_distance:
                min_distance = distance
                best_pcl_contour = contour

        if best_pcl_contour is None or len(best_pcl_contour) == 0:
            return float("inf")
        contour_pcl_dynamic = best_pcl_contour.squeeze(axis=1).astype(
            np.float32
        )
        tree_pcl = KDTree(contour_pcl_dynamic)
        tree_sonar = KDTree(contour_sonar)

        distances_pcl_to_sonar, _ = tree_sonar.query(contour_pcl_dynamic, k=1)
        distances_sonar_to_pcl, _ = tree_pcl.query(contour_sonar, k=1)

        chamfer_loss = np.mean(distances_pcl_to_sonar**2) + np.mean(
            distances_sonar_to_pcl**2
        )
        penalty = (
            (abs(dy - dy0) / 0.05) ** 2
            + (abs(psi - psi0) / 0.05) ** 2
            + (abs(radius_scale - r0) / 0.05) ** 2
            + (abs(az_scale - az0) / 0.05) ** 2
        ) * self.penalty_weight
        chamfer_loss += penalty
        return chamfer_loss
        # saved = self.penalty_weight
        # self.penalty_weight = 0.0
        # try:
        #     return self._loss_function(params, pcl, image_shape, contour_sonar, center_sonar)
        # finally:
        #     self.penalty_weight = saved

    def iou_loss(self, params, pcl, image_shape, contour_sonar, center_sonar):
        """
        Intersection-over-Union (IoU) loss between the sonar target contour and
        the nearest projected point cloud contour. This function computes the
        filled binary masks for both contours and calculates the IoU metric,
        which measures the overlap between the two regions. The IoU loss is
        defined as 1 minus the IoU value, so a lower loss indicates better
        alignment between the contours.

        :param params: A list or numpy array of calibration parameters [dy, psi, radius_scale, az_scale].
        :param pcl: A numpy array of shape (N, 3) representing the 3D point cloud data.
        :param image_shape: A tuple representing the shape of the sonar image (height, width).
        :param contour_sonar: A numpy array of shape (M, 2) representing the contour points of the sonar image.
        :param center_sonar: A numpy array of shape (2,) representing the center of the sonar contour.

        :return: A float representing the IoU loss between the two contours.

        """
        # 1 - Intersection-over-Union
        sonar_mask, pcl_mask = self._matched_masks(
            params, pcl, image_shape, contour_sonar, center_sonar
        )
        sb, pb = sonar_mask > 0, pcl_mask > 0
        if pb.sum() == 0:
            return 1.0
        inter = np.logical_and(sb, pb).sum()
        union = np.logical_or(sb, pb).sum()
        return 1.0 - (inter / union if union > 0 else 0.0)

    def coverage_loss(
        self,
        params,
        pcl,
        image_shape,
        contour_sonar,
        center_sonar,
        direction="sonar",
    ):
        """
        Computes the coverage loss between the sonar target contour and the nearest
        projected point cloud contour. This function calculates the filled binary
        masks for both contours and evaluates the fraction of one contour that is
        covered by the other. The coverage loss is defined as 1 minus the coverage
        value, so a lower loss indicates better alignment between the contours.

        :param params: A list or numpy array of calibration parameters [dy, psi, radius_scale, az_scale].
        :param pcl: A numpy array of shape (N, 3) representing the 3D point cloud data.
        :param image_shape: A tuple representing the shape of the sonar image (height, width).
        :param contour_sonar: A numpy array of shape (M, 2) representing the contour points of the sonar image.
        :param center_sonar: A numpy array of shape (2,) representing the center of the sonar contour.
        :param direction: A string indicating the direction of coverage calculation. If "sonar", the loss
                          is based on the fraction of the sonar contour covered by the point cloud contour.
                          If "pcl", the loss is based on the fraction of the point cloud contour covered by
                          the sonar contour. Default is "sonar".

        :return: A float representing the coverage loss between the two contours.
        """
        # 1 - polygon coverage.
        # direction="sonar": fraction of the target blob covered by the matched cloud blob;
        # direction="pcl": fraction of the cloud blob that lands on the target.
        sonar_mask, pcl_mask = self._matched_masks(
            params, pcl, image_shape, contour_sonar, center_sonar
        )
        sb, pb = sonar_mask > 0, pcl_mask > 0
        if pb.sum() == 0:
            return 1.0
        inter = np.logical_and(sb, pb).sum()
        denom = sb.sum() if direction == "sonar" else pb.sum()
        return 1.0 - (inter / denom if denom > 0 else 0.0)

    def mi_loss(
        self,
        params,
        pcl,
        image_shape,
        contour_sonar,
        center_sonar,
        sonar_gray,
        bins=32,
        blur=3.0,
        pad=25,
    ):
        """
        Computes the Mutual Information (MI) loss between the sonar intensity and
        the matched point cloud density. This function calculates the filled binary
        masks for both contours, applies Gaussian blurring to the point cloud density,
        and computes the mutual information between the sonar intensity values and
        the point cloud density values. The MI loss is defined as the negative of
        the mutual information value, so a lower loss indicates better alignment
        between the sonar intensity and the point cloud density.

        :param params: A list or numpy array of calibration parameters [dy, psi, radius_scale, az_scale].
        :param pcl: A numpy array of shape (N, 3) representing the 3D point cloud data.
        :param image_shape: A tuple representing the shape of the sonar image (height, width).
        :param contour_sonar: A numpy array of shape (M, 2) representing the contour points of the sonar image.
        :param center_sonar: A numpy array of shape (2,) representing the center of the sonar contour.
        :param sonar_gray: A 2D numpy array representing the grayscale sonar image.
        :param bins: An integer representing the number of bins to use for the histogram calculation. Default is 32.
        :param blur: A float representing the standard deviation for Gaussian blurring. Default is 3.0.
        :param pad: An integer representing the padding to apply around the bounding box of the sonar contour. Default is 25.

        :return: A float representing the Mutual Information loss between the sonar intensity and the matched point cloud density.
        """
        # Mutual Information between sonar intensity and matched-cloud density,
        sonar_mask, pcl_mask = self._matched_masks(
            params, pcl, image_shape, contour_sonar, center_sonar
        )
        pts = np.asarray(contour_sonar).reshape(-1, 2).astype(np.int32)
        x, y, ww, hh = cv2.boundingRect(pts)
        h, w = image_shape
        x0, x1 = max(0, x - pad), min(w, x + ww + pad)
        y0, y1 = max(0, y - pad), min(h, y + hh + pad)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        density_image = cv2.GaussianBlur(
            pcl_mask.astype(np.float32), (0, 0), blur
        )[y0:y1, x0:x1]

        gray_image = sonar_gray[y0:y1, x0:x1]
        if density_image.size == 0 or np.ptp(density_image) == 0:
            return 0.0
        hist, _, _ = np.histogram2d(
            gray_image.ravel(), density_image.ravel(), bins=bins
        )
        joint_prob = hist / (hist.sum() + 1e-12)
        marginal_sonar_prob, marginal_density_prob = (
            joint_prob.sum(1),
            joint_prob.sum(0),
        )
        nonzero_prob = joint_prob > 0
        mi = np.sum(
            joint_prob[nonzero_prob]
            * np.log(
                joint_prob[nonzero_prob]
                / (
                    (
                        marginal_sonar_prob[:, None]
                        * marginal_density_prob[None, :]
                    )[nonzero_prob]
                )
            )
        )
        return -float(mi)

    def generate_mask(
        self,
        sonar_image: np.ndarray,
        threshold: float = 99.5,
        kernel_size: int = 3,
    ) -> np.ndarray:
        """
        Generates a confidence mask from the 2D sonar image. This mask can be
        adjusted to highlight areas of interest or to filter out noise based
        on the sonar image. It also allows for convenient visualization of the
        sonar data in conjunction with the point cloud. This also helps in
        identifying the regions of the sonar image that correspond to valid
        features in the point cloud, which can be useful for calibration and
        analysis.

        :param sonar_image: A 2D numpy array representing the sonar image.
        :param threshold: An integer value representing the threshold for
                          generating the mask. Default is 95, which means that
                          pixels with intensity values above this threshold
                          will be considered as valid features in the mask.
        :param kernel_size: An integer value representing the size of the
                            kernel used for morphological operations. Default is 3.
        :return: A 2D numpy array representing the confidence mask.
        """
        # Placeholder for generating a mask from the sonar image
        # This function can be implemented to create a mask based on the sonar image if needed

        gray = cv2.cvtColor(sonar_image, cv2.COLOR_BGR2GRAY)
        threshold_percentile = np.percentile(gray, threshold)
        _, mask = cv2.threshold(
            gray, threshold_percentile, 255, cv2.THRESH_BINARY
        )
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def multi_variable_optimization(self, pcl, contour_sonar, image_shape):
        """
        Performs multi-variable optimization to find the best calibration parameters
        that minimize the Chamfer loss between the contours of the point cloud and
        the sonar image. This function uses the Nelder-Mead optimization method to
        iteratively adjust the calibration parameters and evaluate the loss until
        convergence is achieved.

        :param initial_params: A list or numpy array of initial calibration parameters.
        :param contour_pcl: A numpy array of shape (N, 2) representing the
                            contour points of the point cloud.
        :param contour_sonar: A numpy array of shape (M, 2) representing the
                              contour points of the sonar image.
        :param image_shape: A tuple representing the shape of the sonar image (height, width).
        :return: None
        """

        contour_sonar_pts = contour_sonar.squeeze(axis=1).astype(np.float32)
        x_s, y_s, w_s, h_s = cv2.boundingRect(contour_sonar)
        center_sonar = np.array([x_s + w_s / 2, y_s + h_s / 2])
        initial_params = [
            self.calibration_params.get("dy", 0.0),
            self.calibration_params.get("psi", 0.0),
            self.calibration_params.get("radius_scale", 1.0),
            self.calibration_params.get("az_scale", 1.0),
        ]

        result = opt.minimize(
            self.chamfer_loss,
            initial_params,
            args=(pcl, image_shape, contour_sonar_pts, center_sonar),
            method="Powell",
            options={"maxiter": 100, "disp": False},
        )
        self.calibration_params = {
            "dy": result.x[0],
            "psi": result.x[1],
            "radius_scale": result.x[2],
            "az_scale": result.x[3],
        }
        self.loss_array.append(result.fun)
        self.calibration_array.append(self.calibration_params.copy())
        print(
            f"""Optimized calibration parameters: {self.calibration_params}
            with optimization result: {result.success},
             message: {result.message}
            and final loss: {result.fun}"""
        )

    def display_confidence_image(self, data):
        """
        Displays the confidence mask generated from the sonar image. This
        function provides a visual representation of the areas in the sonar
        image that are considered valid features based on the thresholding
        and morphological operations applied to generate the mask.

        :param data: A 2D numpy array representing the confidence mask.
        """
        mask = self.generate_mask(data)
        contours, mask = self.find_countours(mask)
        cv2.imshow(
            "Sonar mask",
            mask,
        )
        cv2.waitKey(50)

    def find_countours(self, mask: np.ndarray):
        """
        Finds contours in the given mask.

        :param mask: A 2D numpy array representing the confidence mask.
        :return: A list of contours found in the mask.
        """
        contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        output_mask = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
        clean_contours = [
            cnt for cnt in contours if cv2.contourArea(cnt) > 1000
        ]
        for cnt in contours:
            if cv2.contourArea(cnt) < 1000:
                continue
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(output_mask, [box], 0, (0, 255, 0), 2)

        return clean_contours, output_mask

    def find_pcl_contours(
        self, pcl: np.ndarray, image_shape: tuple, calibration_params: dict
    ):
        """
        Finds contours in the projected point cloud data. This function projects
        the 3D point cloud onto the 2D sonar image plane using the provided
        calibration parameters and then applies contour detection to identify
        the boundaries of the projected point cloud in the sonar image.

        :param pcl: A numpy array of shape (N, 3) representing the 3D point cloud data.
        :param image_shape: A tuple representing the shape of the sonar image (height, width).
        :param calibration_params: A numpy array representing the calibration matrix.
        :return: A list of contours found in the projected point cloud data.
        """

        pcl, _ = self._project_pcl_to_image(
            pcl, image_shape, calibration_params
        )

        h, w = image_shape
        pcl_mask = np.zeros((h, w), dtype=np.uint8)

        valid_x = (pcl[:, 0] >= 0) & (pcl[:, 0] < w)
        valid_y = (pcl[:, 1] >= 0) & (pcl[:, 1] < h)
        valid_points = pcl[valid_x & valid_y]
        pcl_mask[valid_points[:, 1], valid_points[:, 0]] = 255
        pcl_mask = cv2.GaussianBlur(pcl_mask, (5, 5), 0)
        pcl_mask = cv2.Canny(pcl_mask, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated_mask = cv2.morphologyEx(pcl_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            dilated_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        clean_contours = [
            cnt for cnt in contours if cv2.contourArea(cnt) > 100
        ]
        return clean_contours, dilated_mask

    def display_collapsed_cloud(
        self, pcl, sonar_image, image_size, color=None, save_frames=False
    ):
        """
        Displays the collapsed point cloud over the sonar image. This function
        overlays the projected point cloud onto the sonar image, allowing for
        visual comparison of the two data sources. It also highlights the contours
        of the point cloud and the sonar image, providing a clear view of how well
        the point cloud aligns with the sonar data after calibration.

        :param pcl: A numpy array of shape (N, 3) representing the 3D point cloud data.
        :param sonar_image: A 2D numpy array representing the sonar image.
        :param image_size: A tuple representing the shape of the sonar image (height, width).
        :param color: An optional tuple representing the color to use for drawing the point cloud.
        :param save_frames: An optional boolean indicating whether to save the frame as an image.

        :return: None
        """

        background_img = cv2.cvtColor(sonar_image.copy(), cv2.COLOR_RGB2BGR)
        mask = self.generate_mask(background_img)
        contours_2d, mask = self.find_countours(mask)
        overlay = background_img.copy()

        collapsed_pcl, _ = self._project_pcl_to_image(
            pcl, image_size, self.calibration_params
        )
        contours, dilated_mask = self.find_pcl_contours(
            pcl, image_size, self.calibration_params
        )
        collapsed_pcl = np.round(collapsed_pcl).astype(np.int32)
        draw_color = color if color is not None else (0, 0, 255)
        for pt in collapsed_pcl:
            x, y = pt[0], pt[1]

            # Guard clause: Only draw points that actually fall inside the image frame
            if 0 <= x < image_size[1] and 0 <= y < image_size[0]:
                cv2.circle(
                    overlay,
                    (x, y),
                    radius=4,
                    color=draw_color,
                    thickness=-1,
                )

        alpha = 1.0
        beta = 0.6
        background_img = cv2.addWeighted(
            overlay, beta, background_img, alpha - beta, 0
        )
        cv2.drawContours(background_img, contours, -1, (0, 255, 106), 2)
        cv2.drawContours(background_img, contours_2d, -1, (0, 246, 255), 2)
        pc_mask = cv2.cvtColor(dilated_mask.copy(), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(pc_mask, contours, -1, (0, 255, 106), 2)
        cv2.imshow("Collapsed Point Cloud", background_img)
        cv2.imshow("Dilated Mask", pc_mask)
        cv2.waitKey(500)
        if save_frames:
            return background_img, pc_mask, overlay

    # =========================================================================
    #  COMBINED-METRIC CALIBRATION
    #  ------------------------------------------------------------------------
    #  Added three extra  metrics (IoU, polygon coverage, mutual information).
    #  Normalized using z-score normalization.
    #  Weighted using perturbations and Spearman rank correlation
    # =========================================================================

    def all_losses(
        self,
        params,
        pcl,
        image_shape,
        contour_sonar,
        center_sonar,
        sonar_mask,
        sonar_gray,
    ):
        return {
            "chamfer": self.chamfer_loss(
                params, pcl, image_shape, contour_sonar, center_sonar
            ),
            "iou": self.iou_loss(
                params, pcl, image_shape, contour_sonar, center_sonar
            ),
            "coverage": self.coverage_loss(
                params, pcl, image_shape, contour_sonar, center_sonar
            ),
            "mi": self.mi_loss(
                params,
                pcl,
                image_shape,
                contour_sonar,
                center_sonar,
                sonar_gray,
            ),
        }

    def fit_norm(
        self,
        pcl,
        image_shape,
        contour_sonar,
        center_sonar,
        sonar_mask,
        sonar_gray,
        center_params,
        n=80,
        scales=None,
        seed=0,
    ):
        """
        Z-score normalization. This measures, per metric, the median and a robust spread
        (16-84th percentile half-width) by sampling `n` random perturbations around the
        optimization start. The normalization is completed as the metrics are not weighted
        and are on different scales. This function computes the median and robust spread for
        each metric by sampling random perturbations around the provided center parameters.
        The resulting normalization values are stored in the `metric_norm` attribute of the class.

        :param pcl: A numpy array of shape (N, 3) representing the 3D point cloud data.
        :param image_shape: A tuple representing the shape of the sonar image (height, width
        :param contour_sonar: A numpy array of shape (M, 2) representing the contour points of the sonar image.
        :param center_sonar: A numpy array of shape (2,) representing the center of the sonar contour.
        :param sonar_mask: A 2D numpy array representing the confidence mask
        :param sonar_gray: A 2D numpy array representing the grayscale sonar image.
        :param center_params: A dictionary containing the center parameters for calibration.
        :param n: An integer representing the number of random perturbations to sample. Default is 80.
        :param scales: A dictionary containing the scales for each parameter. If None, default scales will be used. Default is None.
        :param seed: An integer representing the random seed for reproducibility. Default is 0.
        :return: A dictionary containing the median and robust spread for each metric, stored in the `metric_norm` attribute of the class.
        """
        contour_points = (
            contour_sonar.squeeze(axis=1).astype(np.float32)
            if getattr(contour_sonar, "ndim", 2) == 3
            else np.asarray(contour_sonar, np.float32)
        )
        base = np.array(
            [
                center_params.get(k, d)
                for k, d in (
                    ("dy", 0.0),
                    ("psi", 0.0),
                    ("radius_scale", 1.0),
                    ("az_scale", 1.0),
                )
            ]
        )
        if scales is None:
            scales = {
                "dy": 0.25,
                "psi": 0.15,
                "radius_scale": 0.25,
                "az_scale": 0.25,
            }
        idx = {"dy": 0, "psi": 1, "radius_scale": 2, "az_scale": 3}
        rng = np.random.default_rng(seed)
        metric_values = {m: [] for m in self.METRICS_LIST}
        samples = [base.copy()]
        # Sample random perturbations around the base parameters
        for _ in range(n):
            param = base.copy()
            for k, s in scales.items():
                param[idx[k]] = base[idx[k]] + rng.uniform(-s, s)
            samples.append(param)
        # Evaluate all metrics for each sampled parameter set
        for param in samples:
            raw = self.all_losses(
                list(param),
                pcl,
                image_shape,
                contour_points,
                center_sonar,
                sonar_mask,
                sonar_gray,
            )
            for metric in self.METRICS_LIST:
                if np.isfinite(raw[metric]):
                    metric_values[metric].append(raw[metric])
        # Compute the median and robust spread for each metric
        norm = {}
        for metric in self.METRICS_LIST:
            values = np.array(metric_values[metric], float)
            if values.size >= 4:
                med = float(np.median(values))
                scale = float(
                    (np.percentile(values, 84) - np.percentile(values, 16))
                    / 2.0
                )  # ~1 std, robust
                if not np.isfinite(scale) or scale <= 1e-9:
                    scale = float(np.std(values)) or 1.0
            else:
                med, scale = 0.0, 1.0
            norm[metric] = (med, scale)
        self.metric_norm = norm
        return norm

    def combined_loss(
        self,
        params,
        pcl,
        image_shape,
        contour_sonar,
        center_sonar,
        sonar_mask,
        sonar_gray,
        weights,
    ):
        """
        Combines multiple loss metrics into a single loss value using specified
        weights. This function computes the individual losses for each metric,
        normalizes them using z-score normalization if applicable, and then
        combines them into a total loss value based on the provided weights.
        Additionally, it applies a regularization term to penalize deviations
        from the initial calibration parameters.

        :param params: A list or numpy array of calibration parameters [dy, psi, radius_scale, az_scale].
        :param pcl: A numpy array of shape (N, 3) representing the 3D point cloud data.
        :param image_shape: A tuple representing the shape of the sonar image (height, width).
        :param contour_sonar: A numpy array of shape (M, 2) representing the contour points of the sonar image.
        :param center_sonar: A numpy array of shape (2,) representing the center of the sonar contour.
        :param sonar_mask: A 2D numpy array representing the confidence mask.
        :param sonar_gray: A 2D numpy array representing the grayscale sonar image.
        :param weights: A dictionary containing the weights for each metric to be combined.

        :return: A float representing the combined loss value
        """
        raw = self.all_losses(
            params,
            pcl,
            image_shape,
            contour_sonar,
            center_sonar,
            sonar_mask,
            sonar_gray,
        )
        norm = getattr(self, "metric_norm", None)
        total = 0.0
        for k, wt in weights.items():
            if wt == 0:
                continue
            v = raw[k]
            if norm and k in norm:
                med, sc = norm[k]
                v = (v - med) / (sc + 1e-12)
            if not np.isfinite(v):
                v = 3.0
            total += wt * float(np.clip(v, -5.0, 5.0))
        dy, psi, rs, az = params
        total += getattr(self, "scale_reg", 0.3) * (
            (rs - 1.0) ** 2 + (az - 1.0) ** 2
        )
        return total

    def optimize_weighted(
        self,
        pcl,
        image_shape,
        contour_sonar,
        center_sonar,
        sonar_mask,
        sonar_gray,
        weights,
        maxiter=200,
        refit_norm=True,
    ):
        """
        Optimizes the calibration parameters using a weighted combination of multiple loss metrics.
        This function performs optimization to find the best calibration parameters that minimize
        the combined loss value, taking into account the specified weights for each metric. It also
        allows for optional refitting of the z-score normalization around the current start parameters.

        :param pcl: A numpy array of shape (N, 3) representing the 3D point cloud data.
        :param image_shape: A tuple representing the shape of the sonar image (height, width).
        :param contour_sonar: A numpy array of shape (M, 2) representing the contour points of the sonar image.
        :param center_sonar: A numpy array of shape (2,) representing the center of the sonar contour.
        :param sonar_mask: A 2D numpy array representing the confidence mask.
        :param sonar_gray: A 2D numpy array representing the grayscale sonar image.
        :param weights: A dictionary containing the weights for each metric to be combined.
        :param maxiter: An integer representing the maximum number of iterations for the optimization process. Default is 200.
        :param refit_norm: A boolean indicating whether to refit the z-score normalization around the current start parameters. Default is True.
        :return: The result of the optimization process, which includes the optimized calibration parameters and other relevant information.
        """
        # Refits the z-score normalization around the current start
        cpts = contour_sonar.squeeze(axis=1).astype(np.float32)
        x0 = [
            self.calibration_params.get(k, d)
            for k, d in (
                ("dy", 0.0),
                ("psi", 0.0),
                ("radius_scale", 1.0),
                ("az_scale", 1.0),
            )
        ]
        if refit_norm:
            self.fit_norm(
                pcl,
                image_shape,
                contour_sonar,
                center_sonar,
                sonar_mask,
                sonar_gray,
                center_params=self._params_to_dict(x0),
            )
        result = opt.minimize(
            self.combined_loss,
            x0,
            args=(
                pcl,
                image_shape,
                cpts,
                center_sonar,
                sonar_mask,
                sonar_gray,
                weights,
            ),
            method="Powell",
            options={
                "maxiter": maxiter,
                "xtol": 1e-4,
                "ftol": 1e-4,
                "disp": False,
            },
        )
        self.calibration_params = dict(zip(self.PARAM_KEYS, result.x))
        return result

    def weight_by_prediction(
        self,
        pcl,
        image_shape,
        contour_sonar,
        center_sonar,
        sonar_mask,
        sonar_gray,
        base_params,
        n_samples=200,
        scales=None,
        power=1.0,
        seed=0,
    ):
        """Decides how much each metric should count by measuring how well it predicts true misalignment.
         Draw n_samples random perturbations of normalized distance from base_params, evaluate every metric at each, then
         rank-correlate (Spearman) each metric's value against that true distance.

        :param pcl: A numpy array of shape (N, 3) representing the 3D point cloud data.
        :param image_shape: A tuple representing the shape of the sonar image (height, width).
        :param contour_sonar: A numpy array of shape (M, 2) representing the contour points of the sonar image.
        :param center_sonar: A numpy array of shape (2,) representing the center of the sonar contour.
        :param sonar_mask: A 2D numpy array representing the confidence mask.
        :param sonar_gray: A 2D numpy array representing the grayscale sonar image.
        :param base_params: A dictionary containing the base calibration parameters for perturbation.
        :param n_samples: An integer representing the number of random perturbations to sample. Default is 200.
        :param scales: A dictionary containing the scales for each parameter. If None, default scales will be used. Default is None.
        :param power: A float representing the power to which the rank correlation values are raised when computing weights. Default is 1.
        :param seed: An integer representing the random seed for reproducibility. Default is 0
        :return: A dictionary containing the weights for each metric based on their predictive power, and the rank correlation values for each metric.

        """

        contour_points = contour_sonar.squeeze(axis=1).astype(np.float32)
        base = np.array(
            [
                base_params.get(k, d)
                for k, d in (
                    ("dy", 0.0),
                    ("psi", 0.0),
                    ("radius_scale", 1.0),
                    ("az_scale", 1.0),
                )
            ]
        )
        if scales is None:
            scales = {
                "dy": 0.3,
                "psi": 0.2,
                "radius_scale": 0.3,
                "az_scale": 0.3,
            }
        idx = {"dy": 0, "psi": 1, "radius_scale": 2, "az_scale": 3}
        rng = np.random.default_rng(seed)
        metrics, dists, vals = self.evaluate_metrics_with_perturbations(
            pcl,
            image_shape,
            center_sonar,
            sonar_mask,
            sonar_gray,
            n_samples,
            scales,
            contour_points,
            base,
            idx,
            rng,
        )
        dists = np.array(dists)
        rho = {}
        for m in metrics:
            v = np.array(vals[m], float)
            fin = np.isfinite(v)
            if fin.sum() >= 5 and np.ptp(v[fin]) > 0:
                r = spearmanr(dists[fin], v[fin]).statistic
                rho[m] = 0.0 if (r is None or np.isnan(r)) else float(r)
            else:
                rho[m] = 0.0
        score = {m: max(0.0, rho[m]) ** power for m in metrics}
        tot = sum(score.values())
        weights = {m: (score[m] / tot if tot > 0 else 0.0) for m in metrics}
        print("predictive weighting:")
        print(f"  {'metric':9s} {'rank-corr':>9s} {'weight':>7s}")
        for m in metrics:
            print(f"  {m:9s} {rho[m]:>9.2f} {weights[m]:>7.3f}")
        return {"weights": weights, "rank_corr": rho}

    def evaluate_metrics_with_perturbations(
        self,
        pcl,
        image_shape,
        center_sonar,
        sonar_mask,
        sonar_gray,
        n_samples,
        scales,
        contour_points,
        base,
        idx,
        rng,
    ):
        metrics = list(self.METRICS_LIST)
        dists, vals = [], {m: [] for m in metrics}
        for _ in range(n_samples):
            p = base.copy()
            dd = 0.0
            for k, s in scales.items():
                o = rng.uniform(-s, s)
                p[idx[k]] = base[idx[k]] + o
                dd += (o / s) ** 2
            dists.append(np.sqrt(dd))
            raw = self.all_losses(
                list(p),
                pcl,
                image_shape,
                contour_points,
                center_sonar,
                sonar_mask,
                sonar_gray,
            )
            for m in metrics:
                vals[m].append(raw[m])
        return metrics, dists, vals

    def denoise_pointcloud(
        self, pcl: np.ndarray, sonar_image: np.ndarray, threshold: float = 0.5
    ):
        x = pcl[:, 0]
        y = pcl[:, 1]
        z = pcl[:, 2]
        projected_image, cloud_idx = self._project_pcl_to_image(
            pcl, sonar_image.shape[:2], self.calibration_params
        )
        mask = self.generate_mask(sonar_image)
        x_pix = projected_image[:, 0]
        y_pix = projected_image[:, 1]
        h, w = sonar_image.shape[:2]
        confidence = np.zeros(len(pcl), dtype=np.float32)
        for px, py, idx in zip(x_pix, y_pix, cloud_idx):
            r = 20
            y0 = max(0, py - r)
            y1 = min(py + r + 1, h)

            x0 = max(0, px - r)
            x1 = min(px + r + 1, w)
            patch = mask[y0:y1, x0:x1]
            confidence[idx] = patch.max() / 255.0

        good = confidence > threshold
        bad = ~good
        good_pts = pcl[good]
        return good_pts

    def closest_frame(self, times, t, max_dt_ns=None):
        """
        Finds the index of the closest frame in the given times array to
        the specified time t.

        :param times: A 1D numpy array of timestamps (in nanoseconds) for
                      the frames.
        :param t: A timestamp (in nanoseconds) for which to find the
                  closest frame.
        :param max_dt_ns: An optional maximum allowed time difference
                          (in nanoseconds) for a valid closest frame.
                          If the closest frame is further away than this
                          value, None is returned.
        :return: The index of the closest frame in the times array, or None if
                 no valid frame is found within max_dt_ns.
        """

        times = np.asarray(times)
        if times.size == 0:
            raise ValueError("times is empty")
        if not np.all(np.diff(times) >= 0):
            raise ValueError("times must be sorted ascending")

        idx = np.searchsorted(times, t, side="left")
        if idx == 0:
            cand = 0
        elif idx == len(times):
            cand = len(times) - 1
        else:
            before = times[idx - 1]
            after = times[idx]
            cand = idx - 1 if (t - before) <= (after - t) else idx

        dt = abs(int(times[cand]) - int(t))  # nanoseconds
        if max_dt_ns is not None and dt > int(max_dt_ns):
            return None
        return int(cand)

    def calibrate(self):
        """
        Finds the best calibration parameters that minimize the Chamfer loss between
        the contours of the point cloud and the sonar image. This function just uses
        the calibration parameter array and the loss array to find the best
        calibration parameters that minimize the Chamfer loss

        :return: None
        """

        min_loss_idx = np.argmin(self.loss_array[20:]) + 20
        best_params = self.calibration_array[min_loss_idx]
        self.calibration_params = best_params
        print(
            f"""Best calibration parameters found: {self.calibration_params}
            with loss: {self.loss_array[min_loss_idx]}"""
        )
