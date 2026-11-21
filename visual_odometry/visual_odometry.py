import rclpy
import numpy as np
import cv2
from cv_bridge import CvBridge
from rclpy.node import Node
from interfaces.msg import VehicleState
from std_msgs.msg import Empty, Float64
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation 
from collections import deque
import math
from geometry_msgs.msg import PoseStamped

# Optional: non-blocking real-time plotting
try:
    import matplotlib
    # Try macOS native backend first, then common interactive backends
    for _backend in ['MacOSX', 'TkAgg', 'Qt5Agg']:
        try:
            matplotlib.use(_backend)
            break
        except Exception:
            continue
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Part of the following code has been taken from: https://github.com/SharanIO/Visual-Odometry/blob/main/src/svo.py
# The main idea for the algorithm was taken from: https://webthesis.biblio.polito.it/12500/1/tesi.pdf
# General algorithm overview: 
# We obtain the left and right images, 
# we compute the odometry for each side as if they are monocular, 
# we average the 2 rotation matrices obtained and then we multiply by the previously found matrices.
 
class VisualOdometryNode(Node):
    """
    Class for the Visual Odometry Node. 
    For the moment we use it only to double check the IMU data. 
    """

    def __init__(self):
        super().__init__('visual_odometry_node')
        
        self.declare_parameter("camera_source_left", "LEFT_CAMERA_TOPIC")
        self.declare_parameter("camera_source_right", "RIGHT_CAMERA_TOPIC")
        self.declare_parameter("enable_vo_yaw_plot", False)
        self.camera_source_left = self.get_parameter("camera_source_left").get_parameter_value().string_value
        self.camera_source_right = self.get_parameter("camera_source_right").get_parameter_value().string_value
        self.cv_bridge = CvBridge()
        self.left_images = []
        self.right_images = []
        # real cameras intrinsics 
        self.camera_intrinsic = np.asarray([[820., 0.0, 430.0],
                                      [0.0, 700.0, -5.],
                                      [0.0, 0.0, 1.0]])
        self.x = self.y = self.yaw = ...
        self.v_x = self.v_y = self.yaw_r = ...
        self.s = self.v_s = ...
        self.delta = self.d = ...
        self.starting = True
        self.t_yaw = 0
        # FIX: use a 3x3 rotation accumulator (was 4x4, inconsistent with 3x3 R)
        self.rotation_matrix = np.eye(3)
        self.ground_truth_yaw = ...
        # Bias estimation between IMU yaw and ground-truth yaw
        self._yaw_bias_samples = deque(maxlen=200)
        self._yaw_bias_estimate = 0.0
        self.publisher = self.create_publisher(VehicleState, 'vehicle_state_measure', 10)

        # we want the images recorded in input
        self.image_subscription_left = self.create_subscription(Image, self.camera_source_left, self.image_callback_left, 10)
        self.image_subscription_right = self.create_subscription(Image, self.camera_source_right, self.image_callback_right, 10)
        self.kill_subscription = self.create_subscription(Empty, 'kill', self.kill_callback, 10)
        self.state_estimation_subscription = self.create_subscription(VehicleState, 'vehicle_state_measure', self.state_estimation_callback, 10)
        self.ground_truth_subscription = self.create_subscription(VehicleState, 'vehicle_state', self.ground_truth_callback, 10)
        self.get_logger().info("Visual Odometry node initialized")

        # Publishers for RViz and rqt_plot
        self.vo_yaw_pub = self.create_publisher(Float64, 'vo_yaw', 10)
        self.vo_pose_pub = self.create_publisher(PoseStamped, 'vo_pose', 10)

        # Real-time plotting (if enabled and matplotlib available). Does not block rclpy.spin
        param_enable_plot = self.get_parameter("enable_vo_yaw_plot").get_parameter_value().bool_value
        self.enable_plot = bool(param_enable_plot and (plt is not None))
        if self.enable_plot:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.line_vo, = self.ax.plot([], [], label='VO yaw [rad]')
            self.line_imu, = self.ax.plot([], [], label='IMU yaw [rad]')
            self.line_gt, = self.ax.plot([], [], label='GT yaw [rad]')
            self.ax.set_xlabel('Samples')
            self.ax.set_ylabel('Yaw [rad]')
            self.ax.grid(True)
            self.ax.legend(loc='upper right')
            self.sample_idx = 0
            self.x_data = deque(maxlen=2000)
            self.vo_yaw_data = deque(maxlen=2000)
            self.imu_yaw_data = deque(maxlen=2000)
            self.gt_yaw_data = deque(maxlen=2000)
            self.create_timer(0.05, self._update_plot_callback)

    def detect_features(self, img):
        """
        Detect ORB features in the given image and returns the points and the descriptor.
        """
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(img, None)
        kp = cv2.KeyPoint_convert(kp)
        return kp, des

    def track_features(self, pt1, pt2, des1, des2):
        """
        Match features between two sets of points using BFMatcher.
        """
        #TODO: find a better matcher, like Lowe's ratio? 
        # tried but the only better matcher (SIFT + FLANN) is too slow :(
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
        pts1 = np.float32([pt1[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([pt2[m.trainIdx] for m in matches]).reshape(-1, 1, 2)
        return pts1, pts2

    def image_callback_left(self, img):
        """
        Callback for left image, appending the image to the known images
        """
        left_img = self.cv_bridge.imgmsg_to_cv2(img, "bgr8")
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        self.left_images.append(left_gray)
    
    def ground_truth_callback(self, data): 
        self.ground_truth_yaw = data.yaw

    def image_callback_right(self, data):
        """
        Callback to receive and process the right camera image.
        Also computes odometry and publishes results.
        """ 
        right_img = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        self.right_images.append(right_gray)
        if len(self.left_images) > 1 and len(self.right_images) > 1:
            if self.starting:
                #TODO: check if there are other ways ->  these are radiants btw
                R2 = self.rotation_matrix_from_frames(self.right_images[0], self.right_images[-1])
                R1 = self.rotation_matrix_from_frames(self.left_images[0], self.left_images[-1])

                # FIX: proper None checks for rotation matrices
                if (R1 is not None) and (R2 is not None): 
                    R = self.averaging_rotation_matrices([R1, R2])
                    yaw_vo = np.arctan2(R[1, 0], R[0, 0])
                    
                    # Bias-correct logging with consistent angle wrapping
                    if isinstance(self.yaw, (int, float)) and isinstance(self.ground_truth_yaw, (int, float)):
                        yaw_imu = self.yaw
                        yaw_gt = self.ground_truth_yaw
                        
                        # Update bias estimate (IMU vs GT)
                        err_imu_gt = self.normalize_angle(yaw_imu - yaw_gt)
                        self._yaw_bias_samples.append(err_imu_gt)
                        if len(self._yaw_bias_samples) >= 50:
                            self._yaw_bias_estimate = float(np.median(np.array(self._yaw_bias_samples)))
                        
                        # Apply bias correction and wrap to [-pi, pi]
                        yaw_vo_c = self.normalize_angle(yaw_vo - self._yaw_bias_estimate)
                        yaw_imu_c = self.normalize_angle(yaw_imu - self._yaw_bias_estimate)
                        yaw_gt_c = self.normalize_angle(yaw_gt)
                        # Feed live plot (optional) with corrected and raw values
                        self._append_yaw_sample(yaw_vo, yaw_imu_c, yaw_gt_c)
                    else:
                        self._append_yaw_sample(yaw_vo, None, None)
                    
                    # Publish raw VO yaw and Pose for RViz/rqt_plot
                    self.vo_yaw_pub.publish(Float64(data=float(yaw_vo)))
                    self._publish_vo_pose(yaw_vo)
                    self.rotation_matrix = R 
                    self.starting = False      
            else:
                # IDEA: maybe the error is that i need to check only on some frames and not always the last 2 given the 
                # fact that maybe there aren't enough features? 
                R2 = self.rotation_matrix_from_frames(self.right_images[-2], self.right_images[-1])
                R1 = self.rotation_matrix_from_frames(self.left_images[-2], self.left_images[-1])
                
                # TODO: maybe we shouldn't average the 2 together!!!!
                if (R1 is not None) and (R2 is not None): 
                    R = self.averaging_rotation_matrices([R1, R2])
                    R = self.rotation_matrix @ R
                    self.rotation_matrix = R
                    yaw_vo = np.arctan2(R[1, 0], R[0, 0])
                    if isinstance(self.yaw, (int, float)) and isinstance(self.ground_truth_yaw, (int, float)):
                        yaw_imu = self.yaw
                        yaw_gt = self.ground_truth_yaw
                        err_imu_gt = self.normalize_angle(yaw_imu - yaw_gt)
                        self._yaw_bias_samples.append(err_imu_gt)
                        if len(self._yaw_bias_samples) >= 50:
                            self._yaw_bias_estimate = float(np.median(np.array(self._yaw_bias_samples)))
                        yaw_imu_c = self.normalize_angle(yaw_imu - self._yaw_bias_estimate)
                        yaw_gt_c = self.normalize_angle(yaw_gt)
                        self._append_yaw_sample(yaw_vo, yaw_imu_c, yaw_gt_c)
                    else:
                        self._append_yaw_sample(yaw_vo, None, None)
                    self.vo_yaw_pub.publish(Float64(data=float(yaw_vo)))
                    self._publish_vo_pose(yaw_vo)

    def score_homography_decomposition(self, R, T, P, K, pts1, pts2): 
        """
        Wrapper function for calculating easily the score assigned to an homography decomposition
        """
        # FIX: use normalized points with canonical projection matrices (consistency with undistortPoints)
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = np.hstack((R, T.reshape(3, 1)))

        pts1_h = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None)
        pts2_h = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None)

        points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
        points_3d = points_4d_hom[:3] / points_4d_hom[3]

        z1 = points_3d[2]  # Z in first camera
        z2 = (R[2] @ points_3d + T[2])  # Z in second camera frame
        valid = (z1 > 0) & (z2 > 0)

        return np.sum(valid)

    def state_estimation_callback(self, state): 
        """
        Updates the state estimation saved values for the odometry 
        """
        self.x = state.x
        self.y = state.y
        self.yaw = state.yaw
        self.v_x = state.v_x
        self.v_y = state.v_y
        self.yaw_r = state.yaw_r
        self.s = state.s
        self.v_s = state.v_s
        self.delta = state.delta
        self.d = state.d
        
    def rotation_matrix_from_frames(self, first_img, second_img): 
        """
        Function to obtain the Rotation matrix R and the translation matrix t from 2 frames
        """

        points_first, dsc1 = self.detect_features(first_img)
        points_second, dsc2 = self.detect_features(second_img)

        # FIX: if no descriptors are found, exit early to avoid matcher errors
        if dsc1 is None or dsc2 is None:
            return None

        # Simple slicing to prevent problems with mismatching lengths
        best_len = min(len(points_first), len(points_second))
        points_first = points_first[:best_len]
        points_second = points_second[:best_len]

        # FIX: align descriptor arrays length safely
        best_len = min(len(dsc1), len(dsc2))
        dsc1 = dsc1[:best_len]
        dsc2 = dsc2[:best_len]

        # check for enough points to detect (kinda arbitrary)
        #self.get_logger().info(f"{len(points_first)} - {len(points_second)} \n")
        with open("data.txt", "a") as f: 
            f.write(f"{len(points_first)}\n{len(points_second)}\n")

        # the average seems to be 484, with 480ish we take out the lowest 25% 
        if len(points_second) < 300: 
            return None

        features_first, features_second = self.track_features(points_first, points_second, dsc1, dsc2)
        homo, mask = cv2.findHomography(features_first, features_second, cv2.RANSAC, 5.0)

        # FIX: guard against failed homography estimation
        if homo is None or mask is None:
            return None

        # checkin the reliability of the homo mat
        # FIX: mask can be Nx1; use raveled sum for inlier count
        inlier_ratio = float(np.sum(mask)) / max(1, len(features_first))

        # we use homography
        if inlier_ratio < 0.5 or len(features_first) < 40: 
            num, R, t, p = cv2.decomposeHomographyMat(homo, self.camera_intrinsic)
            # FIX: score homography using matched points (features_first/second)
            scores = [ self.score_homography_decomposition(
                            R[i], t[i], p[i], self.camera_intrinsic,
                            features_first, features_second
                        ) for i in range(num)]
            best_idx = scores.index(max(scores))
            R, t, p = R[best_idx], t[best_idx], p[best_idx]

        # we use essential
        else:
            E, mask = cv2.findEssentialMat(features_first, features_second, self.camera_intrinsic, cv2.RANSAC)
            # FIX: guard against failed essential estimation
            if E is None or mask is None:
                return None
            # FIX: use matched points (features_first/second) in recoverPose
            _, R, t, _ = cv2.recoverPose(E, features_first, features_second, self.camera_intrinsic)
        # Basic validity checks on rotation matrix
        if R is None or not np.isfinite(R).all():
            return None
        detR = np.linalg.det(R)
        if not np.isfinite(detR) or abs(detR - 1.0) > 1e-2:
            return None
                
        return R
        
    def averaging_rotation_matrices(self, rot_matrices): 
        # Step 1: transforming from matrices to quaternions
        quats = np.array([Rotation.from_matrix(Ri).as_quat() for Ri in rot_matrices])  # [x, y, z, w]

        # Step 2: Ensure consistent quaternion signs (avoids canceling)
        for i in range(1, len(quats)):
            if np.dot(quats[0], quats[i]) < 0:
                quats[i] = -quats[i]

        # Step 3: Build the symmetric accumulator matrix
        A = np.zeros((4, 4))
        for q in quats:
            q = q.reshape(4, 1)
            A += q @ q.T

        # Step 4: Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        q_avg = eigenvectors[:, np.argmax(eigenvalues)]  # principal eigenvector

        # Step 5: Convert back to rotation matrix
        R_avg = Rotation.from_quat(q_avg).as_matrix()

        return R_avg

    def _append_yaw_sample(self, vo_yaw, imu_yaw=None, gt_yaw=None):
        if not hasattr(self, 'enable_plot') or not self.enable_plot:
            return
        # Allow None for imu/gt
        if not hasattr(self, 'sample_idx'):
            self.sample_idx = 0
        self.sample_idx += 1
        self.x_data.append(self.sample_idx)
        self.vo_yaw_data.append(float(vo_yaw))
        if imu_yaw is None:
            imu_val = float('nan')
        else:
            imu_val = float(imu_yaw)
        if gt_yaw is None:
            gt_val = float('nan')
        else:
            gt_val = float(gt_yaw)
        self.imu_yaw_data.append(imu_val)
        self.gt_yaw_data.append(gt_val)

    def _update_plot_callback(self):
        if not self.enable_plot or len(self.x_data) == 0:
            return
        self.line_vo.set_data(list(self.x_data), list(self.vo_yaw_data))
        self.line_imu.set_data(list(self.x_data), list(self.imu_yaw_data))
        self.line_gt.set_data(list(self.x_data), list(self.gt_yaw_data))
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _publish_vo_pose(self, yaw_rad):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        if isinstance(self.x, (int, float)):
            pose_msg.pose.position.x = float(self.x)
        if isinstance(self.y, (int, float)):
            pose_msg.pose.position.y = float(self.y)
        q = Rotation.from_euler('z', float(-yaw_rad)).as_quat()  # [x, y, z, w]
        pose_msg.pose.orientation.x = float(q[0])
        pose_msg.pose.orientation.y = float(q[1])
        pose_msg.pose.orientation.z = float(q[2])
        pose_msg.pose.orientation.w = float(q[3])
        self.vo_pose_pub.publish(pose_msg)

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Wrap any angle to the range [-pi, pi] in a numerically stable way."""
        return math.atan2(math.sin(angle), math.cos(angle))

    def kill_callback(self, _):
        # FIX: clean shutdown instead of abrupt exit
        try:
            self.destroy_node()
        finally:
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = VisualOdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
