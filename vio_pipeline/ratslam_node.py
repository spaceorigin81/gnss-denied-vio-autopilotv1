#!/usr/bin/env python3
"""
RatSLAM Fusion Node — DP7 Loop Closure & Place Recognition
===========================================================
Implements a bio-inspired RatSLAM-style topological map for:
  1. Loop closure detection in featureless tunnel corridors
  2. Long-term drift correction via experience replay
  3. Place cell network for spatial memory

Architecture based on Milford & Wyeth (2008) "RatSLAM: a hippocampal
model for simultaneous localization and mapping" — adapted for:
  • ROS 2 Humble
  • Stereo camera input (appearance fingerprinting)
  • Integration with MSCKF continuous pose estimate
  • Tunnel-specific: handles repeated/similar visual appearances

Pipeline:
  FourierVIO odometry + Camera image
    → Visual Template Matching (appearance fingerprint)
    → Pose Cell Network (attractor dynamics)
    → Experience Map (topological graph)
    → Loop-closure-corrected odometry
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import PoseArray, Pose

import numpy as np
import cv2
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Deque
import threading
import time
import math


# ─────────────────────────────────────────────────────────────
#  Visual Template (appearance fingerprint)
# ─────────────────────────────────────────────────────────────

@dataclass
class VisualTemplate:
    """
    Compressed appearance representation of a place.
    Uses patch-normalized grayscale descriptor (similar to RatSLAM's
    log-polar patch but adapted for tunnel corridor views).
    """
    id:        int
    descriptor: np.ndarray     # compressed appearance vector
    created_at: float          # timestamp
    x: float = 0.0            # estimated position when template was created
    y: float = 0.0
    z: float = 0.0
    decay:  float = 1.0        # activation strength (fades over time)


class VisualTemplateManager:
    """
    Manages visual templates and computes appearance similarity.

    For tunnel environments, standard ORB/SIFT descriptors fail because
    visual appearance repeats every ~3m (ceiling lights, wall texture).
    We use a multi-scale patch descriptor that captures the GRADIENT
    STRUCTURE rather than absolute appearance, making it invariant to
    the repeating pattern while sensitive to global position changes.
    """

    def __init__(self, descriptor_size: int = 64,
                 match_threshold: float = 0.85,
                 min_template_spacing: float = 0.5):
        self.desc_size = descriptor_size
        self.match_thr = match_threshold
        self.min_spacing = min_template_spacing

        self.templates: List[VisualTemplate] = []
        self._next_id = 0

        # Crop region for descriptor (use central strip for tunnel)
        self.crop_y = (0.3, 0.7)   # use middle 40% of image height
        self.crop_x = (0.1, 0.9)   # use middle 80% of image width

    def compute_descriptor(self, gray: np.ndarray) -> np.ndarray:
        """
        Compute a 64-D appearance descriptor.

        Steps:
          1. Crop to region of interest (tunnel walls/ceiling)
          2. Compute horizontal gradient magnitude (structural features)
          3. Divide into patches, compute mean gradient per patch
          4. L2-normalize for illumination invariance
          5. Apply Fourier low-pass to reduce texture aliasing
        """
        h, w = gray.shape
        y0, y1 = int(h * self.crop_y[0]), int(h * self.crop_y[1])
        x0, x1 = int(w * self.crop_x[0]), int(w * self.crop_x[1])
        roi = gray[y0:y1, x0:x1].astype(float)

        # Compute gradient magnitude
        gx = np.gradient(roi, axis=1)
        gy = np.gradient(roi, axis=0)
        grad_mag = np.sqrt(gx**2 + gy**2)

        # Frequency domain: keep only low spatial frequencies
        # This reduces sensitivity to repeating high-freq tunnel texture
        from numpy.fft import fft2, ifft2
        F = fft2(grad_mag)
        freq_h = grad_mag.shape[0]
        freq_w = grad_mag.shape[1]
        cutoff_h = max(1, freq_h // 8)
        cutoff_w = max(1, freq_w // 8)
        mask = np.zeros_like(F, dtype=float)
        mask[:cutoff_h, :cutoff_w] = 1.0
        mask[-cutoff_h:, :cutoff_w] = 1.0
        mask[:cutoff_h, -cutoff_w:] = 1.0
        mask[-cutoff_h:, -cutoff_w:] = 1.0
        grad_lf = np.real(ifft2(F * mask))

        # Resize to fixed grid and flatten
        n = int(np.sqrt(self.desc_size))
        grid = cv2.resize(grad_lf, (n, n), interpolation=cv2.INTER_AREA)
        desc = grid.ravel().astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(desc)
        if norm > 1e-6:
            desc /= norm

        return desc

    def match(self, desc: np.ndarray) -> Tuple[int, float]:
        """
        Find the best matching template.
        Returns (template_id, similarity_score) or (-1, 0.0) if no match.
        """
        if len(self.templates) == 0:
            return -1, 0.0

        # Vectorised dot product (cosine similarity after L2 norm)
        descs = np.array([t.descriptor for t in self.templates])
        sims  = descs @ desc
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self.match_thr:
            return self.templates[best_idx].id, best_sim
        return -1, best_sim

    def add_template(self, desc: np.ndarray, x: float, y: float,
                     z: float, timestamp: float) -> int:
        # Check spacing — don't add templates too close together
        if self.templates:
            last = self.templates[-1]
            dist = math.sqrt((x-last.x)**2 + (y-last.y)**2 + (z-last.z)**2)
            if dist < self.min_spacing:
                return self.templates[-1].id

        tmpl = VisualTemplate(
            id=self._next_id, descriptor=desc.copy(),
            created_at=timestamp, x=x, y=y, z=z)
        self.templates.append(tmpl)
        self._next_id += 1
        return tmpl.id

    def decay_all(self, dt: float = 0.1, tau: float = 30.0):
        """Exponentially decay template activations."""
        for t in self.templates:
            t.decay *= math.exp(-dt / tau)


# ─────────────────────────────────────────────────────────────
#  Pose Cell Network (continuous attractor dynamics)
# ─────────────────────────────────────────────────────────────

class PoseCellNetwork:
    """
    3D continuous attractor network modelling the hippocampal place cells.

    The network maintains a "bump" of activity that represents the current
    estimated position.  The bump is:
      • Shifted by odometry (path integration)
      • Anchored by visual template matches (place cell activation)

    This provides drift correction that is qualitatively different from
    standard EKF: instead of covariance-weighted fusion, it uses the
    competitive dynamics of the attractor network to resist drift.
    """

    def __init__(self, size: int = 20):
        self.S   = size           # network dimensions
        self.act = np.zeros((size, size, size))  # 3D activity

        # Place the initial bump at the centre
        cx = cy = cz = size // 2
        self.act[cx, cy, cz] = 1.0
        self._apply_diffusion(sigma=1.5)
        self._normalise()

        # Shift accumulators (sub-cell precision)
        self._delta = np.zeros(3)

        # Precompute gaussian kernel for bump injection
        self._gauss_kernel = self._make_gaussian(3, 1.0)

    def path_integrate(self, dx: float, dy: float, dz: float,
                       scale: float = 2.0):
        """
        Shift the activity bump based on odometry.

        scale: how many cells per meter of travel
        """
        self._delta += np.array([dx, dy, dz]) * scale
        shifts = self._delta.astype(int)
        self._delta -= shifts

        for axis, shift in enumerate(shifts):
            if shift != 0:
                self.act = np.roll(self.act, shift, axis=axis)

        # Diffuse slightly to maintain smooth bump
        self._apply_diffusion(sigma=0.3)
        self._normalise()

    def inject_template(self, template_x: float, template_y: float,
                        template_z: float, strength: float = 0.3,
                        scale: float = 2.0):
        """
        Inject activity at the position corresponding to a recognised template.
        This is the loop-closure mechanism: seeing a known place pulls the
        bump toward the template's remembered position.
        """
        # Convert world coords to network indices
        cx = int(np.clip(template_x * scale + self.S//2, 0, self.S-1))
        cy = int(np.clip(template_y * scale + self.S//2, 0, self.S-1))
        cz = int(np.clip(template_z * scale + self.S//2, 0, self.S-1))

        # Add Gaussian bump at template location
        h, w, d = self.act.shape
        for dz_ in range(-1, 2):
            for dy_ in range(-1, 2):
                for dx_ in range(-1, 2):
                    xi = (cx + dx_) % self.S
                    yi = (cy + dy_) % self.S
                    zi = (cz + dz_) % self.S
                    dist2 = dx_**2 + dy_**2 + dz_**2
                    self.act[xi, yi, zi] += strength * math.exp(-dist2 / 2.0)

        self._normalise()

    def get_best_estimate(self, scale: float = 2.0) -> Tuple[float, float, float]:
        """Return the world position of the activity bump's centre of mass."""
        # Find peak
        idx = np.unravel_index(np.argmax(self.act), self.act.shape)

        # Weighted centroid around peak for sub-cell precision
        window = 3
        x_sum = y_sum = z_sum = w_sum = 0.0
        for dx in range(-window, window+1):
            for dy in range(-window, window+1):
                for dz in range(-window, window+1):
                    xi = (idx[0]+dx) % self.S
                    yi = (idx[1]+dy) % self.S
                    zi = (idx[2]+dz) % self.S
                    a = self.act[xi, yi, zi]
                    x_sum += xi * a
                    y_sum += yi * a
                    z_sum += zi * a
                    w_sum += a

        if w_sum < 1e-10:
            return 0.0, 0.0, 0.0

        px = (x_sum / w_sum - self.S//2) / scale
        py = (y_sum / w_sum - self.S//2) / scale
        pz = (z_sum / w_sum - self.S//2) / scale
        return px, py, pz

    def _apply_diffusion(self, sigma: float = 0.5):
        """Gaussian smoothing to maintain continuous attractor."""
        from scipy.ndimage import gaussian_filter
        self.act = gaussian_filter(self.act, sigma=sigma)

    def _normalise(self):
        s = np.sum(self.act)
        if s > 1e-10:
            self.act /= s

    def _make_gaussian(self, size: int, sigma: float) -> np.ndarray:
        r = np.arange(size) - size//2
        g = np.exp(-r**2 / (2*sigma**2))
        k3d = g[:, None, None] * g[None, :, None] * g[None, None, :]
        return k3d / k3d.sum()


# ─────────────────────────────────────────────────────────────
#  Experience Map (topological graph)
# ─────────────────────────────────────────────────────────────

@dataclass
class Experience:
    """A node in the topological experience map."""
    id:          int
    x_m:         float         # metric position
    y_m:         float
    z_m:         float
    template_id: int           # visual template when created
    links:       List[int] = field(default_factory=list)  # connected experience IDs
    timestamp:   float = 0.0


class ExperienceMap:
    """
    Topological map of experiences (places).
    Nodes are experiences; edges are transitions.
    Loop closure: when a known template is seen, relaxation moves
    experiences to be geometrically consistent with the loop.
    """

    def __init__(self):
        self.experiences: List[Experience] = []
        self.current_id: int = -1
        self._next_id   = 0
        self._lock = threading.Lock()
        self.loop_closures: List[Tuple[int, int, float]] = []  # (exp_a, exp_b, distance)

    def add_experience(self, x: float, y: float, z: float,
                       template_id: int, timestamp: float) -> int:
        with self._lock:
            exp = Experience(id=self._next_id, x_m=x, y_m=y, z_m=z,
                             template_id=template_id, timestamp=timestamp)
            if self.current_id >= 0:
                # Link to previous experience
                exp.links.append(self.current_id)
                self.experiences[self.current_id].links.append(exp.id)
            self.experiences.append(exp)
            self.current_id = self._next_id
            self._next_id += 1
            return exp.id

    def register_loop_closure(self, exp_id_a: int, exp_id_b: int,
                               distance: float):
        """Record a loop closure between two experiences."""
        with self._lock:
            self.loop_closures.append((exp_id_a, exp_id_b, distance))
            self.get_logger_msg = (f'Loop closure: exp {exp_id_a} ↔ exp {exp_id_b} '
                                   f'dist={distance:.2f}m')

    def relax(self, iterations: int = 5, alpha: float = 0.1):
        """
        Spring-mass relaxation to make the map geometrically consistent.
        Called after each loop closure event.
        """
        with self._lock:
            for _ in range(iterations):
                for lc in self.loop_closures:
                    a_id, b_id, d = lc
                    if a_id >= len(self.experiences) or b_id >= len(self.experiences):
                        continue
                    ea = self.experiences[a_id]
                    eb = self.experiences[b_id]
                    dx = eb.x_m - ea.x_m
                    dy = eb.y_m - ea.y_m
                    dz = eb.z_m - ea.z_m
                    actual_d = math.sqrt(dx**2 + dy**2 + dz**2)
                    if actual_d < 1e-6:
                        continue
                    error = actual_d - d
                    correction = alpha * error / actual_d
                    ea.x_m += correction * dx
                    ea.y_m += correction * dy
                    ea.z_m += correction * dz
                    eb.x_m -= correction * dx
                    eb.y_m -= correction * dy
                    eb.z_m -= correction * dz

    def get_current_corrected_pose(self) -> Optional[Tuple[float, float, float]]:
        with self._lock:
            if self.current_id < 0 or self.current_id >= len(self.experiences):
                return None
            e = self.experiences[self.current_id]
            return e.x_m, e.y_m, e.z_m

    def to_pose_array(self) -> List[Tuple[float, float, float]]:
        with self._lock:
            return [(e.x_m, e.y_m, e.z_m) for e in self.experiences]


# ─────────────────────────────────────────────────────────────
#  ROS 2 Node
# ─────────────────────────────────────────────────────────────

class RatSLAMNode(Node):

    def __init__(self):
        super().__init__('ratslam_node')

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5)

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=20)

        # ── Components ──────────────────────────────────────
        self.template_mgr = VisualTemplateManager(
            descriptor_size=64,
            match_threshold=0.88,
            min_template_spacing=0.5)

        self.pose_cells   = PoseCellNetwork(size=24)
        self.exp_map      = ExperienceMap()

        # ── State ───────────────────────────────────────────
        self._prev_x = self._prev_y = self._prev_z = 0.0
        self._prev_ts = 0.0
        self._loop_count = 0
        self._exp_count  = 0
        self._current_template_id = -1
        self._initialized = False

        # ── Subscribers ─────────────────────────────────────
        self.sub_odom = self.create_subscription(
            Odometry, '/fourier_vio/odometry',
            self.odom_callback, reliable_qos)

        self.sub_cam = self.create_subscription(
            Image, '/simple_drone/left_camera/image_raw',
            self.image_callback, sensor_qos)

        # ── Publishers ──────────────────────────────────────
        self.pub_odom   = self.create_publisher(
            Odometry, '/ratslam/odometry', reliable_qos)
        self.pub_map    = self.create_publisher(
            PoseArray, '/ratslam/experience_map', reliable_qos)
        self.pub_status = self.create_publisher(
            String, '/ratslam/status', reliable_qos)

        # ── Timers ──────────────────────────────────────────
        self.create_timer(0.5, self.publish_map)
        self.create_timer(5.0, self.relax_map)
        self.create_timer(1.0, self.status_callback)

        self.get_logger().info('RatSLAM Node active — building experience map...')

    # ── Callbacks ────────────────────────────────────────────

    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if not self._initialized:
            self._prev_x  = x
            self._prev_y  = y
            self._prev_z  = z
            self._prev_ts = ts
            self._initialized = True
            self.exp_map.add_experience(x, y, z, -1, ts)
            return

        dx = x - self._prev_x
        dy = y - self._prev_y
        dz = z - self._prev_z

        # Path integrate pose cells
        self.pose_cells.path_integrate(dx, dy, dz)

        # Get pose cell estimate
        pc_x, pc_y, pc_z = self.pose_cells.get_best_estimate()

        # Blend metric VIO with pose cell estimate
        # Pose cells provide loop-closure correction
        blend = 0.15   # how much to trust pose cells vs raw VIO
        corrected_x = x * (1 - blend) + pc_x * blend
        corrected_y = y * (1 - blend) + pc_y * blend
        corrected_z = z * (1 - blend) + pc_z * blend

        # Publish corrected odometry
        out = Odometry()
        out.header = msg.header
        out.header.frame_id = 'ratslam_odom'
        out.child_frame_id  = 'base_link'
        out.pose.pose = msg.pose.pose
        out.pose.pose.position.x = corrected_x
        out.pose.pose.position.y = corrected_y
        out.pose.pose.position.z = corrected_z
        out.twist = msg.twist
        self.pub_odom.publish(out)

        # Create new experience if moved enough
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        if dist > 0.3:
            self.exp_map.add_experience(
                corrected_x, corrected_y, corrected_z,
                self._current_template_id, ts)
            self._exp_count += 1

        self._prev_x  = x
        self._prev_y  = y
        self._prev_z  = z
        self._prev_ts = ts

    def image_callback(self, msg: Image):
        # Convert to grayscale
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding in ('rgb8', 'bgr8'):
            arr = arr.reshape(msg.height, msg.width, 3)
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            gray = arr.reshape(msg.height, msg.width)

        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Compute appearance descriptor
        desc = self.template_mgr.compute_descriptor(gray)

        # Try to match against existing templates
        matched_id, sim = self.template_mgr.match(desc)

        if matched_id >= 0:
            # Known place — inject into pose cell network
            tmpl = next((t for t in self.template_mgr.templates
                         if t.id == matched_id), None)
            if tmpl is not None:
                self.pose_cells.inject_template(
                    tmpl.x, tmpl.y, tmpl.z, strength=0.3)

                # Check if this is a loop closure (revisiting old place)
                if (self._current_template_id >= 0 and
                        matched_id != self._current_template_id and
                        abs(matched_id - self._current_template_id) > 5):
                    self._handle_loop_closure(matched_id, sim)

            self._current_template_id = matched_id
        else:
            # New place — add template
            if self._initialized:
                x, y, z = self._prev_x, self._prev_y, self._prev_z
                new_id = self.template_mgr.add_template(desc, x, y, z, ts)
                self._current_template_id = new_id

        self.template_mgr.decay_all(dt=1.0/60.0)

    def _handle_loop_closure(self, template_id: int, similarity: float):
        """Process a loop closure event."""
        self._loop_count += 1
        current_exp = self.exp_map.get_current_corrected_pose()
        tmpl = next((t for t in self.template_mgr.templates
                     if t.id == template_id), None)

        if current_exp is None or tmpl is None:
            return

        # Distance between current position and template's remembered position
        dist = math.sqrt(
            (current_exp[0] - tmpl.x)**2 +
            (current_exp[1] - tmpl.y)**2 +
            (current_exp[2] - tmpl.z)**2)

        # Register loop closure in experience map
        if self.exp_map.current_id >= 0:
            # Find experience associated with this template
            for exp in self.exp_map.experiences:
                if exp.template_id == template_id:
                    self.exp_map.register_loop_closure(
                        self.exp_map.current_id, exp.id, dist)
                    break

        self.get_logger().info(
            f'🔄 Loop closure #{self._loop_count}: '
            f'template={template_id} sim={similarity:.3f} dist={dist:.2f}m')

    def publish_map(self):
        poses = self.exp_map.to_pose_array()
        if not poses:
            return
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = 'odom'
        for (x, y, z) in poses:
            p = Pose()
            p.position.x = x
            p.position.y = y
            p.position.z = z
            p.orientation.w = 1.0
            pa.poses.append(p)
        self.pub_map.publish(pa)

    def relax_map(self):
        self.exp_map.relax(iterations=10, alpha=0.05)

    def status_callback(self):
        msg = String()
        msg.data = (f'Templates:{len(self.template_mgr.templates)} '
                    f'Experiences:{self._exp_count} '
                    f'LoopClosures:{self._loop_count}')
        self.pub_status.publish(msg)
        self.get_logger().info(f'[RatSLAM] {msg.data}')


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = RatSLAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()