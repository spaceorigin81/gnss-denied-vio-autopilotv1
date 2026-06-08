#!/usr/bin/env python3
"""
Tunnel Constraint Node — DP7 Environment-Aware Drift Reduction
===============================================================
Exploits the tunnel geometry as a physical sensor:
  1. ZUPT  — Zero Velocity Updates when drone hovers
  2. Wall  — Lateral velocity bounds (can't fly through walls)
  3. Floor — Altitude lower-bound constraint
  4. Ceiling — Altitude upper-bound constraint
  5. Heading alignment — tunnel axis aligns long-axis motion

Subscribes to VIO + IMU, publishes constraint-corrected odometry.
This is the final fusion stage before the autopilot.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistWithCovarianceStamped
from std_msgs.msg import Bool

import numpy as np
from collections import deque
from typing import Optional, Deque
import math
import time


class ZUPTDetector:
    """
    Detects zero-velocity phases from IMU acceleration variance.
    Triggers a velocity correction to counter VIO drift during hover.
    """

    def __init__(self, imu_rate: float = 200.0,
                 window_sec: float = 0.1,
                 threshold: float = 0.08):
        self.window_samples = int(imu_rate * window_sec)
        self.threshold      = threshold
        self._buf: Deque[float] = deque(maxlen=self.window_samples)
        self.is_static = False
        self._static_frames = 0
        self._CONFIRM_FRAMES = 3  # require N consecutive static detections

    def update(self, acc: np.ndarray) -> bool:
        """Return True if drone is detected as stationary."""
        # Dynamic acceleration magnitude (subtract gravity along z)
        acc_dyn = np.array([acc[0], acc[1], acc[2] - 9.81])
        self._buf.append(float(np.linalg.norm(acc_dyn)))

        if len(self._buf) < self.window_samples:
            return False

        variance = np.var(list(self._buf))
        if variance < self.threshold:
            self._static_frames += 1
        else:
            self._static_frames = 0
            self.is_static = False

        if self._static_frames >= self._CONFIRM_FRAMES:
            self.is_static = True

        return self.is_static


class TunnelGeometryModel:
    """
    Maintains a simple model of tunnel geometry for constraint generation.
    In a real deployment, these would come from a map.
    For simulation, we use reasonable defaults and update from VIO.
    """

    def __init__(self):
        # Tunnel dimensions (meters) — update from your Gazebo world
        self.tunnel_width  = 4.0    # total width
        self.tunnel_height = 3.0    # total height
        self.floor_z       = 0.0    # world z of floor
        self.ceiling_z     = 3.0    # world z of ceiling
        self.drone_radius  = 0.3    # safety margin

        # Operational bounds
        self.min_y = -self.tunnel_width/2 + self.drone_radius
        self.max_y =  self.tunnel_width/2 - self.drone_radius
        self.min_z =  self.floor_z + 0.3
        self.max_z =  self.ceiling_z - self.drone_radius

    def compute_position_violation(self, x: float, y: float,
                                    z: float) -> np.ndarray:
        """
        Returns correction vector [dx, dy, dz] to push position
        back within tunnel bounds.
        """
        corr = np.zeros(3)

        if y < self.min_y:
            corr[1] = self.min_y - y
        elif y > self.max_y:
            corr[1] = self.max_y - y

        if z < self.min_z:
            corr[2] = self.min_z - z
        elif z > self.max_z:
            corr[2] = self.max_z - z

        return corr

    def compute_velocity_constraint_cov(self, vx: float, vy: float,
                                         vz: float) -> np.ndarray:
        """
        Returns 3×3 velocity measurement covariance for the constraint.
        Tight lateral covariance = strong wall constraint.
        """
        # If lateral velocity is large, increase uncertainty
        # (could be intentional manoeuvre, not drift)
        lat_uncertain = max(0.005, min(0.1, abs(vy) * 0.1))

        return np.diag([
            0.05,             # forward — allow real motion
            lat_uncertain,    # lateral — tight near-zero constraint
            0.02              # vertical — moderate
        ])


class TunnelConstraintNode(Node):

    def __init__(self):
        super().__init__('tunnel_constraint_node')

        # ── Parameters ──────────────────────────────────────
        self.declare_parameter('tunnel_width',  4.0)
        self.declare_parameter('tunnel_height', 3.0)
        self.declare_parameter('zupt_threshold', 0.08)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5)
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=20)

        # ── Components ──────────────────────────────────────
        self.zupt     = ZUPTDetector()
        self.geometry = TunnelGeometryModel()

        self.geometry.tunnel_width  = self.get_parameter('tunnel_width').value
        self.geometry.tunnel_height = self.get_parameter('tunnel_height').value

        # ── State ───────────────────────────────────────────
        self._last_odom: Optional[Odometry] = None
        self._zupt_count = 0
        self._wall_count = 0
        self._total_corrections = np.zeros(3)

        # ── Subscribers ─────────────────────────────────────
        self.sub_odom = self.create_subscription(
            Odometry, '/ratslam/odometry',
            self.odom_callback, reliable_qos)

        self.sub_imu = self.create_subscription(
            Imu, '/simple_drone/imu/out',
            self.imu_callback, sensor_qos)

        # ── Publishers ──────────────────────────────────────
        self.pub_constrained = self.create_publisher(
            Odometry, '/tunnel_nav/odometry', reliable_qos)
        self.pub_zupt = self.create_publisher(
            Bool, '/tunnel_nav/zupt_active', reliable_qos)

        self.create_timer(2.0, self.report_stats)
        self.get_logger().info('Tunnel Constraint Node active')

    def imu_callback(self, msg: Imu):
        acc = np.array([msg.linear_acceleration.x,
                        msg.linear_acceleration.y,
                        msg.linear_acceleration.z])
        is_static = self.zupt.update(acc)
        self.pub_zupt.publish(Bool(data=is_static))

    def odom_callback(self, msg: Odometry):
        self._last_odom = msg

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z

        out = Odometry()
        out.header = msg.header
        out.header.frame_id = 'odom'
        out.child_frame_id  = 'base_link'
        out.pose  = msg.pose
        out.twist = msg.twist

        # ── ZUPT correction ─────────────────────────────────
        if self.zupt.is_static:
            # Drone is hovering: set velocity to zero with tight covariance
            out.twist.twist.linear.x = vx * 0.1
            out.twist.twist.linear.y = vy * 0.1
            out.twist.twist.linear.z = vz * 0.1
            for i in [0, 7, 14]:
                out.twist.covariance[i] = 0.001
            self._zupt_count += 1

        # ── Wall / geometry constraint ───────────────────────
        pos_corr = self.geometry.compute_position_violation(x, y, z)
        if np.any(np.abs(pos_corr) > 0.01):
            out.pose.pose.position.y = float(y + pos_corr[1])
            out.pose.pose.position.z = float(z + pos_corr[2])
            self._wall_count += 1
            self._total_corrections += np.abs(pos_corr)

        # ── Lateral velocity constraint ──────────────────────
        # In a tunnel, large lateral velocity = drift
        if abs(vy) > 0.5 and not self.zupt.is_static:
            # Soft constraint: attenuate but don't zero (might be real motion)
            out.twist.twist.linear.y = vy * 0.4
            cov = list(out.twist.covariance)
            cov[7] = max(cov[7], 0.02)
            out.twist.covariance = cov

        self.pub_constrained.publish(out)

    def report_stats(self):
        self.get_logger().info(
            f'[Constraints] ZUPT:{self._zupt_count} Wall:{self._wall_count} '
            f'TotalCorr:[{self._total_corrections[0]:.2f},'
            f'{self._total_corrections[1]:.2f},'
            f'{self._total_corrections[2]:.2f}]m')


def main(args=None):
    rclpy.init(args=args)
    node = TunnelConstraintNode()
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