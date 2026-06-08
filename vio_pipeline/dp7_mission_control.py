#!/usr/bin/env python3
"""
DP7 Mission Control Dashboard (Freeze-Frame Edition)
=========================================================
ISRO-style real-time telemetry for the DP7 VIO autonomous drone project.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
import threading
import time
import math
import numpy as np
from collections import deque

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

MAX_LEN          = 2000      
UPDATE_INTERVAL  = 150       
DRIFT_THRESHOLD  = 0.015     

C_BG, C_BG2, C_PANEL, C_BORDER = '#0a0a0f', '#0f0f1a', '#12121f', '#1e2040'
C_GT, C_VIO, C_ERROR, C_WARN, C_PASS = '#00ff88', '#00e5ff', '#ff4444', '#ffaa00', '#00ff88'
C_TEXT, C_TEXT2, C_GRID, C_ACCENT = '#e0e8f0', '#8090a8', '#1a2030', '#3060ff'

class TelemetryStore:
    def __init__(self, max_len: int = MAX_LEN):
        self._lock = threading.Lock()
        self.gt_history = deque(maxlen=3000)
        self.unified_data = deque(maxlen=max_len)
        self.mission_start_t  = None
        self.total_distance   = 0.0
        self._prev_vio        = None
        self.mission_completed = False  # THE FREEZE FLAG

    def push_gt(self, x, y, z, stamp):
        with self._lock:
            self.gt_history.append((stamp, x, y, z))

    def push_vio(self, x, y, z, stamp):
        with self._lock:
            # FREEZE TELEMETRY UPON REACHING TARGET
            if self.mission_completed: return
            if x >= 199.8: 
                self.mission_completed = True
                return

            if not self.gt_history: return
            if self.mission_start_t is None: self.mission_start_t = stamp

            closest_gt = min(self.gt_history, key=lambda g: abs(g[0] - stamp))
            gx, gy, gz = closest_gt[1], closest_gt[2], closest_gt[3]

            err = math.sqrt((x-gx)**2 + (y-gy)**2 + (z-gz)**2)

            velx, vely, velz = 0.0, 0.0, 0.0
            if self._prev_vio is not None:
                pt, px, py, pz = self._prev_vio
                dt = stamp - pt
                if dt > 0: velx, vely, velz = (x-px)/dt, (y-py)/dt, (z-pz)/dt
                dist_step = math.sqrt((x-px)**2 + (y-py)**2 + (z-pz)**2)
                if dist_step < 2.0: self.total_distance += dist_step
                
            self._prev_vio = (stamp, x, y, z)
            self.unified_data.append((stamp, x, y, z, gx, gy, gz, velx, vely, velz, err))

    def snapshot(self):
        with self._lock:
            if not self.unified_data: return None
            dl = list(self.unified_data)
            return {
                'vio_t': [d[0] for d in dl], 'vio_x': [d[1] for d in dl], 'vio_y': [d[2] for d in dl], 'vio_z': [d[3] for d in dl],
                'gt_x': [d[4] for d in dl], 'gt_y': [d[5] for d in dl], 'gt_z': [d[6] for d in dl],
                'vio_vx': [d[7] for d in dl], 'vio_vy': [d[8] for d in dl], 'vio_vz': [d[9] for d in dl],
                'drift_error': [d[10] for d in dl],
                'mission_t': self.mission_start_t, 'total_dist': self.total_distance,
                'completed': self.mission_completed
            }

class DP7TelemetryNode(Node):
    def __init__(self, store: TelemetryStore):
        super().__init__('dp7_mission_control')
        self.store = store
        self.sub_vio = self.create_subscription(Odometry, '/ov_msckf/odomimu', self._vio_cb, 10)
        self.sub_gt = self.create_subscription(Odometry, '/simple_drone/odom', self._gt_cb, 10)

    def _vio_cb(self, msg: Odometry):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.store.push_vio(msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, ts)

    def _gt_cb(self, msg: Odometry):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.store.push_gt(msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, ts)

class MissionControlDashboard:
    def __init__(self, store: TelemetryStore):
        self.store = store
        self._build_figure()
        self._init_artists()

    def _build_figure(self):
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(20, 11), facecolor=C_BG, num='DP7 Mission Control')
        self.fig.patch.set_facecolor(C_BG)

        self.fig.text(0.5, 0.97, 'DP7 AUTONOMOUS NAVIGATOR  ·  MISSION CONTROL', ha='center', va='top', fontsize=14, fontweight='bold', color=C_TEXT, fontfamily='monospace')
        outer = gridspec.GridSpec(3, 1, figure=self.fig, height_ratios=[0.04, 0.70, 0.20], hspace=0.12, left=0.04, right=0.97, top=0.93, bottom=0.04)

        self.ax_status = self.fig.add_subplot(outer[0])
        self.ax_status.set_facecolor(C_PANEL); self.ax_status.set_xticks([]); self.ax_status.set_yticks([])
        for spine in self.ax_status.spines.values(): spine.set_edgecolor(C_BORDER)

        main_gs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], wspace=0.30, width_ratios=[2.2, 1.4, 1.0, 1.0])

        self.ax3d = self.fig.add_subplot(main_gs[0], projection='3d')
        self.ax3d.set_facecolor(C_BG2)
        self.ax3d.xaxis.pane.fill = False; self.ax3d.yaxis.pane.fill = False; self.ax3d.zaxis.pane.fill = False
        self.ax3d.grid(True, color=C_GRID, linewidth=0.4, alpha=0.6)
        self.ax3d.set_xlabel('X (m)', color=C_TEXT2, fontsize=8); self.ax3d.set_ylabel('Y (m)', color=C_TEXT2, fontsize=8); self.ax3d.set_zlabel('Z (m)', color=C_TEXT2, fontsize=8)
        self.ax3d.tick_params(colors=C_TEXT2, labelsize=7)
        self.ax3d.set_title('TRAJECTORY  /  3D', color=C_TEXT, fontsize=9, fontfamily='monospace', pad=8)

        self.ax_drift = self.fig.add_subplot(main_gs[1])
        self._style_2d_ax(self.ax_drift, 'DRIFT ANALYSIS  /  ATE (m)')

        self.ax_vel = self.fig.add_subplot(main_gs[2])
        self._style_2d_ax(self.ax_vel, 'VELOCITY  /  m·s⁻¹')

        self.ax_alt = self.fig.add_subplot(main_gs[3])
        self._style_2d_ax(self.ax_alt, 'ALTITUDE  /  Z (m)')

        bot_gs = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=outer[2], wspace=0.06)
        self.ax_metrics = []
        metric_titles = ['MISSION TIME', 'DISTANCE', 'CURRENT ATE', 'MAX ATE', 'MEAN ATE', 'STATUS']
        for i, title in enumerate(metric_titles):
            ax = self.fig.add_subplot(bot_gs[i])
            ax.set_facecolor(C_PANEL); ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_edgecolor(C_BORDER)
            ax.text(0.5, 0.82, title, ha='center', va='top', fontsize=6.5, color=C_TEXT2, fontfamily='monospace', transform=ax.transAxes)
            self.ax_metrics.append(ax)

        self._t0_wall = time.monotonic()

    def _style_2d_ax(self, ax, title):
        ax.set_facecolor(C_BG2); ax.tick_params(colors=C_TEXT2, labelsize=7)
        ax.set_title(title, color=C_TEXT, fontsize=8, fontfamily='monospace', pad=6)
        ax.grid(True, color=C_GRID, linewidth=0.4, alpha=0.7)
        for s in ax.spines.values(): s.set_color(C_BORDER)

    def _init_artists(self):
        self.line3d_gt, = self.ax3d.plot([], [], [], color=C_GT, linewidth=1.4, alpha=0.9)
        self.line3d_vio, = self.ax3d.plot([], [], [], color=C_VIO, linewidth=1.0, alpha=0.8, linestyle='--')
        self.dot3d_gt, = self.ax3d.plot([], [], [], 'o', color=C_GT, markersize=5, zorder=5)
        self.dot3d_vio, = self.ax3d.plot([], [], [], 's', color=C_VIO, markersize=4, zorder=5)
        self.ax3d.legend(handles=[Line2D([0],[0], color=C_GT, lw=1.5, label='Ground Truth'), Line2D([0],[0], color=C_VIO, lw=1.0, ls='--', label='VIO Estimate')], loc='upper left', fontsize=7, facecolor=C_PANEL, edgecolor=C_BORDER, labelcolor=C_TEXT, framealpha=0.85)

        self.line_drift, = self.ax_drift.plot([], [], color=C_ERROR, linewidth=1.2, alpha=0.9)
        self.fill_drift = None
        self.ax_drift.axhline(0, color=C_WARN, linewidth=0.6, linestyle=':', alpha=0.6)
        self.txt_drift_status = self.ax_drift.text(0.97, 0.95, '', ha='right', va='top', fontsize=10, fontweight='bold', fontfamily='monospace', transform=self.ax_drift.transAxes, bbox=dict(boxstyle='round,pad=0.3', facecolor=C_PANEL, edgecolor=C_BORDER, alpha=0.9))
        self.txt_drift_dist = self.ax_drift.text(0.03, 0.95, 'dist: 0.00m', ha='left', va='top', fontsize=7, color=C_TEXT2, fontfamily='monospace', transform=self.ax_drift.transAxes)

        self.line_vx, = self.ax_vel.plot([], [], color='#ff6666', linewidth=1.0, label='Vx')
        self.line_vy, = self.ax_vel.plot([], [], color='#66ff66', linewidth=1.0, label='Vy')
        self.line_vz, = self.ax_vel.plot([], [], color='#6699ff', linewidth=1.0, label='Vz')
        self.ax_vel.legend(fontsize=7, facecolor=C_PANEL, edgecolor=C_BORDER, labelcolor=C_TEXT, loc='upper right', framealpha=0.8)

        self.line_alt_vio, = self.ax_alt.plot([], [], color=C_VIO, linewidth=1.0, alpha=0.8, linestyle='--', label='VIO')
        self.line_alt_gt, = self.ax_alt.plot([], [], color=C_GT, linewidth=1.2, alpha=0.9, label='GT')
        self.ax_alt.legend(fontsize=7, facecolor=C_PANEL, edgecolor=C_BORDER, labelcolor=C_TEXT, loc='upper right', framealpha=0.8)

        self.metric_vals = []
        for ax in self.ax_metrics:
            self.metric_vals.append(ax.text(0.5, 0.42, '---', ha='center', va='center', fontsize=15, fontweight='bold', color=C_TEXT, fontfamily='monospace', transform=ax.transAxes))

        self.txt_status_bar = self.ax_status.text(0.5, 0.5, 'WAITING FOR TELEMETRY', ha='center', va='center', fontsize=8, color=C_TEXT2, fontfamily='monospace', transform=self.ax_status.transAxes)
        self.scan_line = self.ax_status.axvline(0, color=C_ACCENT, linewidth=1.0, alpha=0.6)

    def update(self, frame):
        d = self.store.snapshot()
        t_norm = (time.monotonic() % 4.0) / 4.0
        self.scan_line.set_xdata([t_norm])

        if d is None: return

        # STATUS BAR FREEZE LOGIC
        if d['completed']:
            self.txt_status_bar.set_text(f'MISSION ACCOMPLISHED   ·   TELEMETRY FROZEN   ·   DIST {d["total_dist"]:.1f}m')
            self.txt_status_bar.set_color(C_PASS)
        else:
            wall_t = time.monotonic() - self._t0_wall
            min_t, sec_t = int(wall_t) // 60, int(wall_t) % 60
            self.txt_status_bar.set_text(f'VIO ● LIVE   ·   GT ● LIVE   ·   T+{min_t:02d}:{sec_t:02d}   ·   DIST {d["total_dist"]:.1f}m')
            self.txt_status_bar.set_color(C_GT)

        vx, vy, vz, gx, gy, gz = d['vio_x'], d['vio_y'], d['vio_z'], d['gt_x'], d['gt_y'], d['gt_z']
        vt, err = d['vio_t'], d['drift_error']

        self.line3d_vio.set_data_3d(vx, vy, vz)
        self.dot3d_vio.set_data_3d([vx[-1]], [vy[-1]], [vz[-1]])
        self.line3d_gt.set_data_3d(gx, gy, gz)
        self.dot3d_gt.set_data_3d([gx[-1]], [gy[-1]], [gz[-1]])

        all_x, all_y, all_z = vx + gx, vy + gy, vz + gz
        pad = 0.5
        self.ax3d.set_xlim(min(all_x)-pad, max(all_x)+pad)
        self.ax3d.set_ylim(min(all_y)-pad, max(all_y)+pad)
        self.ax3d.set_zlim(max(0, min(all_z)-0.3), max(all_z)+0.5)
        self.ax3d.view_init(elev=22, azim=(30 + frame * 0.4) % 360)

        t0 = vt[0]
        ts_rel = [t - t0 for t in vt]

        self.line_drift.set_data(ts_rel, err)
        self.ax_drift.relim(); self.ax_drift.autoscale_view()
        if self.fill_drift is not None: self.fill_drift.remove()
        self.fill_drift = self.ax_drift.fill_between(ts_rel, err, alpha=0.18, color=C_ERROR)

        dist = d['total_dist']
        for line in self.ax_drift.lines[1:]:
            if hasattr(line, '_dp7_thresh'): line.remove()
        thr_line = self.ax_drift.axhline(dist * DRIFT_THRESHOLD, color=C_WARN, linewidth=0.8, linestyle='--', alpha=0.7)
        thr_line._dp7_thresh = True

        curr_drift = err[-1]
        drift_pct = (curr_drift / max(dist, 0.01)) * 100
        is_pass = drift_pct <= 1.5
        self.txt_drift_status.set_text('PASS ✓' if is_pass else 'FAIL ✗')
        self.txt_drift_status.set_color(C_PASS if is_pass else C_ERROR)
        self.txt_drift_status.get_bbox_patch().set_edgecolor(C_PASS if is_pass else C_ERROR)
        self.txt_drift_dist.set_text(f'err: {curr_drift:.3f}m  |  {drift_pct:.2f}%')

        self.line_vx.set_data(ts_rel, d['vio_vx'])
        self.line_vy.set_data(ts_rel, d['vio_vy'])
        self.line_vz.set_data(ts_rel, d['vio_vz'])
        self.ax_vel.relim(); self.ax_vel.autoscale_view()

        self.line_alt_vio.set_data(ts_rel, vz)
        self.line_alt_gt.set_data(ts_rel, gz)
        self.ax_alt.relim(); self.ax_alt.autoscale_view()

        elapsed = vt[-1] - d['mission_t']
        self.metric_vals[0].set_text(f'{int(elapsed)//60:02d}:{int(elapsed)%60:02d}')
        self.metric_vals[1].set_text(f'{dist:.1f}m')
        self.metric_vals[2].set_text(f'{curr_drift:.3f}m')
        self.metric_vals[2].set_color(C_PASS if is_pass else C_ERROR)
        
        max_ate = max(err)
        self.metric_vals[3].set_text(f'{max_ate:.3f}m')
        self.metric_vals[3].set_color(C_PASS if ((max_ate / max(dist, 0.01))*100 <= 1.5) else C_WARN)
        
        self.metric_vals[4].set_text(f'{float(np.mean(err)):.3f}m')

        self.metric_vals[5].set_text('PASS ✓' if is_pass else 'FAIL ✗')
        self.metric_vals[5].set_color(C_PASS if is_pass else C_ERROR)

    def start(self):
        self.anim = animation.FuncAnimation(self.fig, self.update, interval=UPDATE_INTERVAL, blit=False)
        plt.tight_layout(); plt.show()

def main():
    store = TelemetryStore(max_len=MAX_LEN)
    rclpy.init(); node = DP7TelemetryNode(store)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    try: MissionControlDashboard(store).start()
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()