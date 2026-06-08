#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import threading
from collections import deque

class UnstoppablePlotter(Node):
    def __init__(self):
        super().__init__('unstoppable_plotter')
        self.lock = threading.Lock()
        
        # Subscriptions
        self.sub_vio = self.create_subscription(Odometry, '/ov_msckf/odomimu', self.vio_cb, 10)
        self.sub_gt = self.create_subscription(Odometry, '/simple_drone/odom', self.gt_cb, 10)
        
        # Unrestricted Data Buffers
        self.gt_data = deque(maxlen=2000)
        self.vio_data = deque(maxlen=2000)
        
        self.t_start = None
        self.plot_times = deque(maxlen=2000)
        self.plot_errors = deque(maxlen=2000)
        
        self.get_logger().info("✅ Live Tracking Engaged. All filters removed. Data will flow.")

    def gt_cb(self, msg):
        # Save time and raw coordinates
        t = msg.header.stamp.sec + (msg.header.stamp.nanosec / 1e9)
        p = msg.pose.pose.position
        with self.lock:
            self.gt_data.append((t, p.x, p.y, p.z))

    def vio_cb(self, msg):
        t = msg.header.stamp.sec + (msg.header.stamp.nanosec / 1e9)
        p = msg.pose.pose.position
        
        if self.t_start is None: 
            self.t_start = t
            
        with self.lock:
            self.vio_data.append((t, p.x, p.y, p.z))
            
            # If we have Ground Truth, calculate error to the closest point
            if len(self.gt_data) > 0:
                # Proper Logic: Find the GT point with the smallest time difference
                closest_gt = min(self.gt_data, key=lambda gt: abs(gt[0] - t))
                
                # Absolute Spatial Error (3D Euclidean Distance)
                true_err = math.sqrt((p.x - closest_gt[1])**2 + (p.y - closest_gt[2])**2 + (p.z - closest_gt[3])**2)
                
                self.plot_times.append(t - self.t_start)
                self.plot_errors.append(true_err)

def main():
    rclpy.init()
    node = UnstoppablePlotter()
    
    # Run ROS node in background thread so GUI can own the main thread
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 8))
    fig.canvas.manager.set_window_title('Mission Control: Live VIO Tracking')
    
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_err = fig.add_subplot(122)

    def update(frame):
        # Extract data safely
        with node.lock:
            gt_list = list(node.gt_data)
            vio_list = list(node.vio_data)
            t_list = list(node.plot_times)
            err_list = list(node.plot_errors)

        ax_3d.clear()
        
        # Draw Ground Truth
        if gt_list:
            gx = [d[1] for d in gt_list]; gy = [d[2] for d in gt_list]; gz = [d[3] for d in gt_list]
            ax_3d.plot(gx, gy, gz, color='#00FF00', label='Ground Truth', linewidth=2)
            
        # Draw VIO
        if vio_list:
            vx = [d[1] for d in vio_list]; vy = [d[2] for d in vio_list]; vz = [d[3] for d in vio_list]
            ax_3d.plot(vx, vy, vz, color='#00FFFF', linestyle='--', label='VIO Estimate', linewidth=2)

        ax_3d.set_title("Live 3D Trajectory Tracking", pad=20, fontsize=14)
        ax_3d.set_xlabel('X (m)'); ax_3d.set_ylabel('Y (m)'); ax_3d.set_zlabel('Altitude (m)')
        ax_3d.legend(loc='upper left')
        ax_3d.set_zlim(0, 3)
        if vio_list: ax_3d.set_xlim(0, max(210, vio_list[-1][1] + 10))

        ax_err.clear()
        
        # Draw Error Profile
        if err_list and t_list:
            ax_err.plot(t_list, err_list, color='#FF3333', linewidth=2)
            ax_err.fill_between(t_list, 0, err_list, color='#FF3333', alpha=0.3)
            
            cur_err = err_list[-1]
            current_x = vio_list[-1][1] if vio_list else 0.1
            drift_pct = (cur_err / max(0.1, current_x)) * 100 if current_x > 1.0 else 0.0
            
            status = "PASS (<1.5%)" if drift_pct < 1.5 else "WARNING"
            color = "#00FF00" if drift_pct < 1.5 else "#FF0000"
            
            ax_err.text(0.05, 0.85, f"Absolute Spatial Error: {cur_err:.3f}m\nDrift Margin: {drift_pct:.2f}%\nStatus: {status}", 
                     transform=ax_err.transAxes, fontsize=14, color=color, weight='bold')

            ax_err.set_title("Time-Aligned Absolute Trajectory Error", fontsize=14)
            ax_err.set_xlabel("Elapsed Time (s)", fontsize=12)
            ax_err.set_ylabel("Error (m)", fontsize=12)
            ax_err.set_ylim(0, max(max(err_list, default=1.0) + 0.5, 2.0))
            ax_err.grid(True, alpha=0.2)

    # 400ms update interval prevents X11/WSL GUI crashes while keeping it "live"
    ani = animation.FuncAnimation(fig, update, interval=400)
    plt.tight_layout()
    plt.show()
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()