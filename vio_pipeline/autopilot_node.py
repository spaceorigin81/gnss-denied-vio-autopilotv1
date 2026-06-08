#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

class PID:
    def __init__(self, kp, ki, kd, out_max):
        self.kp, self.ki, self.kd, self.out_max = kp, ki, kd, out_max
        self.i, self.prev_e = 0.0, 0.0
    def update(self, e, dt):
        if dt <= 0: return 0.0
        self.i = max(min(self.i + e*dt, 1.0), -1.0)
        d = (e - self.prev_e) / dt; self.prev_e = e
        return max(min((self.kp*e) + (self.ki*self.i) + (self.kd*d), self.out_max), -self.out_max)

class PrecisionAutopilot(Node):
    def __init__(self):
        super().__init__('autopilot')
        self.sub_vio = self.create_subscription(Odometry, '/ov_msckf/odomimu', self.vio_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        
        self.pos = [0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.active = False
        self.pt = 0.0
        self.mission_complete = False
        
        self.takeoff_z = None
        self.py = PID(1.5, 0.05, 0.3, 0.8)
        self.pz = PID(2.0, 0.1, 0.5, 1.0) 
        self.pyaw = PID(1.2, 0.0, 0.2, 0.5)
        
        self.create_timer(0.05, self.loop)
        self.get_logger().info("🔓 Precision Autopilot Online. Dynamic Braking Enabled.")

    def vio_cb(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        if math.isfinite(p.x):
            self.pos = [p.x, p.y, p.z]
            self.yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))
            self.active = True

    def loop(self):
        if not self.active: return
        now = self.get_clock().now().nanoseconds / 1e9
        if self.pt == 0.0: self.pt = now; return
        dt = now - self.pt; self.pt = now
        
        dist_to_target = 200.0 - self.pos[0]

        # Exact Stop Condition
        if dist_to_target <= 0.2:
            if not self.mission_complete:
                self.get_logger().warn(f"🛑 TARGET SECURED AT {self.pos[0]:.2f}m. KILLING MOTORS.")
            self.mission_complete = True
            self.pub_cmd.publish(Twist()) # Hold Hover
            return

        if self.takeoff_z is None and self.pos[2] > 0.5:
            self.takeoff_z = self.pos[2]
            
        target_z = self.takeoff_z if self.takeoff_z is not None else 1.0

        # Dynamic Braking (Slows down perfectly to a stop)
        cx = max(0.05, min(0.8, dist_to_target * 0.4)) 
        
        cy = self.py.update(0.0 - self.pos[1], dt)
        cz = self.pz.update(target_z - self.pos[2], dt)
        cyaw = self.pyaw.update(0.0 - self.yaw, dt)

        cos_y, sin_y = math.cos(self.yaw), math.sin(self.yaw)
        t = Twist()
        t.linear.x = cx * cos_y + cy * sin_y
        t.linear.y = -cx * sin_y + cy * cos_y
        t.linear.z = cz
        t.angular.z = cyaw
        self.pub_cmd.publish(t)

def main(): rclpy.init(); rclpy.spin(PrecisionAutopilot()); rclpy.shutdown()
if __name__ == '__main__': main()