#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, Image
import time
import sys

class SanityCheck(Node):
    def __init__(self):
        super().__init__('sanity_check_node')
        self.imu_c = 0
        self.cam_l_c = 0
        self.sub_i = self.create_subscription(Imu, '/simple_drone/imu/out', self.i_cb, 10)
        self.sub_l = self.create_subscription(Image, '/simple_drone/left_camera/image_raw', self.l_cb, 10)
        self.timer = self.create_timer(1.0, self.check)
        self.elapsed = 0

    def i_cb(self, m): self.imu_c += 1
    def l_cb(self, m): self.cam_l_c += 1

    def check(self):
        self.elapsed += 1
        if self.elapsed >= 10:
            self.get_logger().info('='*50)
            self.get_logger().info(f'✅ IMU Rate: {self.imu_c / 10.0} Hz')
            self.get_logger().info(f'✅ CAM Rate: {self.cam_l_c / 10.0} Hz')
            if self.imu_c > 0 and self.cam_l_c > 0:
                self.get_logger().info('✅ SYSTEM HEALTHY. READY FOR FLIGHT.')
            else:
                self.get_logger().error('❌ SENSORS OFFLINE.')
            self.get_logger().info('='*50)
            sys.exit(0)

def main():
    rclpy.init()
    try: rclpy.spin(SanityCheck())
    except SystemExit: pass
    finally: rclpy.shutdown()
if __name__ == '__main__': main()
