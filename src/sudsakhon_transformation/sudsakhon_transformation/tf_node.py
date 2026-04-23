#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
from builtin_interfaces.msg import Time
from rclpy.qos import QoSProfile

class SudsakhonBroadcaster(Node):
    def __init__(self):
        super().__init__('sudsakhon_broadcaster')
        self.get_logger().info("[Brobot][TF] Initializing 4-wheel Mechanum TF...")

        self.br = TransformBroadcaster(self)
        self.pub = self.create_publisher(Time, 'ros_time', QoSProfile(depth=10))
        self.timer = self.create_timer(0.1, self.broadcast_tf)
  
        # --- Robot Dimensions ---
        total_length_x = 0.456
        total_width_y = 0.377
        
        # Half-width and half-length from robot center (base_link)
        half_length = total_length_x / 2  # 0.228m
        half_width = total_width_y / 2    # 0.1885m
        
        # Wheel positions (m1:FL, m2:FR, m3:BL, m4:BR)
        self.wheel_fl_x = half_length    
        self.wheel_fl_y = half_width     
        self.wheel_fr_x = half_length    
        self.wheel_fr_y = -half_width    
        self.wheel_bl_x = -half_length   
        self.wheel_bl_y = half_width     
        self.wheel_br_x = -half_length   
        self.wheel_br_y = -half_width    
        self.wheel_z = 0.00  

        # --- Lidar Configuration: laser_base ---
        self.laser_base_x = 0.1125  
        self.laser_base_y = 0.0 
        self.laser_base_z = 0.114  

        # --- Lidar Configuration: laser_chair ---
        self.laser_chair_x = -0.082
        self.laser_chair_y = 0
        self.laser_chair_z = 0.7705

        # --- Camera Configuration: orbbec ---
        # Position provided: 30.45, 90.4, 0 (cm) -> 0.3045, 0.904, 0.0 (m)
        self.orbbec_x = 0.3125
        self.orbbec_y = 0
        self.orbbec_z = 0.90
        
        self.get_logger().info(f"[Brobot][TF] Configured with X: {total_length_x}m, Y: {total_width_y}m")
        self.get_logger().info(f"[Brobot][TF] Lidar 'laser_base' at: {self.laser_base_x}, {self.laser_base_y}, {self.laser_base_z}")
        self.get_logger().info(f"[Brobot][TF] Lidar 'laser_chair' at: {self.laser_chair_x}, {self.laser_chair_y}, {self.laser_chair_z}")
        self.get_logger().info(f"[Brobot][TF] Camera 'Camera_Orbbec' at: {self.orbbec_x}, {self.orbbec_y}, {self.orbbec_z}")

    def broadcast_tf(self):
        now = self.get_clock().now().to_msg()

        # 1. Base Lidar (laser_base)
        self.send_transform(now, 'base_link', 'laser_base', 
                          self.laser_base_x, self.laser_base_y, self.laser_base_z, 
                          0.0, 0.0, 0.0)

        # 2. Chair Lidar (laser_chair)
        self.send_transform(now, 'base_link', 'laser_chair',
                          self.laser_chair_x, self.laser_chair_y, self.laser_chair_z,
                          0.0, 0.0, 0.0)

        # 3. Camera Sensor (orbbec)
        self.send_transform(now, 'base_link', 'orbbec',
                          self.orbbec_x, self.orbbec_y, self.orbbec_z,
                          0.0, 0.0, 0.0)

        # 4. Mecanum Wheels
        self.send_transform(now, 'base_link', 'wheel_front_left', 
                          self.wheel_fl_x, self.wheel_fl_y, self.wheel_z, 0.0, 0.0, 0.0)
        self.send_transform(now, 'base_link', 'wheel_front_right', 
                          self.wheel_fr_x, self.wheel_fr_y, self.wheel_z, 0.0, 0.0, 0.0)
        self.send_transform(now, 'base_link', 'wheel_back_left', 
                          self.wheel_bl_x, self.wheel_bl_y, self.wheel_z, 0.0, 0.0, 0.0)
        self.send_transform(now, 'base_link', 'wheel_back_right', 
                          self.wheel_br_x, self.wheel_br_y, self.wheel_z, 0.0, 0.0, 0.0)

        self.pub.publish(now)

    def send_transform(self, time_msg, parent, child, x, y, z, roll, pitch, yaw):
        t = TransformStamped()
        t.header.stamp = time_msg
        t.header.frame_id = parent
        t.child_frame_id = child

        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = float(z)

        qx, qy, qz, qw = self.euler_to_quaternion(roll, pitch, yaw)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.br.sendTransform(t)

    def euler_to_quaternion(self, roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return (qx, qy, qz, qw)

def main():
    rclpy.init()
    node = SudsakhonBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()