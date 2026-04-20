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
  
        # --- Updated Dimensions based on User Input (22.8, 18.85, 0) ---
        total_length_x = 0.456
        total_width_y = 0.377
        
        # Half-width and half-length from robot center (base_link)
        half_length = total_length_x / 2  # 0.114m
        half_width = total_width_y / 2    # 0.09425m
        
        # Wheel positions (X = forward/back, Y = left/right, Z = up/down)
        # m1 = Front Left
        self.wheel_fl_x = half_length    
        self.wheel_fl_y = half_width     
        
        # m2 = Front Right
        self.wheel_fr_x = half_length    
        self.wheel_fr_y = -half_width    
        
        # m3 = Back Left
        self.wheel_bl_x = -half_length   
        self.wheel_bl_y = half_width     
        
        # m4 = Back Right
        self.wheel_br_x = -half_length   
        self.wheel_br_y = -half_width    
        
        # Wheel height (Z-axis) 
        self.wheel_z = 0.00  

        # --- New Lidar Configuration: laser_base ---
        # Position provided: 22.8, 11.25, 11.4 (cm)
        self.laser_base_x = 0.228  
        self.laser_base_y = 0.1125 
        self.laser_base_z = 0.114  
        
        self.get_logger().info(f"[Brobot][TF] Configured with X: {total_length_x}m, Y: {total_width_y}m")
        self.get_logger().info(f"[Brobot][TF] New Lidar 'laser_base' at: {self.laser_base_x}, {self.laser_base_y}, {self.laser_base_z}")

    def broadcast_tf(self):
        now = self.get_clock().now().to_msg()

        # 1. New Base Lidar (laser_base)
        self.send_transform(now, 'base_link', 'laser_base', 
                          self.laser_base_x, self.laser_base_y, self.laser_base_z, 
                          0.0, 0.0, 0.0)

        # 2. Mecanum Wheels
        # Front Left
        self.send_transform(now, 'base_link', 'wheel_front_left', 
                          self.wheel_fl_x, self.wheel_fl_y, self.wheel_z, 
                          0.0, 0.0, 0.0)
        
        # Front Right
        self.send_transform(now, 'base_link', 'wheel_front_right', 
                          self.wheel_fr_x, self.wheel_fr_y, self.wheel_z, 
                          0.0, 0.0, 0.0)
        
        # Back Left
        self.send_transform(now, 'base_link', 'wheel_back_left', 
                          self.wheel_bl_x, self.wheel_bl_y, self.wheel_z, 
                          0.0, 0.0, 0.0)
        
        # Back Right
        self.send_transform(now, 'base_link', 'wheel_back_right', 
                          self.wheel_br_x, self.wheel_br_y, self.wheel_z, 
                          0.0, 0.0, 0.0)

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