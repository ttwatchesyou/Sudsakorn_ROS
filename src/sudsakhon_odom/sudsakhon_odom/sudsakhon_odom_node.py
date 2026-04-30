#!/usr/bin/env python3
"""
Robot Bridge Node — ROS2 ↔ STM32 Serial  (merged v3)

Serial format รับจาก STM32:
  POS:curX:curY:curYaw:tick_x:tick_y

Commands ส่งไป STM32:
  G<x>:<y>:<yaw>        → goal position
  V<vx>:<vy>            → manual velocity
  Z                     → reset odometry
  S<speed>              → set max speed
  P5:<rpm>:<rad>:<kp>   → yaw profile
  P6:<kp>:<ki>:<kd>     → position PID
  T<accel>:<decel>:<mv> → trapezoidal profile
"""

import rclpy
from rclpy.node import Node
import tf2_ros
import serial
import threading
import time
import math

from nav_msgs.msg import Odometry
from geometry_msgs.msg import (Quaternion, TransformStamped,
                                Pose2D, Twist,
                                PoseWithCovarianceStamped)
from std_msgs.msg import Float32, Empty, Float32MultiArray
from tf2_ros import TransformBroadcaster


class SudsakhonOdomNode(Node):
    def __init__(self):
        super().__init__('sudsakhon_odom_node')

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('serial_port', '/dev/Controller_Base')
        self.declare_parameter('baud_rate',   115200)
        self.declare_parameter('max_speed',   0.6)

        # ── Serial ───────────────────────────────────────────────────────────
        port      = self.get_parameter('serial_port').value
        baud_rate = self.get_parameter('baud_rate').value
        self.ser  = None
        try:
            self.ser = serial.Serial(port, baud_rate, timeout=0.1)
            time.sleep(1)
            self.get_logger().info(f"✅ Serial connected: {port}")
            self.send_to_robot('S', f"{self.get_parameter('max_speed').value}")
        except Exception as e:
            self.get_logger().error(f"❌ Serial error: {e}")

        # ── Publishers & TF ──────────────────────────────────────────────────
        self.odom_pub    = self.create_publisher(Odometry, '/odom', 10)
        self.tf_br       = TransformBroadcaster(self)

        # ── Subscriptions ─────────────────────────────────────────────────────
        # --- motion commands ---
        self.create_subscription(Twist,  '/cmd_vel',  self.vel_callback,  10)
        self.create_subscription(Pose2D, '/cmd_goal', self.goal_callback, 10)

        # --- tuner topics ---
        self.create_subscription(Empty,             '/cmd_reset_odom',   self.reset_callback,   10)
        self.create_subscription(Float32MultiArray, '/cmd_pos_pid',      self.pos_pid_callback, 10)
        self.create_subscription(Float32MultiArray, '/cmd_yaw_pid',      self.yaw_pid_callback, 10)
        self.create_subscription(Float32,           '/cmd_max_speed',    self.speed_callback,   10)
        self.create_subscription(Float32MultiArray, '/cmd_trap_profile', self.trap_callback,    10)

        # --- nav2 init pose (2D Pose Estimate จาก RViz) ---
        self.create_subscription(PoseWithCovarianceStamped,
                                 '/initialpose', self.init_pose_cb, 10)

        # ── Odometry state ────────────────────────────────────────────────────
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0

        # ── Serial read thread ────────────────────────────────────────────────
        self.running = True
        threading.Thread(target=self.serial_loop, daemon=True).start()

        # ── Parameter live-update ─────────────────────────────────────────────
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.get_logger().info("SudsakhonOdomNode ready ✅")

    # ── Parameter callback ────────────────────────────────────────────────────
    def parameter_callback(self, params):
        for p in params:
            if p.name == 'max_speed':
                self.send_to_robot('S', f"{p.value}")
        return rclpy.node.SetParametersResult(successful=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TOPIC CALLBACKS
    # ══════════════════════════════════════════════════════════════════════════

    def vel_callback(self, msg):
        """
        /cmd_vel → V<vx>:<vy>
        ใช้ linear.x / linear.y เป็น velocity ใน robot frame
        angular.z ถูกละเว้น — STM32 ใช้ yaw lock ของตัวเอง
        """
        vx = round(msg.linear.x, 3)
        vy = round(msg.linear.y, 3)
        self.send_to_robot('V', f"{vx}:{vy}")

    def goal_callback(self, msg):
        """
        /cmd_goal (Pose2D) → G<x>:<y>:<theta>
        """
        self.send_to_robot('G', f"{msg.x}:{msg.y}:{msg.theta}")

    def reset_callback(self, _):
        """
        /cmd_reset_odom → Z
        รีเซ็ต odom ทั้ง STM32 และ state ใน Python
        """
        self.x = self.y = 0.0
        self.send_to_robot('Z', "")
        self.get_logger().info("Odometry reset")

    def init_pose_cb(self, msg):
        """
        /initialpose (2D Pose Estimate จาก RViz / Nav2)
        ตั้งค่าตำแหน่งเริ่มต้นโดยไม่ reset STM32
        """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self.get_logger().info(
            f"📍 Init pose → X:{self.x:.3f}  Y:{self.y:.3f}  θ:{math.degrees(self.yaw):.1f}°")

    def pos_pid_callback(self, msg):
        """
        /cmd_pos_pid [kp, ki, kd] → P6:<kp>:<ki>:<kd>
        """
        if len(msg.data) >= 3:
            kp, ki, kd = msg.data[0], msg.data[1], msg.data[2]
            self.send_to_robot('P', f"6:{kp}:{ki}:{kd}")

    def yaw_pid_callback(self, msg):
        """
        /cmd_yaw_pid [max_rpm, brake_rad, fine_kp] → P5:<max_rpm>:<brake_rad>:<fine_kp>
        """
        if len(msg.data) >= 3:
            rpm, brake, fkp = msg.data[0], msg.data[1], msg.data[2]
            self.send_to_robot('P', f"5:{rpm}:{brake}:{fkp}")

    def speed_callback(self, msg):
        """
        /cmd_max_speed (Float32) → S<speed>
        """
        spd = round(float(msg.data), 3)
        self.send_to_robot('S', f"{spd}")

    def trap_callback(self, msg):
        """
        /cmd_trap_profile [accel, decel, min_v] → T<accel>:<decel>:<min_v>
        """
        if len(msg.data) >= 3:
            accel, decel, min_v = msg.data[0], msg.data[1], msg.data[2]
            self.send_to_robot('T', f"{accel}:{decel}:{min_v}")
            self.get_logger().info(
                f"Trap profile → accel:{accel:.1f}  decel:{decel:.1f}  min_v:{min_v:.2f}")

    # ══════════════════════════════════════════════════════════════════════════
    # SERIAL SEND
    # ══════════════════════════════════════════════════════════════════════════
    def send_to_robot(self, cmd_type, data):
        if self.ser and self.ser.is_open:
            full_cmd = f"{cmd_type}{data}\n"
            try:
                self.ser.write(full_cmd.encode())
                if cmd_type != 'V':   # ไม่ log velocity spam
                    self.get_logger().info(f"TX: {full_cmd.strip()}")
            except Exception as e:
                self.get_logger().error(f"Serial write error: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # SERIAL READ LOOP
    # STM32 ส่ง: POS:curX:curY:curYaw:tick_x:tick_y
    # ══════════════════════════════════════════════════════════════════════════
    def serial_loop(self):
        while self.running and rclpy.ok():
            if self.ser and self.ser.in_waiting > 0:
                try:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()

                    if line.startswith("POS:"):
                        parts = line.split(':')
                        if len(parts) >= 4:
                            x   = float(parts[1])   # curX   (m) — คำนวณแล้วจาก STM32
                            y   = float(parts[2])   # curY   (m)
                            yaw = float(parts[3])   # curYaw (rad)
                            # tick_x = int(parts[4])  # สำรองไว้ถ้าต้องการ
                            # tick_y = int(parts[5])
                            self._update_pose(x, y, yaw)

                except Exception as e:
                    self.get_logger().debug(f"Serial parse error: {e}")
            time.sleep(0.005)

    def _update_pose(self, x, y, yaw):
        """อัปเดต pose จาก STM32 แล้ว publish odom + TF"""
        # ถ้า init_pose_cb ตั้งค่าไว้ จะ override ค่าจาก STM32
        # ใช้ค่าจาก STM32 ตรงๆ (STM32 คำนวณ odom เองแล้ว)
        self.x   = x
        self.y   = y
        self.yaw = yaw
        self.publish_odom(x, y, yaw)

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLISH ODOM + TF
    # ══════════════════════════════════════════════════════════════════════════
    def publish_odom(self, x, y, yaw):
        now = self.get_clock().now().to_msg()
        q   = self._euler_to_quat(yaw)

        # TF: odom → base_link
        t = TransformStamped()
        t.header.stamp        = now
        t.header.frame_id     = 'odom'
        t.child_frame_id      = 'base_link'
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0
        t.transform.rotation      = q
        self.tf_br.sendTransform(t)

        # Odometry message
        odom = Odometry()
        odom.header.stamp        = now
        odom.header.frame_id     = 'odom'
        odom.child_frame_id      = 'base_link'
        odom.pose.pose.position.x  = x
        odom.pose.pose.position.y  = y
        odom.pose.pose.position.z  = 0.0
        odom.pose.pose.orientation = q
        self.odom_pub.publish(odom)

    # ══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════════
    def _euler_to_quat(self, yaw):
        q = Quaternion()
        q.w = math.cos(yaw * 0.5)
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw * 0.5)
        return q

    def destroy_node(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        super().destroy_node()


# ══════════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = SudsakhonOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()