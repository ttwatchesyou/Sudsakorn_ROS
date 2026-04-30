#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import tf2_ros
import serial
import threading
import time
import math
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Empty, Float32MultiArray
from geometry_msgs.msg import Quaternion, TransformStamped, Pose2D, Twist

class RobotBridgeNode(Node):
    def __init__(self):
        super().__init__('robot_bridge')

        # ─────────────────────────────────────────────
        # 1. PARAMETERS
        # ─────────────────────────────────────────────
        self.declare_parameter('serial_port', '/dev/Controller_Base')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('max_speed', 0.6)

        # --- Encoder / Wheel geometry ---
        # ขนาดล้อของ drag encoder (ล้อที่ลากพื้น)
        self.declare_parameter('encoder_wheel_diameter', 0.060)   # เมตร (48.26 mm = 1.9 inch แบบ standard)
        # จำนวน pulse ต่อ 1 รอบ (รวม quadrature X4 และ gear ratio แล้ว)
        self.declare_parameter('encoder_pulses_per_rev', 10.0)    # pulse/rev

        # ตำแหน่งของ drag encoder สัมพัทธ์จากจุดกึ่งกลางหุ่นยนต์
        # encoder_x ติดตั้งให้วัดการเคลื่อนที่แนว X (ด้านหน้า-หลัง)
        #   → offset ทางแกน Y (ด้านข้าง) จากจุดศูนย์กลาง
        # encoder_y ติดตั้งให้วัดการเคลื่อนที่แนว Y (ด้านข้าง)
        #   → offset ทางแกน X (หน้า-หลัง) จากจุดศูนย์กลาง
        #
        # หน่วย: เมตร  (บวก = ไปทาง +X หรือ +Y)
        self.declare_parameter('encoder_x_offset_y', 0.0)   # encoder วัดแนว X, ห่างจากศูนย์ในแกน Y
        self.declare_parameter('encoder_y_offset_x', 0.0)   # encoder วัดแนว Y, ห่างจากศูนย์ในแกน X

        # ─────────────────────────────────────────────
        # 2. โหลดค่า parameter มาใช้
        # ─────────────────────────────────────────────
        wheel_diam   = self.get_parameter('encoder_wheel_diameter').value
        ppr          = self.get_parameter('encoder_pulses_per_rev').value
        self.meters_per_pulse = (math.pi * wheel_diam) / ppr   # เมตร/pulse

        self.enc_x_offset_y = self.get_parameter('encoder_x_offset_y').value
        self.enc_y_offset_x = self.get_parameter('encoder_y_offset_x').value

        self.get_logger().info(
            f"Encoder cal: {self.meters_per_pulse*1000:.4f} mm/pulse | "
            f"enc_x offset_y={self.enc_x_offset_y*100:.1f} cm | "
            f"enc_y offset_x={self.enc_y_offset_x*100:.1f} cm"
        )

        # ─────────────────────────────────────────────
        # 3. SERIAL SETUP
        # ─────────────────────────────────────────────
        serial_port = self.get_parameter('serial_port').value
        baud_rate   = self.get_parameter('baud_rate').value
        self.ser = None
        try:
            self.ser = serial.Serial(serial_port, baud_rate, timeout=0.1)
            self.get_logger().info(f"Connected to {serial_port}")
            init_speed = self.get_parameter('max_speed').value
            self.send_to_robot('S', f"{init_speed}")
        except Exception as e:
            self.get_logger().error(f"Serial error: {e}")

        # ─────────────────────────────────────────────
        # 4. ROS PUBLISHERS & TF
        # ─────────────────────────────────────────────
        self.odom_pub      = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ─────────────────────────────────────────────
        # 5. ROS SUBSCRIPTIONS
        # ─────────────────────────────────────────────
        self.create_subscription(Twist,           '/cmd_vel',        self.vel_callback,     10)
        self.create_subscription(Pose2D,          '/cmd_goal',       self.goal_callback,    10)
        self.create_subscription(Empty,           '/cmd_reset_odom', lambda _: self.reset_odom(), 10)
        self.create_subscription(Float32MultiArray,'/cmd_pos_pid',   self.pos_pid_callback, 10)
        self.create_subscription(Float32MultiArray,'/cmd_yaw_pid',   self.yaw_pid_callback, 10)
        self.create_subscription(Float32,         '/cmd_max_speed',  self.speed_limit_callback, 10)

        # ─────────────────────────────────────────────
        # 6. ODOMETRY STATE
        #    รับ pulse สะสมจาก STM32 (absolute counter)
        # ─────────────────────────────────────────────
        self.pose_x   = 0.0   # เมตร (โลก)
        self.pose_y   = 0.0   # เมตร (โลก)
        self.pose_yaw = 0.0   # radian (จาก IMU)

        self.prev_enc_x_pulses = None   # pulse สะสม enc X (แนว forward)
        self.prev_enc_y_pulses = None   # pulse สะสม enc Y (แนว lateral)

        # ─────────────────────────────────────────────
        # 7. SERIAL THREAD
        # ─────────────────────────────────────────────
        self.serial_thread = threading.Thread(target=self.serial_loop, daemon=True)
        self.serial_thread.start()

        # ─────────────────────────────────────────────
        # 8. PARAMETER LIVE-UPDATE CALLBACK
        # ─────────────────────────────────────────────
        self.add_on_set_parameters_callback(self.parameter_callback)

    # ──────────────────────────────────────────────────
    # PARAMETER CALLBACKS
    # ──────────────────────────────────────────────────
    def parameter_callback(self, params):
        recalc = False
        for p in params:
            if p.name == 'max_speed':
                self.send_to_robot('S', f"{p.value}")
            elif p.name in ('encoder_wheel_diameter', 'encoder_pulses_per_rev'):
                recalc = True
            elif p.name == 'encoder_x_offset_y':
                self.enc_x_offset_y = p.value
            elif p.name == 'encoder_y_offset_x':
                self.enc_y_offset_x = p.value

        if recalc:
            d   = self.get_parameter('encoder_wheel_diameter').value
            ppr = self.get_parameter('encoder_pulses_per_rev').value
            self.meters_per_pulse = (math.pi * d) / ppr
            self.get_logger().info(f"Recalc meters_per_pulse = {self.meters_per_pulse*1000:.4f} mm/pulse")

        return rclpy.node.SetParametersResult(successful=True)

    # ──────────────────────────────────────────────────
    # TOPIC CALLBACKS
    # ──────────────────────────────────────────────────
    def speed_limit_callback(self, msg):
        val = round(float(msg.data), 2)
        self.send_to_robot('S', f"{val}")

    def vel_callback(self, msg):
        self.send_to_robot('V', f"{round(msg.linear.x,3)}:{round(msg.linear.y,3)}")

    def goal_callback(self, msg):
        self.send_to_robot('G', f"{msg.x}:{msg.y}:{msg.theta}")

    def pos_pid_callback(self, msg):
        if len(msg.data) >= 3:
            self.send_to_robot('P', f"6:{msg.data[0]}:{msg.data[1]}:{msg.data[2]}")

    def yaw_pid_callback(self, msg):
        if len(msg.data) >= 3:
            self.send_to_robot('P', f"5:{msg.data[0]}:{msg.data[1]}:{msg.data[2]}")

    def reset_odom(self):
        self.pose_x = self.pose_y = self.pose_yaw = 0.0
        self.prev_enc_x_pulses = None
        self.prev_enc_y_pulses = None
        self.send_to_robot('Z', "")
        self.get_logger().info("Odometry reset")

    # ──────────────────────────────────────────────────
    # SERIAL SEND
    # ──────────────────────────────────────────────────
    def send_to_robot(self, cmd_type, data):
        if self.ser and self.ser.is_open:
            full_cmd = f"{cmd_type}{data}\n"
            try:
                self.ser.write(full_cmd.encode())
                if cmd_type != 'V':
                    self.get_logger().info(f"Serial Send: {full_cmd.strip()}")
            except Exception as e:
                self.get_logger().error(f"Serial write error: {e}")

    # ──────────────────────────────────────────────────
    # SERIAL READ LOOP
    # ──────────────────────────────────────────────────
    def serial_loop(self):
        while rclpy.ok():
            if self.ser and self.ser.in_waiting > 0:
                try:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith("POS:"):
                        parts = line.split(':')
                        if len(parts) >= 4:
                            x   = float(parts[1])  # curX จาก STM32 (เมตร)
                            y   = float(parts[2])  # curY จาก STM32 (เมตร)
                            yaw = float(parts[3])  # curYaw (radian)
                            self.publish_odom_and_tf(x, y, yaw)
                except Exception as e:
                    self.get_logger().debug(f"Serial parse error: {e}")
            time.sleep(0.005)

    # ──────────────────────────────────────────────────
    # ODOMETRY UPDATE — Dead Reckoning ด้วย Drag Encoder + IMU
    # ──────────────────────────────────────────────────
    def update_odometry(self, enc_x_pulse, enc_y_pulse, yaw):
        """
        Drag Encoder Odometry พร้อมแก้ค่า rotational component:

        เมื่อหุ่นยนต์หมุนตัว encoder ที่ติดตั้งไม่อยู่ที่จุดศูนย์กลาง
        จะอ่านค่าการเคลื่อนที่เสมือนหุ่นยนต์เคลื่อนที่ทั้งที่จริงๆ แค่หมุน
        จึงต้องหักลบ rotational component ออกก่อน

        สูตร:
          Δθ = yaw_current - yaw_previous
          raw_dx_local = ΔPulse_X * meters_per_pulse
          raw_dy_local = ΔPulse_Y * meters_per_pulse

          แก้ rotation crosstalk:
          true_dx_local = raw_dx_local - (enc_x_offset_y * Δθ)
          true_dy_local = raw_dy_local - (enc_y_offset_x * Δθ)  ← เครื่องหมายขึ้นกับการติดตั้ง

          แปลงเป็นโลก (World frame) ด้วย yaw ปัจจุบัน:
          Δx_world = true_dx_local * cos(yaw) - true_dy_local * sin(yaw)
          Δy_world = true_dx_local * sin(yaw) + true_dy_local * cos(yaw)
        """

        # ── เริ่มต้น: บันทึก pulse แรกเป็น baseline ──
        if self.prev_enc_x_pulses is None:
            self.prev_enc_x_pulses = enc_x_pulse
            self.prev_enc_y_pulses = enc_y_pulse
            self.pose_yaw = yaw
            return

        # ── คำนวณ delta ──
        delta_x_pulse = enc_x_pulse - self.prev_enc_x_pulses
        delta_y_pulse = enc_y_pulse - self.prev_enc_y_pulses
        delta_yaw     = yaw - self.pose_yaw

        # ── normalize delta_yaw ให้อยู่ใน [-π, π] ──
        delta_yaw = math.atan2(math.sin(delta_yaw), math.cos(delta_yaw))

        # ── แปลง pulse → เมตร (local frame) ──
        raw_dx_local = delta_x_pulse * self.meters_per_pulse
        raw_dy_local = delta_y_pulse * self.meters_per_pulse

        # ── แก้ rotational crosstalk ──
        #   enc_x วัดแนว X แต่ติดตั้งห่างจากศูนย์ในแกน Y
        #   ดังนั้นเมื่อหมุน Δθ จะเพิ่ม arc = offset_y * Δθ ในแนว X
        #   (เครื่องหมาย: ขึ้นอยู่กับว่า encoder อยู่ด้าน +Y หรือ -Y)
        true_dx_local = raw_dx_local - (self.enc_x_offset_y * delta_yaw)
        #   enc_y วัดแนว Y แต่ติดตั้งห่างจากศูนย์ในแกน X
        true_dy_local = raw_dy_local + (self.enc_y_offset_x * delta_yaw)

        # ── rotate เข้า World frame ──
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        dx_world = true_dx_local * cos_yaw - true_dy_local * sin_yaw
        dy_world = true_dx_local * sin_yaw + true_dy_local * cos_yaw

        # ── update pose ──
        self.pose_x   += dx_world
        self.pose_y   += dy_world
        self.pose_yaw  = yaw

        # ── บันทึก pulse ปัจจุบันเป็น baseline รอบหน้า ──
        self.prev_enc_x_pulses = enc_x_pulse
        self.prev_enc_y_pulses = enc_y_pulse

        # ── publish ──
        self.publish_odom_and_tf(self.pose_x, self.pose_y, self.pose_yaw)

    # ──────────────────────────────────────────────────
    # PUBLISH ODOM + TF
    # ──────────────────────────────────────────────────
    def publish_odom_and_tf(self, x, y, yaw):
        now = self.get_clock().now().to_msg()
        q   = self.euler_to_quaternion(0, 0, yaw)

        # TF
        t = TransformStamped()
        t.header.stamp        = now
        t.header.frame_id     = 'odom'
        t.child_frame_id      = 'base_link'
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.rotation  = q
        self.tf_broadcaster.sendTransform(t)

        # Odometry message
        odom = Odometry()
        odom.header.stamp        = now
        odom.header.frame_id     = 'odom'
        odom.child_frame_id      = 'base_link'
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation = q
        self.odom_pub.publish(odom)

    # ──────────────────────────────────────────────────
    # EULER → QUATERNION
    # ──────────────────────────────────────────────────
    def euler_to_quaternion(self, roll, pitch, yaw):
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        q = Quaternion()
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy
        return q


def main(args=None):
    rclpy.init(args=args)
    node = RobotBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()