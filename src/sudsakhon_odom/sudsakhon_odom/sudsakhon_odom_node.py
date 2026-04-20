import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import serial
import time
import threading
import math
from tf2_ros import TransformBroadcaster

class SudsakhonOdomNode(Node):
    def __init__(self):
        super().__init__('sudsakhon_odom_node')
        
        # --- การตั้งค่า Serial ---
        port_name = '/dev/Controller_Base' 
        baud_rate = 115200
        
        # --- พารามิเตอร์ของ Dead Wheels (ปรับให้ตรงกับอุปกรณ์จริง) ---
        self.dead_wheel_radius = 0.027  # รัศมีล้ออิสระ (เช่น 24mm)
        self.ticks_per_rev = 600.0      # จำนวน Tick ต่อรอบของ Encoder X/Y
        self.meters_per_tick = (2.0 * math.pi * self.dead_wheel_radius) / self.ticks_per_rev

        try:
            self.serial_port = serial.Serial(port_name, baud_rate, timeout=0.1)
            time.sleep(2) 
            self.get_logger().info(f"✅ Connected to STM32 (Dead Wheels Mode) on {port_name}")
        except Exception as e:
            self.get_logger().error(f"❌ Serial Error: {e}")
            raise SystemExit

        # Subscribers
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.init_pose_cb, 10)
        
        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_br = TransformBroadcaster(self)

        # --- State Variables ---
        self.x = 0.0; self.y = 0.0; self.th = 0.0
        self.prev_tick_x = 0
        self.prev_tick_y = 0
        self.prev_yaw = 0.0
        self.yaw_offset = 0.0 
        
        self.first_read = True 
        self.last_time = self.get_clock().now()
        
        # Thread สำหรับอ่านค่าจาก Serial
        self.running = True
        self.read_thread = threading.Thread(target=self.serial_loop, daemon=True)
        self.read_thread.start()

    def init_pose_cb(self, msg):
        """ ฟังก์ชันรีเซ็ตตำแหน่งจาก GUI (2D Pose Estimate) """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        target_yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0-2.0*(q.y*q.y + q.z*q.z))
        self.yaw_offset = target_yaw - self.prev_yaw
        self.get_logger().info(f"📍 Odom Reset! X:{self.x:.2f} Y:{self.y:.2f} Yaw:{target_yaw:.2f}")

    def cmd_vel_cb(self, msg):
        """ ส่งคำสั่งความเร็วไปยัง STM32 """
        cmd = f"C{msg.linear.x:.3f}:{msg.linear.y:.3f}:{msg.angular.z:.3f}\n"
        if self.serial_port.is_open:
            self.serial_port.write(cmd.encode('utf-8'))

    def serial_loop(self):
        while self.running:
            if self.serial_port.in_waiting:
                try:
                    line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith("FB:"):
                        p = line.split(':')
                        # p[14] คือ tick_x, p[15] คือ tick_y (อ้างอิงจากโค้ด main.cpp ล่าสุด)
                        if len(p) >= 16:
                            raw_yaw = float(p[5])
                            curr_tick_x = int(p[14])
                            curr_tick_y = int(p[15])
                            self.process_dead_wheels(curr_tick_x, curr_tick_y, raw_yaw)
                except: pass
            time.sleep(0.005)

    def process_dead_wheels(self, tick_x, tick_y, raw_yaw):
        now = self.get_clock().now()
        if self.first_read:
            self.prev_tick_x = tick_x
            self.prev_tick_y = tick_y
            self.prev_yaw = raw_yaw
            self.last_time = now
            self.first_read = False
            return

        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 0: return

        # 1. คำนวณระยะทางที่เคลื่อนที่ได้ในแนวพิกัดหุ่นยนต์ (Local Frame)
        # ใช้ Ticks จากล้ออิสระโดยตรง ไม่ผ่านค่าเฉลี่ยของล้อขับ
        dx_local = (tick_x - self.prev_tick_x) * self.meters_per_tick
        dy_local = (tick_y - self.prev_tick_y) * self.meters_per_tick
        
        # 2. คำนวณมุม (Yaw)
        dyaw = raw_yaw - self.prev_yaw
        while dyaw > math.pi: dyaw -= 2*math.pi
        while dyaw < -math.pi: dyaw += 2*math.pi
        
        # ทิศทางหุ่นในพิกัดโลก
        self.th = raw_yaw + self.yaw_offset
        while self.th > math.pi: self.th -= 2*math.pi
        while self.th < -math.pi: self.th += 2*math.pi

        # 3. แปลงระยะทาง Local เป็นพิกัด Global (X, Y ของแผนที่โลก)
        # displacement = R(theta) * local_displacement
        delta_x_world = (dx_local * math.cos(self.th)) - (dy_local * math.sin(self.th))
        delta_y_world = (dx_local * math.sin(self.th)) + (dy_local * math.cos(self.th))

        self.x += delta_x_world
        self.y += delta_y_world

        # เก็บค่าปัจจุบันไว้ใช้ในรอบหน้า
        self.prev_tick_x = tick_x
        self.prev_tick_y = tick_y
        self.prev_yaw = raw_yaw
        self.last_time = now

        # --- ส่วนของการส่งข้อมูลออก (Publish & TF) ---
        q_z = math.sin(self.th / 2.0)
        q_w = math.cos(self.th / 2.0)

        # Broadcast TF: odom -> base_link
        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation.z = q_z
        t.transform.rotation.w = q_w
        self.tf_br.sendTransform(t)

        # Publish Odometry Message
        msg = Odometry()
        msg.header = t.header
        msg.child_frame_id = 'base_link'
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.orientation = t.transform.rotation
        
        # ความเร็วปัจจุบัน (m/s)
        msg.twist.twist.linear.x = dx_local / dt
        msg.twist.twist.linear.y = dy_local / dt
        msg.twist.twist.angular.z = dyaw / dt
        
        self.odom_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    n = SudsakhonOdomNode()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    finally:
        n.running = False
        if n.serial_port.is_open: n.serial_port.close()
        n.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()