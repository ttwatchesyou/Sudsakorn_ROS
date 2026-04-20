import math
import time
from geometry_msgs.msg import Twist

class NavigationSystem:
    """
    ระบบนำทางสำหรับหุ่นยนต์ Mecanum รองรับการปรับความเร็วรายครั้ง
    และระบบชะลอความเร็วแบบอัตโนมัติเมื่อเข้าใกล้จุดหมาย (Smooth Deceleration)
    """
    def __init__(self, node):
        self.node = node
        
        # --- ค่าเริ่มต้นมาตรฐาน (Default Settings) ---
        self.default_cruise_speed = 0.6    # ความเร็วมาตรฐาน (m/s)
        self.default_rotate_speed = 0.3    # ความเร็วหมุนมาตรฐาน (rad/s)
        
        self.dist_tolerance = 0.05         # ระยะหยุด (5 cm)
        self.yaw_tolerance = 0.05          # มุมหยุด (~3 degree)
        
        self.slow_down_radius = 0.40       # เริ่มชะลอความเร็วที่ระยะ 40 cm (ขยายรัศมีเพื่อให้ลดความเร็วได้นิ่มขึ้น)
        self.min_linear_speed = 0.06       # ความเร็วต่ำสุดที่จะประคองให้ถึงจุด (ไม่ให้มอเตอร์ค้าง)
        self.min_rotate_speed = 0.04       # ความเร็วหมุนต่ำสุด

        # --- ตัวแปรควบคุมภายใน ---
        self.current_cruise = self.default_cruise_speed
        self.current_rotate = self.default_rotate_speed
        
        self.ramp_duration = 0.5
        self.start_time = 0.0
        
        self.start_x = 0.0
        self.start_y = 0.0
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_yaw = 0.0
        
        self.current_mode = "DIRECT"
        self.is_active = False
        self.arrived = False

    def set_goal(self, x, y, yaw, curr_x, curr_y, mode="DIRECT", 
                 cruise_speed=None, rotate_speed=None, ramp_time=0.5):
        """
        ตั้งค่าเป้าหมายพร้อมปรับแต่งความเร็ว
        """
        self.goal_x = x
        self.goal_y = y
        self.goal_yaw = yaw
        self.start_x = curr_x
        self.start_y = curr_y
        self.current_mode = mode
        
        self.current_cruise = cruise_speed if cruise_speed is not None else self.default_cruise_speed
        self.current_rotate = rotate_speed if rotate_speed is not None else self.default_rotate_speed
        
        self.ramp_duration = ramp_time
        self.start_time = time.time()
        self.is_active = True
        self.arrived = False
        
        self.node.get_logger().info(f"📍 กำหนดเป้าหมาย: ({x:.2f}, {y:.2f}) โหมด: {mode} ความเร็วสูงสุด: {self.current_cruise}")

    def calculate_velocity(self, curr_x, curr_y, curr_yaw):
        if not self.is_active:
            return Twist(), False

        # 1. คำนวณหา Error
        err_x = self.goal_x - curr_x
        err_y = self.goal_y - curr_y
        err_yaw = self.goal_yaw - curr_yaw
        dist = math.sqrt(err_x**2 + err_y**2)

        # Normalize Yaw (-pi ถึง pi)
        while err_yaw > math.pi: err_yaw -= 2.0 * math.pi
        while err_yaw < -math.pi: err_yaw += 2.0 * math.pi

        # 2. เช็คว่าถึงเป้าหมายหรือยัง
        if dist < self.dist_tolerance and abs(err_yaw) < self.yaw_tolerance:
            self.is_active = False
            self.arrived = True
            return Twist(), True

        # 3. กำหนดทิศทางการวิ่ง (Global Frame)
        if dist > 0:
            dir_x = err_x / dist
            dir_y = err_y / dist
        else:
            dir_x, dir_y = 0.0, 0.0

        # 4. ระบบ Soft Start (Ramp-up)
        elapsed_time = time.time() - self.start_time
        ramp_scale = min(elapsed_time / self.ramp_duration, 1.0)
        
        # 5. ระบบลดความเร็วแบบทวีคูณ (Non-linear Deceleration)
        # ใช้การยกกำลัง (Exponential/Power) เพื่อให้ความเร็วลดลงอย่างรวดเร็วเมื่อเข้าใกล้จุด
        if dist < self.slow_down_radius:
            slow_ratio = dist / self.slow_down_radius
            # ใช้ Quadratic Scaling (ratio^2) เพื่อให้ความเร็วลดลงแบบทวีคูณ
            # สิ่งนี้จะช่วยลดแรงเฉื่อยได้อย่างมากเมื่อใกล้ถึง Tolerance
            scaled_ratio = math.pow(slow_ratio, 2) 
            current_target_speed = self.min_linear_speed + (self.current_cruise - self.min_linear_speed) * scaled_ratio
            slow_scale = current_target_speed / self.current_cruise
        else:
            slow_scale = 1.0
        
        final_linear_scale = ramp_scale * slow_scale

        # 6. คำนวณความเร็ว Global พร้อมตัวคูณลดความเร็ว (Applied to Mode Logic)
        if self.current_mode == "X_FIRST" and abs(err_x) > self.dist_tolerance:
            vx_global = (err_x / abs(err_x)) * self.current_cruise
            # ปรับ Gain การชดเชย (Correction) ให้ลดลงตามความเร็วหลัก (slow_scale) เพื่อป้องกันการแกว่ง
            vy_global = (self.start_y - curr_y) * (2.5 * slow_scale) 
        elif self.current_mode == "Y_FIRST" and abs(err_y) > self.dist_tolerance:
            # ปรับ Gain การชดเชย (Correction) ให้ลดลงตามความเร็วหลัก (slow_scale) เพื่อป้องกันการแกว่ง
            vx_global = (self.start_x - curr_x) * (2.5 * slow_scale) 
            vy_global = (err_y / abs(err_y)) * self.current_cruise
        else:
            # โหมด DIRECT หรือช่วงสไลด์เข้าจุดหมาย
            vx_global = dir_x * self.current_cruise
            vy_global = dir_y * self.current_cruise

        # 7. แปลงเป็น Local Frame ของหุ่นยนต์ (Mecanum Kinematics)
        local_x = (vx_global * math.cos(curr_yaw)) + (vy_global * math.sin(curr_yaw))
        local_y = -(vx_global * math.sin(curr_yaw)) + (vy_global * math.cos(curr_yaw))

        # 8. คำนวณความเร็วการหมุน (Angular Velocity)
        # ชะลอการหมุนลงเมื่อมุมใกล้ถึงเป้าหมายด้วยอัตราทวีคูณเช่นกัน
        angular_dist = abs(err_yaw)
        if angular_dist < 0.4: # เริ่มชะลอหมุนที่ระยะ 0.4 rad (~23 องศา)
            ang_slow_ratio = angular_dist / 0.4
            # ใช้ Quadratic Scaling สำหรับการหมุน
            ang_scaled_ratio = math.pow(ang_slow_ratio, 2)
            current_ang_speed = self.min_rotate_speed + (self.current_rotate - self.min_rotate_speed) * ang_scaled_ratio
            vz = (err_yaw / angular_dist) * current_ang_speed
        else:
            vz = (err_yaw / angular_dist) * self.current_rotate if angular_dist > 0 else 0.0

        # 9. สร้างคำสั่ง Twist
        cmd = Twist()
        cmd.linear.x = local_x * final_linear_scale
        cmd.linear.y = local_y * final_linear_scale
        cmd.angular.z = vz

        return cmd, False

    def clamp(self, val, limit):
        return max(min(val, limit), -limit)