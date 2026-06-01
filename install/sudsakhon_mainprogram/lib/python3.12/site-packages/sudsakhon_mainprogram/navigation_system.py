import math
import time
from geometry_msgs.msg import Twist


class NavigationSystem:
    """
    ระบบนำทาง Mecanum รองรับ DIRECT และ BEZIER (โค้ง หน้าไม่หมุน)

    FIX 1: Phase advance ตอนนี้ return Twist() หยุด 1 tick ก่อน
           ป้องกัน residual velocity จาก Phase 0 ไหลเข้า Phase 1
    FIX 2: Phase 0 decel ใส่ min_speed floor 0.05 m/s
           ป้องกัน odometry drift ตอนหุ่นเกือบหยุด
    FIX 3: Phase 1 LOCK_Y velocity frame ตรวจสอบแล้ว + เพิ่ม TOL hysteresis
           ป้องกัน oscillate รอบ TOL แล้วไม่ advance
    FIX 4: 🌟 เพิ่ม Feedback Control ดึงหุ่นกลับเข้าเส้นโค้งแบบ Real-time
    """

    def __init__(self, node):
        self.node = node

        self.default_cruise_speed = 1.0
        self.default_rotate_speed = 0.3
        self.dist_tolerance       = 0.06   # m
        self.yaw_tolerance        = 0.05   # rad
        self.min_rotate_speed     = 0.04

        # ── State ─────────────────────────────────────────────────────────────
        self.current_cruise = self.default_cruise_speed
        self.current_rotate = self.default_rotate_speed
        self.ramp_duration  = 0.4
        self.start_time     = 0.0
        self.start_x        = 0.0
        self.start_y        = 0.0
        self.goal_x         = 0.0
        self.goal_y         = 0.0
        self.goal_yaw       = 0.0
        self.current_mode   = "DIRECT"
        self.is_active      = False
        self.arrived        = False

        # ── Bezier state ──────────────────────────────────────────────────────
        self._bezier_phase      = 0     # 0=วิ่งโค้ง 1=LOCK_Y 2=LOCK_X
        self._bezier_t          = 0.0
        self._p1_x              = 0.0
        self._p1_y              = 0.0
        self._phase_stop_tick   = False # flag หยุด 1 tick ตอน advance
        self._lock_gap          = 0.30  # m — เริ่ม LOCK_Y ก่อนถึง goal
        self._log_throttle      = 0.0

        # ── Bezier curve tuning ───────────────────────────────────────────────
        self.default_curve_strength   = 0.15 

        self.bezier_decel_ratio       = 0.25 

        self.bezier_curve_speed_ratio = 0.65 

        self.Kp = 1.5 # 🌟 จูนตรงนี้! ถ้าหลุด 10 cm ให้ใส่ประมาณ 3.0 - 4.5

        # ── DIRECT state ──────────────────────────────────────────────────────
        self.slow_down_radius = 0.50
        self.min_linear_speed = 0.06

    # ══════════════════════════════════════════════════════════════════════════
    # SET GOAL — DIRECT
    # ══════════════════════════════════════════════════════════════════════════
    def set_goal(self, x, y, yaw, curr_x, curr_y, mode="DIRECT",
                 cruise_speed=None, rotate_speed=None, ramp_time=0.4):
        self.is_active      = True
        self.arrived        = False
        self.goal_x       = x
        self.goal_y       = y
        self.goal_yaw     = yaw
        self.start_x      = curr_x
        self.start_y      = curr_y
        self.current_mode = mode
        self.current_cruise = cruise_speed or self.default_cruise_speed
        self.current_rotate = rotate_speed or self.default_rotate_speed
        self.ramp_duration  = ramp_time
        self.start_time     = time.time()
        
        self.node.get_logger().info(
            f"📍 Goal:({x:.2f},{y:.2f}) speed:{self.current_cruise}")

    # ══════════════════════════════════════════════════════════════════════════
    # SET BEZIER GOAL — โค้ง หน้าไม่หมุน
    # ══════════════════════════════════════════════════════════════════════════
    def set_bezier_goal(self, x, y, curr_x, curr_y, curr_yaw,
                        curve_side="AUTO", cruise_speed=None,
                        curve_strength=None,curve_kp=1.5):
        self.goal_x       = x
        self.goal_y       = y
        self.goal_yaw     = curr_yaw   # หน้าคงเดิม
        self.start_x      = curr_x
        self.start_y      = curr_y
        self.current_mode = "BEZIER"
        self.current_cruise = cruise_speed or self.default_cruise_speed
        self.ramp_duration  = 0.4
        self.start_time     = time.time()
        self.is_active      = True
        self.arrived        = False
        self._bezier_phase    = 0
        self._bezier_t        = 0.0
        self._phase_stop_tick = False
        self._log_throttle    = 0.0
        self.Kp = curve_kp
        _curve_strength = curve_strength if curve_strength is not None \
                          else self.default_curve_strength

        # ── Control Point P1 ──────────────────────────────────────────────────
        dist         = math.sqrt((x-curr_x)**2 + (y-curr_y)**2)
        offset       = dist * _curve_strength
        angle_path   = math.atan2(y-curr_y, x-curr_x)
        perp         = angle_path + math.pi / 2.0

        if curve_side == "LEFT":
            sign =  1.0
        elif curve_side == "RIGHT":
            sign = -1.0
        else:  # AUTO
            dx    = x - curr_x
            dy    = y - curr_y
            cross = math.cos(curr_yaw)*dy - math.sin(curr_yaw)*dx
            sign  = 1.0 if cross >= 0 else -1.0

        mid_x      = (curr_x + x) / 2.0
        mid_y      = (curr_y + y) / 2.0
        self._p1_x = mid_x + sign * offset * math.cos(perp)
        self._p1_y = mid_y + sign * offset * math.sin(perp)

        self.node.get_logger().info(
            f"🌀 Bezier:({x:.2f},{y:.2f}) "
            f"P1:({self._p1_x:.2f},{self._p1_y:.2f}) "
            f"side:{curve_side} dist:{dist:.2f}m speed:{self.current_cruise}")

    def set_bezier_goal_new(self, x, y, target_yaw, curr_x, curr_y, curr_yaw, 
                        curve_side="AUTO", cruise_speed=None, 
                        curve_strength=None, curve_kp=1.5):
        
        self.goal_x       = x
        self.goal_y       = y
        self.goal_yaw     = target_yaw   # เปลี่ยนให้รับค่ามุมเป้าหมายที่ต้องการ
        
        self.start_x      = curr_x
        self.start_y      = curr_y
        self.start_yaw    = curr_yaw     # เพิ่มการเก็บค่ามุมเริ่มต้น เพื่อนำไปใช้คำนวณระหว่างวิ่ง
        
        self.current_mode = "BEZIER"
        self.current_cruise = cruise_speed or self.default_cruise_speed
        self.ramp_duration  = 0.4
        self.start_time     = time.time()
        self.is_active      = True
        self.arrived        = False
        self._bezier_phase  = 0
        self._bezier_t      = 0.0
        self._phase_stop_tick = False
        self._log_throttle    = 0.0
        self.Kp = curve_kp
        _curve_strength = curve_strength if curve_strength is not None \
                          else self.default_curve_strength

        # ── Control Point P1 ──────────────────────────────────────────────────
        dist         = math.sqrt((x - curr_x)**2 + (y - curr_y)**2)
        offset       = dist * _curve_strength
        angle_path   = math.atan2(y - curr_y, x - curr_x)
        perp         = angle_path + math.pi / 2.0

        if curve_side == "LEFT":
            sign =  1.0
        elif curve_side == "RIGHT":
            sign = -1.0
        else:  # AUTO
            dx    = x - curr_x
            dy    = y - curr_y
            cross = math.cos(curr_yaw)*dy - math.sin(curr_yaw)*dx
            sign  = 1.0 if cross >= 0 else -1.0

        mid_x      = (curr_x + x) / 2.0
        mid_y      = (curr_y + y) / 2.0
        self._p1_x = mid_x + sign * offset * math.cos(perp)
        self._p1_y = mid_y + sign * offset * math.sin(perp)

        self.node.get_logger().info(
            f"🌀 Bezier:({x:.2f},{y:.2f}, yaw:{target_yaw:.2f}) "
            f"P1:({self._p1_x:.2f},{self._p1_y:.2f}) "
            f"side:{curve_side} dist:{dist:.2f}m speed:{self.current_cruise}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # CALCULATE VELOCITY
    # ══════════════════════════════════════════════════════════════════════════
    def calculate_velocity(self, curr_x, curr_y, curr_yaw):
        if not self.is_active:
            return Twist(), False
        if self.current_mode == "BEZIER":
            return self._calc_bezier(curr_x, curr_y, curr_yaw)
        return self._calc_direct(curr_x, curr_y, curr_yaw)

    # ══════════════════════════════════════════════════════════════════════════
    # DIRECT
    # ══════════════════════════════════════════════════════════════════════════
    def _calc_direct(self, curr_x, curr_y, curr_yaw):
        err_x   = self.goal_x - curr_x
        err_y   = self.goal_y - curr_y
        err_yaw = self.goal_yaw - curr_yaw
        dist    = math.sqrt(err_x**2 + err_y**2)

        while err_yaw >  math.pi: err_yaw -= 2*math.pi
        while err_yaw < -math.pi: err_yaw += 2*math.pi

        if dist < self.dist_tolerance and abs(err_yaw) < self.yaw_tolerance:
            self.is_active = False
            self.arrived   = True
            return Twist(), True

        dir_x = err_x/dist if dist > 0 else 0.0
        dir_y = err_y/dist if dist > 0 else 0.0

        elapsed    = time.time() - self.start_time
        ramp_scale = min(elapsed / self.ramp_duration, 1.0)
        if dist < self.slow_down_radius:
            ratio      = math.pow(dist / self.slow_down_radius, 1.5)
            slow_speed = self.min_linear_speed + \
                         (self.current_cruise - self.min_linear_speed) * ratio
            slow_scale = slow_speed / self.current_cruise
        else:
            slow_scale = 1.0

        speed   = self.current_cruise * ramp_scale * slow_scale
        vx_g    = dir_x * speed
        vy_g    = dir_y * speed
        local_x =  vx_g * math.cos(curr_yaw) + vy_g * math.sin(curr_yaw)
        local_y = -vx_g * math.sin(curr_yaw) + vy_g * math.cos(curr_yaw)

        ang = abs(err_yaw)
        if ang < 0.4:
            r   = math.pow(ang/0.4, 2)
            spd = self.min_rotate_speed + \
                  (self.current_rotate - self.min_rotate_speed) * r
            vz  = (err_yaw/ang)*spd if ang > 0 else 0.0
        else:
            vz = (err_yaw/ang)*self.current_rotate if ang > 0 else 0.0

        cmd = Twist()
        cmd.linear.x  = local_x
        cmd.linear.y  = local_y
        cmd.angular.z = vz
        return cmd, False

    # ══════════════════════════════════════════════════════════════════════════
    # BEZIER — state machine 3 phase
    # ══════════════════════════════════════════════════════════════════════════
    def _calc_bezier(self, curr_x, curr_y, curr_yaw):
        """
        Phase 0 CURVE : วิ่งตาม Bezier tangent + Feedback ดึงเข้าเส้น (รักษาหน้าหุ่นคงที่)
        Phase 1 LOCK_Y: ดึง Y ให้เข้าเป้า
        Phase 2 LOCK_X: ดึง X และ Y ย้ำอีกครั้งจนเข้าเป้าเป๊ะๆ + หน่วงเวลา(Settling Time) → arrived
        """
        p2x, p2y  = self.goal_x, self.goal_y
        err_x     = p2x - curr_x
        err_y     = p2y - curr_y
        dist      = math.sqrt(err_x**2 + err_y**2)
        dist_full = math.sqrt((p2x-self.start_x)**2 + (p2y-self.start_y)**2)

        TOL_ENTER    = 0.07   # m — หยุด P-control เมื่อเข้าใกล้ ก่อน advance
        TOL_HOLD     = 0.03   # 🌟 ปรับจาก 0.04 เป็น 0.03 เพื่อให้ระยะทแยงรวมไม่เกิน 0.05 ของ Main Controller
        MAX_SPD_LOCK = 0.35   # m/s สูงสุดขณะ lock
        MIN_CURVE_SPD = 0.05  # m/s floor ป้องกัน drift ตอน decel

        decel_radius = max(dist_full * self.bezier_decel_ratio, 0.30)

        # -------------------------------------------------------------
        # 🔄 รักษาหน้าหุ่น (เป้าหมายคงที่ ไม่บิดตัวตามโค้ง)
        # -------------------------------------------------------------
        yaw_err = self.goal_yaw - curr_yaw
        while yaw_err >  math.pi: yaw_err -= 2.0 * math.pi
        while yaw_err < -math.pi: yaw_err += 2.0 * math.pi
        v_yaw = 2.0 * yaw_err 

        # ── Phase advance ────────────────────────────────────────────────────
        if self._bezier_phase == 0 and (dist < self._lock_gap or self._bezier_t >= 1.0):
            self._bezier_phase    = 1
            self._phase_stop_tick = True
            self.node.get_logger().info(
                f"🔒 LOCK_Y triggered  dist:{dist:.3f}m  t:{self._bezier_t:.2f}  "
                f"err_x:{err_x:.3f}  err_y:{err_y:.3f}")

        if self._bezier_phase == 1 and abs(err_y) <= TOL_HOLD:
            self._bezier_phase    = 2
            self._phase_stop_tick = True
            self._arrive_tick     = 0   # 🌟 รีเซ็ตตัวหน่วงเวลาเมื่อเตรียมตัวหยุด
            self.node.get_logger().info(
                f"🔒 LOCK_X triggered  err_x:{err_x:.3f}  err_y:{err_y:.3f}")

        if self._phase_stop_tick:
            self._phase_stop_tick = False
            return Twist(), False

        # ────────────────────────────────────────────────────────────────────
        # PHASE 0 : CURVE + DECEL + 🌟 FEEDBACK CONTROL
        # ────────────────────────────────────────────────────────────────────
        if self._bezier_phase == 0:
            p0x, p0y = self.start_x, self.start_y
            p1x, p1y = self._p1_x, self._p1_y

            # หา t ที่ใกล้ position ปัจจุบันมากที่สุด
            best_t = self._bezier_t
            best_d = float('inf')
            expected_x, expected_y = curr_x, curr_y # ค่าตั้งต้น
            
            for i in range(15):
                ti = min(self._bezier_t + i*0.015, 1.0)
                bx = (1-ti)**2*p0x + 2*(1-ti)*ti*p1x + ti**2*p2x
                by = (1-ti)**2*p0y + 2*(1-ti)*ti*p1y + ti**2*p2y
                d  = math.sqrt((bx-curr_x)**2 + (by-curr_y)**2)
                if d < best_d:
                    best_d = d; best_t = ti
                    expected_x, expected_y = bx, by # จำพิกัดที่ควรจะอยู่ไว้ด้วย
                    
            self._bezier_t = min(best_t, 1.0)
            t = self._bezier_t

            # Feedforward Vector (วิ่งตามเส้น)
            if t < 0.999:
                tdx = 2*(1-t)*(p1x-p0x) + 2*t*(p2x-p1x)
                tdy = 2*(1-t)*(p1y-p0y) + 2*t*(p2y-p1y)
                mag = math.sqrt(tdx**2 + tdy**2) or 1.0
                nx, ny = tdx/mag, tdy/mag
            else:
                mag = dist or 1.0
                nx, ny = err_x/mag, err_y/mag

            elapsed    = time.time() - self.start_time
            ramp_scale = min(elapsed / self.ramp_duration, 1.0)

            if dist < decel_radius:
                decel_scale = dist / decel_radius
            else:
                decel_scale = 1.0

            max_curve_spd = self.current_cruise * self.bezier_curve_speed_ratio
            ff_speed = max_curve_spd * ramp_scale * decel_scale
            ff_speed = max(ff_speed, MIN_CURVE_SPD)

            # -------------------------------------------------------------
            # 🌟 Feedback Control (ดึงหุ่นสู้แรงสลิป)
            # -------------------------------------------------------------
            fb_vx = self.Kp * (expected_x - curr_x)
            fb_vy = self.Kp * (expected_y - curr_y)
            
            global_vx = (nx * ff_speed) + fb_vx
            global_vy = (ny * ff_speed) + fb_vy
            
            speed_mag = math.sqrt(global_vx**2 + global_vy**2)
            max_allowed = ff_speed * 1.3 
            if speed_mag > max_allowed:
                global_vx = (global_vx / speed_mag) * max_allowed
                global_vy = (global_vy / speed_mag) * max_allowed

            # Transform Global -> Local (Mecanum Frame)
            local_x =  global_vx * math.cos(curr_yaw) + global_vy * math.sin(curr_yaw)
            local_y = -global_vx * math.sin(curr_yaw) + global_vy * math.cos(curr_yaw)

            now = time.time()
            if now - self._log_throttle >= 0.2:
                self._log_throttle = now
                self.node.get_logger().info(
                    f"[CURVE+FB] t:{t:.2f}  dist:{dist:.3f}m  "
                    f"err_drift:{best_d:.3f}m  "
                    f"lx:{local_x:.2f}  ly:{local_y:.2f}")

            cmd = Twist()
            cmd.linear.x = local_x
            cmd.linear.y = local_y
            cmd.angular.z = v_yaw
            return cmd, False

        # ────────────────────────────────────────────────────────────────────
        # PHASE 1 : LOCK_Y
        # ────────────────────────────────────────────────────────────────────
        if self._bezier_phase == 1:
            if abs(err_y) <= TOL_ENTER:
                return Twist(), False

            spd_y = err_y * 2.0
            spd_y = max(min(spd_y, MAX_SPD_LOCK), -MAX_SPD_LOCK)

            local_x =  spd_y * math.sin(curr_yaw)
            local_y =  spd_y * math.cos(curr_yaw)

            cmd = Twist()
            cmd.linear.x = local_x
            cmd.linear.y = local_y
            cmd.angular.z = v_yaw # รักษาหน้าหุ่นกันหมุนเพี้ยน
            self.node.get_logger().info(
                f"[LOCK_Y]  err_x:{err_x:.3f}  err_y:{err_y:.3f}  spd_y:{spd_y:.3f}")
            return cmd, False

        # ────────────────────────────────────────────────────────────────────
        # PHASE 2 : LOCK_X (🌟และ Y พร้อมกัน) + ⏳ Settling Time → ARRIVED
        # ────────────────────────────────────────────────────────────────────
        if self._bezier_phase == 2:
            
            # 🌟 เช็คว่าอยู่ในระยะหรือยัง
            if abs(err_x) <= TOL_HOLD and abs(err_y) <= TOL_HOLD:
                if not hasattr(self, '_arrive_tick'): self._arrive_tick = 0
                self._arrive_tick += 1
                
                # ⏳ หน่วงเวลา 5 ลูป (ประมาณ 0.25 วินาที) สั่งให้รถหยุดนิ่งเพื่อความชัวร์ว่าไม่ไถล
                if self._arrive_tick >= 5:
                    self.is_active     = False
                    self.arrived       = True
                    self._bezier_phase = 0
                    self.node.get_logger().info(
                        f"✅ ARRIVED (SETTLED)  x:{curr_x:.3f}  y:{curr_y:.3f}  "
                        f"err_x:{err_x:.3f}  err_y:{err_y:.3f}")
                    return Twist(), True
                else:
                    # อยู่ในเป้าแล้ว แต่รอให้ชัวร์ -> สั่งเบรก (หยุดล้อ) ให้ความเร็ว x, y เป็น 0
                    cmd = Twist()
                    cmd.angular.z = v_yaw
                    return cmd, False
            else:
                self._arrive_tick = 0 # 🌟 ถ้าไถลหลุดเป้า ให้เริ่มนับการหน่วงเวลาใหม่

            # แก้อาการ X ไหล
            spd_x = err_x * 2.0
            spd_x = max(min(spd_x, MAX_SPD_LOCK), -MAX_SPD_LOCK)

            # แก้อาการ Y ไหล
            spd_y = err_y * 2.0
            spd_y = max(min(spd_y, MAX_SPD_LOCK), -MAX_SPD_LOCK)

            local_x =  spd_x * math.cos(curr_yaw) + spd_y * math.sin(curr_yaw)
            local_y = -spd_x * math.sin(curr_yaw) + spd_y * math.cos(curr_yaw)

            cmd = Twist()
            cmd.linear.x = local_x
            cmd.linear.y = local_y
            cmd.angular.z = v_yaw # รักษาหน้าหุ่นกันหมุนเพี้ยน
            self.node.get_logger().info(
                f"[LOCK_XY] err_x:{err_x:.3f} err_y:{err_y:.3f} spd_x:{spd_x:.3f} spd_y:{spd_y:.3f}")
            return cmd, False

        return Twist(), False
    def clamp(self, val, limit):
        return max(min(val, limit), -limit)