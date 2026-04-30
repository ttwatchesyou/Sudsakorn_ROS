import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Pose2D
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32MultiArray, Int8MultiArray, Float32, Float32MultiArray
from sensor_msgs.msg import LaserScan
import math
import time
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Int32

try:
    from .navigation_system import NavigationSystem
    from .sensor_system import SensorSystem
    from .servo_system import ServoSystem
    from .actuators import SlideSystem, BottleSystem, BucketSystem
except ImportError:
    from navigation_system import NavigationSystem
    from sensor_system import SensorSystem
    from servo_system import ServoSystem
    from actuators import SlideSystem, BottleSystem, BucketSystem




class SudSakhonMainController(Node):
    def __init__(self):
        super().__init__('main_controller')

        # ── Publishers ────────────────────────────────────────────────────────
        self.cmd_pub       = self.create_publisher(Int32MultiArray,  'motor_commands',  10)
        self.vel_pub       = self.create_publisher(Twist,            '/cmd_vel',         10)
        self.goal_pub      = self.create_publisher(Pose2D,           '/cmd_goal',        10)
        self.max_speed_pub = self.create_publisher(Float32,          '/cmd_max_speed',   10)
        self.pos_pid_pub   = self.create_publisher(Float32MultiArray,'/cmd_pos_pid',     10)
        self.yaw_pid_pub   = self.create_publisher(Float32MultiArray,'/cmd_yaw_pid',     10)
        self.marker_pub    = self.create_publisher(Marker,           'detected_marker',  10)
        self.chair_params  = self.create_publisher(Float32MultiArray, '/chair_params', 10)
        self.control_servo  = self.create_publisher(Int32MultiArray, '/automation/servo', 10)
        self.control_states  = self.create_publisher(Int32MultiArray, '/automation/control_states', 10)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Int32MultiArray, '/automation/sensors',  self.sensor_callback,   10)
        self.create_subscription(Odometry,       '/odom',          self.odom_callback,     10)
        self.create_subscription(PoseStamped,    '/goal_pose',     self.rviz_goal_callback,10)
        self.create_subscription(Int32,          '/chair_count',   self.chaircount_callback,10)
        self.create_subscription(LaserScan,      'scan',           self.filter_callback,   10)
        self.create_subscription(Int32,          '/Program/Color',   self.program_color_callback,10)
        self.create_subscription(Int32,          '/Program/Game',   self.program_game_callback,10)
        self.create_subscription(Int32,          '/Program/Command',   self.program_command_callback,10)

        # ── State ─────────────────────────────────────────────────────────────
        self.ServoGriper_R = [90,120]
        self.ServoGriper_L = [90,60]

        self.StateControlAutomation = [0,0,0,0]

        self.ColorRed = 0
        self.ColorBlue = 1
        self.Programcolor = 0
        self.ProgramGame = 0

        self.curr_x   = 0.0
        self.curr_y   = 0.0
        self.curr_yaw = 0.0

        self.target_x   = 0.0
        self.target_y   = 0.0
        self.target_yaw = 0.0

        self.mission_step   = 0
        self.control_mode   = "DIRECT_STM32"
        self.manual_vx      = 0.0
        self.manual_vy      = 0.0
        self.current_state  = [0, 0, 0, 0, 0, 0, 0]

        self.dist_tolerance = 0.05
        self.yaw_tolerance  = math.radians(3.0)

        self.ChairCount     = 0
        self.ChairCount_RUN = 0
        self.counterboxout  = 0

        self.ProgramCommand = 0
        self.ProgramCommand_Start = 1
        self.ProgramCommand_Reset = 2

        self.Lidar_left_dist   = None
        self.Lidar_right_dist  = None
        self.Lidar_center_dist = None

        # ── Systems ───────────────────────────────────────────────────────────
        self.slide_system  = SlideSystem(self)
        self.bottle_system = BottleSystem(self)
        self.bucket_system = BucketSystem(self)
        self.servo_system  = ServoSystem(self)
        self.sensors       = SensorSystem()
        self.nav           = NavigationSystem(self)
        


        self.Chair_params_RED    = [0.15, 1.2, 0.5, 0.05, 180.0, 270.0]
        self.Chair_params_BLUE   = [0.15, 1.2, 0.5, 0.05, 270.0, 90.0]

        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info('✅ SudSakhon Controller Ready')

        self.send_parameters(0.15, 1.2, 0.5, 0.05, 180.0, 270.0)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP HELPERS — ใช้ใน control_loop เพื่อให้อ่านง่าย
    # ══════════════════════════════════════════════════════════════════════════

    def send_parameters(self, min_w, max_w, threshold, gap, min_deg, max_deg):
        """ ฟังก์ชันสำหรับส่งค่าพารามิเตอร์ """
        msg = Float32MultiArray()
        msg.data = [
            float(min_w), 
            float(max_w), 
            float(threshold), 
            float(gap), 
            float(min_deg), 
            float(max_deg)
        ]
        
        # ส่งข้อความ
        self.chair_params.publish(msg)
        self.get_logger().info(f'ส่งค่าสำเร็จ: {msg.data}')

    def ControlServo(self,Number,Control):
        msg = Int32MultiArray()
        msg.data = [
            Number,Control
        ]
        # ส่งข้อความ
        self.control_servo.publish(msg)

    def ControlAutomation(self):
        msg = Int32MultiArray()
        msg.data = self.StateControlAutomation
        # ส่งข้อความ
        self.control_states.publish(msg)

    def step(self, current, expected):
        """เช็คว่าอยู่ step ที่ถูกต้องไหม"""
        return self.mission_step == expected or self.mission_step == current

    def next_step(self, step_num):
        """ไป step ถัดไป"""
        self.get_logger().info(f'➡️  Step {self.mission_step} → {step_num}')
        self.mission_step = step_num

    def arrived(self, tol=None):
        """
        ถึงเป้าหมายแล้วไหม — แยกตาม mode:
          DIRECT_STM32 : เช็คจาก odom กับ target ตรงๆ
          INTERNAL     : เช็คจาก nav.arrived (state machine ของ bezier/direct)
        """
        if self.control_mode == "INTERNAL":
            # ให้ NavigationSystem เป็นคนตัดสินว่าถึงหรือยัง
            return self.nav.arrived

        # DIRECT_STM32 / MANUAL — เช็คจาก odom
        t    = tol if tol else self.dist_tolerance
        dist = math.sqrt((self.target_x - self.curr_x)**2 +
                         (self.target_y - self.curr_y)**2)
        yaw_err = abs(self.target_yaw - self.curr_yaw)
        while yaw_err > math.pi:
            yaw_err = abs(yaw_err - 2 * math.pi)
        return dist < t and yaw_err < self.yaw_tolerance

    def wait_sensor(self, sensor_value, expected=0):
        """รอเซนเซอร์ให้ได้ค่าที่ต้องการ คืน True เมื่อตรง"""
        return sensor_value == expected

    def sleep_move(self, vx, vy, duration):
        """
        เคลื่อนที่ด้วยความเร็วคงที่ตาม duration แล้วหยุด
        ⚠️ ใช้ time.sleep — เหมาะกับ step ที่ต้องการระยะสั้นๆ
        """
        self.set_manual_velocity(vx, vy)
        time.sleep(duration)
        self.set_manual_velocity(0.0, 0.0)

    def do_servo(self, servo, action, delay_after=0.5):
        """
        สั่ง servo แล้วรอ
        action: 'on' หรือ 'off'
        """
        if action == 'on':
            servo.on()
        else:
            servo.off()
        if delay_after > 0:
            time.sleep(delay_after)

    def lidar_approach(self, lidar_dist, direction='x+',
                       stop_dist=0.30,
                       slow_dist=0.50,
                       cruise_speed=0.20,
                       slow_speed=0.06,
                       lateral_speed=0.06,
                       lateral_detect_dist=0.80,
                       align_kp=1.5):
        """
        คลานเข้าหาสิ่งกีดขวาง + lateral correction + align มุม

        Forward : ลดความเร็วตาม lidar หน้า (linear decel)
        Lateral : ขวาโล่ง→เลื่อนขวา / ซ้ายโล่ง→เลื่อนซ้าย
        Align   : ถ้าเจอซ้ายและขวาพร้อมกัน ใช้ผลต่าง (L-R) คำนวณ
                  angular.z เพื่อปรับให้หุ่นขนานกับผนัง/สิ่งกีดขวาง
        ทั้งสามทำงานพร้อมกันทุก loop

        Parameters:
          lidar_dist          : ค่า Lidar หน้า (self.Lidar_center_dist)
          direction           : 'x+' | 'x-' | 'y+' | 'y-'
          stop_dist           : หยุดเมื่อ lidar หน้า <= นี้ (m)
          slow_dist           : เริ่มชะลอเมื่อ lidar หน้า <= นี้ (m)
          cruise_speed        : ความเร็วปกติ (m/s)
          slow_speed          : ความเร็วต่ำสุดก่อนหยุด (m/s)
          lateral_speed       : ความเร็วเลื่อนข้าง (m/s)
          lateral_detect_dist : lidar ข้าง > ค่านี้ = ไม่เจอ (m)
          align_kp            : P gain สำหรับ angular alignment
                                เพิ่ม = หมุนเร็ว / ลด = หมุนช้า

        คืน True เมื่อถึง stop_dist
        """
        if lidar_dist is None:
            self.set_manual_velocity(0.0, 0.0)
            return False

        # ── Map direction ─────────────────────────────────────────────────────
        dir_map = {
            'x+': (( 1, 0), ( 0, 1)),
            'x-': ((-1, 0), ( 0,-1)),
            'y+': (( 0, 1), (-1, 0)),
            'y-': (( 0,-1), ( 1, 0)),
        }
        (fdx, fdy), (ldx, ldy) = dir_map.get(direction, (( 1, 0), ( 0, 1)))

        # ── ถึง stop_dist → หยุด ─────────────────────────────────────────────
        if lidar_dist <= stop_dist:
            self.set_manual_velocity(0.0, 0.0)
            self.get_logger().info(
                f'🛑 Lidar stop  front:{lidar_dist:.3f}  '
                f'L:{self.Lidar_left_dist}  R:{self.Lidar_right_dist}')
            return True

        # ── Forward speed (linear decel) ──────────────────────────────────────
        if lidar_dist <= slow_dist:
            ratio   = (lidar_dist - stop_dist) / (slow_dist - stop_dist)
            ratio   = max(0.0, min(1.0, ratio))
            fwd_spd = slow_speed + (cruise_speed - slow_speed) * ratio
        else:
            fwd_spd = cruise_speed

        # ── Lateral + Align ───────────────────────────────────────────────────
        lat_spd = 0.0
        angular = 0.0

        L = self.Lidar_left_dist
        R = self.Lidar_right_dist

        left_missing  = (L is None) or (L > lateral_detect_dist)
        right_missing = (R is None) or (R > lateral_detect_dist)

        if not left_missing and not right_missing:
            # ── เจอทั้งคู่ → ปรับมุม (align) ─────────────────────────────────
            # ถ้าขนาน: L == R → diff = 0 → angular = 0
            # L > R: หัวเอียงซ้าย (ซ้ายไกลกว่า) → หมุนขวา (angular < 0)
            # L < R: หัวเอียงขวา (ขวาไกลกว่า) → หมุนซ้าย (angular > 0)
            diff    = L - R                        # บวก=เอียงซ้าย ลบ=เอียงขวา
            angular = -align_kp * diff             # P control
            angular = max(min(angular, 0.5), -0.5) # clamp ไม่หมุนเร็วเกิน

        elif right_missing and not left_missing:
            # ขวาโล่ง → เลื่อนขวา
            lat_spd = +lateral_speed

        elif left_missing and not right_missing:
            # ซ้ายโล่ง → เลื่อนซ้าย
            lat_spd = -lateral_speed
        # ไม่เจอทั้งคู่ → lat=0, angular=0 วิ่งตรง

        # ── รวม forward + lateral + angular ──────────────────────────────────
        vx = fdx * fwd_spd + ldx * lat_spd
        vy = fdy * fwd_spd + ldy * lat_spd

        # ส่งผ่าน cmd_vel โดยตรง (ต้องการ angular.z)
        msg = Twist()
        msg.linear.x  = float(vx)
        msg.linear.y  = float(vy)
        msg.angular.z = float(angular)
        self.control_mode = "MANUAL"
        self.manual_vx = float(vx)
        self.manual_vy = float(vy)
        self.vel_pub.publish(msg)
        return False

    # ══════════════════════════════════════════════════════════════════════════
    # CONTROL LOOP — เขียนเป็น step อ่านง่าย
    # ══════════════════════════════════════════════════════════════════════════

    def control_loop(self):

        #print(self.sensors.SW_1)
        
        if( self.sensors.SW_1 == 0 ):
            self.ControlBottle_L('up')
            #self.Griper_R('active')
            #self.Griper_L('active')

        if( self.sensors.SW_2 == 0 ):
            self.ControlBottle_L('down')
            #self.Griper_R('unactive')
            #self.Griper_L('unactive')
            #self.send_parameters(0.15, 1.2, 0.5, 0.05, 0.0, 90.0)


        '''
        if(self.ProgramCommand == self.ProgramCommand_Reset):
            self.mission_step = 0
            self.ProgramCommand = 0
            print("Reset")
        '''

        if self.Programcolor == self.ColorRed:
            if self.ProgramGame == 1:
        

                #print(str(self.mission_step) +" : "+str(self.arrived()))
                #Start ColorRed zero
                if self.mission_step == 0:
                    if self.sensors.SW_1 == 0 or self.ProgramCommand == self.ProgramCommand_Start:
                        #self.send_parameters(0.15, 1.2, 0.5, 0.05, 180.0, 270.0)
                        self.get_logger().info('🔘 เริ่มภารกิจ!')
                        self.ChairCount_RUN = self.ChairCount
                        #self.bucket_system.out_pos()
                        #self.slide_system.enable()
                        #self.slide_system.down()

                        #self.go_to(0.0, 0.0, 0.0, speed_limit=0.5, pos_pid=[1.480, 0.0, 0.121])

                        print(self.ChairCount_RUN)

                        self.next_step(1)

                #Table 1 Curve
                elif self.mission_step == 1 and self.arrived():
                    #self.go_to(-2.00, -0.0, 0.0,speed_limit=1.4,pos_pid=[1.1, 0.0, 0.0])
                    self.go_to_curve(-2.1, -1.8, speed_limit=1.0)
                    print("Mission 1")
                    self.next_step(2)

                # รอให้วิ่ง Curve เสร็จ
                elif self.mission_step == 2 and self.arrived():
                    print("Mission 2")
                    # ถึงจุด Curve แล้ว เด้งไปโหมด Lidar
                    self.next_step(2.5) 

                # โหมด Lidar Approach (❌ ห้ามใส่ and self.arrived() ตรงนี้เด็ดขาด)
                elif self.mission_step == 2.5: 
                    print(f"Lidar Dist: {self.Lidar_center_dist}") # ปรินต์เช็คค่าได้ปกติ
                    
                    if self.lidar_approach(self.Lidar_center_dist,
                                        direction='x+',
                                        stop_dist=0.20,
                                        slow_dist=0.30,
                                        cruise_speed=0.20,
                                        slow_speed=0.06):
                        self.next_step(3)

                elif self.mission_step == 3:
                    self.go_to(-2.25, -4.0, 0.0,
                            speed_limit=1.0,
                            pos_pid=[1.25, 0.0, 0.0])
                    self.next_step(4)

                # รอให้วิ่ง Curve เสร็จ
                elif self.mission_step == 4 and self.arrived():
                    # ถึงจุด Curve แล้ว เด้งไปโหมด Lidar
                    self.next_step(4.5) 

                # โหมด Lidar Approach (❌ ห้ามใส่ and self.arrived() ตรงนี้เด็ดขาด)
                elif self.mission_step == 4.5: 
                    print(f"Lidar Dist: {self.Lidar_center_dist}") # ปรินต์เช็คค่าได้ปกติ
                    
                    if self.lidar_approach(self.Lidar_center_dist,
                                        direction='x+',
                                        stop_dist=0.21,
                                        slow_dist=0.30,
                                        cruise_speed=0.20,
                                        slow_speed=0.06):
                        self.next_step(5)

                elif self.mission_step == 5 :
                    self.go_to(-2.35, -4.0, 0,
                            speed_limit=1.0,
                            pos_pid=[1.25, 0.0, 0.0])
                    self.next_step(5.5)

                elif self.mission_step == 5.5 and self.arrived():
                    self.go_to(-2.35, -4.0, 180,
                            speed_limit=1.0,
                            pos_pid=[1.25, 0.0, 0.0])
                    self.next_step(6)

                # รอให้วิ่ง Curve เสร็จ
                elif self.mission_step == 6 and self.arrived():
                    # ถึงจุด Curve แล้ว เด้งไปโหมด Lidar
                    self.next_step(6.5) 

                # โหมด Lidar Approach (❌ ห้ามใส่ and self.arrived() ตรงนี้เด็ดขาด)
                elif self.mission_step == 6.5: 
                    print(f"Lidar Dist: {self.Lidar_center_dist}") # ปรินต์เช็คค่าได้ปกติ
                    
                    if self.lidar_approach(self.Lidar_center_dist,
                                        direction='x+',
                                        stop_dist=0.25,
                                        slow_dist=0.30,
                                        cruise_speed=0.20,
                                        slow_speed=0.06):
                        self.next_step(7)

                elif self.mission_step == 7:
                    self.go_to(-2.5, -1.85, 180,
                            speed_limit=1.0,
                            pos_pid=[1.23, 0.0, 0.0])
                    self.next_step(8)


                # รอให้วิ่ง Curve เสร็จ
                elif self.mission_step == 8 and self.arrived():
                    # ถึงจุด Curve แล้ว เด้งไปโหมด Lidar
                    self.next_step(8.5) 

                # โหมด Lidar Approach (❌ ห้ามใส่ and self.arrived() ตรงนี้เด็ดขาด)
                elif self.mission_step == 8.5: 
                    print(f"Lidar Dist: {self.Lidar_center_dist}") # ปรินต์เช็คค่าได้ปกติ
                    
                    if self.lidar_approach(self.Lidar_center_dist,
                                        direction='x+',
                                        stop_dist=0.25,
                                        slow_dist=0.30,
                                        cruise_speed=0.20,
                                        slow_speed=0.06):
                        self.next_step(9)

                elif self.mission_step == 9 :
                    self.go_to(-2.2, -1.85, 180,
                            speed_limit=1.0,
                            pos_pid=[1.25, 0.0, 0.0])
                    self.next_step(10)
                
                elif self.mission_step == 10 and self.arrived():
                    self.go_to(-2.3, -1.85, 270,
                            speed_limit=1.0,
                            pos_pid=[1.25, 0.0, 0.0])
                    self.next_step(10.5)

                elif self.mission_step == 10.5 and self.arrived():
                    self.go_to(-2.3, -4, 270,
                            speed_limit=1.0,
                            pos_pid=[1.25, 0.0, 0.0])
                    self.next_step(11)

        '''
        
        # ────────────────────────────────────────────────────────────────────
        # STEP 0 — รอกดปุ่มเริ่ม
        # ────────────────────────────────────────────────────────────────────
        if self.mission_step == 0:
            if self.sensors.sw1 == 0:
                self.get_logger().info('🔘 เริ่มภารกิจ!')
                self.bucket_system.out_pos()
                self.slide_system.enable()
                self.slide_system.down()
                self.go_to(0.0, 0.0, 0.0,
                           speed_limit=0.5,
                           pos_pid=[13.5, 0.00002, 4.8])
                self.next_step(1)

        # ────────────────────────────────────────────────────────────────────
        # STEP 1 — วิ่งไปจุดที่ 2
        # ────────────────────────────────────────────────────────────────────
        elif self.mission_step == 1 and self.arrived():
            self.go_to(-0.50, 0.0, 0.0,
                       speed_limit=1.6,
                       pos_pid=[13.5, 0.00002, 6.5])
            self.next_step(2)

        # ────────────────────────────────────────────────────────────────────
        # STEP 2 — ถึงจุด 2 → ทำ task ต่อ (เพิ่มเองตรงนี้)
        # ────────────────────────────────────────────────────────────────────
        elif self.mission_step == 2 and self.arrived():
            # ตัวอย่าง: เปิด servo แล้วไปต่อ
            self.do_servo(self.servo_system.PullBox, 'on', delay_after=0.5)
            self.do_servo(self.servo_system.PullBox, 'off', delay_after=0.0)
            self.next_step(3)

        # ────────────────────────────────────────────────────────────────────
        # STEP 3 — คลานเข้าโต๊ะ (ใช้ Lidar)
        # ────────────────────────────────────────────────────────────────────
        elif self.mission_step == 3:
            if self.Lidar_center_dist is not None:
                if self.Lidar_center_dist <= 0.38:
                    self.set_manual_velocity(0.06, 0.0)
                else:
                    self.set_manual_velocity(0.10, 0.0)

                # ถึงระยะ → หยุดแล้วไป step ถัดไป
                if self.Lidar_center_dist <= 0.30:
                    self.set_manual_velocity(0.0, 0.0)
                    self.next_step(4)

        # ────────────────────────────────────────────────────────────────────
        # STEP 7.1 — Case 1 เก้าอี้
        # ────────────────────────────────────────────────────────────────────
        elif self.mission_step == 7.1:
            self.do_servo(self.servo_system.PullBox, 'on',  delay_after=0.5)
            self.do_servo(self.servo_system.PullBox, 'off', delay_after=0.0)
            self.slide_system.enable()
            self.slide_system.down()
            self.next_step(7.11)

        elif self.mission_step == 7.11:
            if self.wait_sensor(self.sensors.bottle_detected, expected=0):
                # ขวดมาแล้ว
                self.bottle_system.disable()
                self.bottle_system.out_pos()
                time.sleep(0.5)
                self.do_servo(self.servo_system.BlockBottle, 'on', delay_after=0.0)
                self.sleep_move(-0.25, 0.0, duration=0.75)
                self.do_servo(self.servo_system.BlockBottle, 'off', delay_after=0.0)
                self.next_step(8)
            else:
                # รอขวด
                self.bottle_system.enable()
                self.bottle_system.out_pos()

        # ────────────────────────────────────────────────────────────────────
        # STEP 7.2 — Case 2 เก้าอี้
        # ────────────────────────────────────────────────────────────────────
        elif self.mission_step == 7.2:
            if self.wait_sensor(self.sensors.box_detected, expected=0):
                self.do_servo(self.servo_system.PullBox, 'on',  delay_after=0.5)
                self.do_servo(self.servo_system.PullBox, 'off', delay_after=1.5)
                self.slide_system.enable()
                self.slide_system.up()
                self.next_step(7.21)

        elif self.mission_step == 7.21:
            if self.wait_sensor(self.sensors.box_detected, expected=0) \
                    and self.counterboxout >= 1:
                self.do_servo(self.servo_system.PullBox, 'on',  delay_after=0.5)
                self.do_servo(self.servo_system.PullBox, 'off', delay_after=0.0)
                self.get_logger().info('---> Case Ok')
                self.next_step(7.22)
            else:
                self.counterboxout += 1
                self.get_logger().info(f'---> Up {self.counterboxout}')

        elif self.mission_step == 7.22:
            if self.wait_sensor(self.sensors.bottle_detected, expected=0):
                self.bottle_system.disable()
                self.bottle_system.out_pos()
                time.sleep(0.5)
                self.do_servo(self.servo_system.BlockBottle, 'on', delay_after=0.0)
                self.next_step(7.23)
            else:
                self.bottle_system.enable()
                self.bottle_system.out_pos()

        elif self.mission_step == 7.23:
            if self.wait_sensor(self.sensors.table_l, expected=1):
                self.do_servo(self.servo_system.BlockBottle, 'off', delay_after=0.2)
                self.sleep_move(0.0, -0.15, duration=0.8)
                self.next_step(7.24)
            else:
                self.set_manual_velocity(0.0, 0.15)

        elif self.mission_step == 7.24:
            if self.wait_sensor(self.sensors.bottle_detected, expected=0):
                self.bottle_system.disable()
                self.bottle_system.out_pos()
                time.sleep(0.5)
                self.do_servo(self.servo_system.BlockBottle, 'on', delay_after=0.0)
                self.sleep_move(-0.15, 0.0, duration=0.85)
                self.do_servo(self.servo_system.BlockBottle, 'off', delay_after=0.0)
                self.next_step(8)
            else:
                self.bottle_system.enable()
                self.bottle_system.out_pos()

        # ────────────────────────────────────────────────────────────────────
        # STEP 8 → 9 → 10 → 11 — วิ่งกลับ
        # ────────────────────────────────────────────────────────────────────
        elif self.mission_step == 8:
            self.slide_system.enable()
            self.slide_system.down()
            self.go_to(-0.48, -0.45, 0.0,
                       speed_limit=1.4,
                       pos_pid=[6.5, 0.00002, 3.0])
            self.next_step(9)

        elif self.mission_step == 9 and self.arrived():
            self.next_step(10)

        elif self.mission_step == 10:
            self.go_to(-0.48, -0.95, 0.0,
                       speed_limit=1.4,
                       pos_pid=[6.5, 0.00002, 3.0])
            self.next_step(11)

        elif self.mission_step == 11 and self.arrived():
            self.slide_system.enable()
            self.slide_system.down()
            self.next_step(12)
        '''
        # ────────────────────────────────────────────────────────────────────
        # ส่งความเร็ว / actuator state
        # ────────────────────────────────────────────────────────────────────
        if self.control_mode == "MANUAL":
            msg = Twist()
            msg.linear.x = self.manual_vx
            msg.linear.y = self.manual_vy
            self.vel_pub.publish(msg)

        elif self.control_mode == "INTERNAL":
            cmd_vel, is_arrived = self.nav.calculate_velocity(
                self.curr_x, self.curr_y, self.curr_yaw)

            if is_arrived:
                self.control_mode = "MANUAL"
                self.manual_vx    = 0.0
                self.manual_vy    = 0.0
                self.vel_pub.publish(Twist())
                self.get_logger().info("🏁 INTERNAL arrived — stopped")
            else:
                self.vel_pub.publish(cmd_vel)

        self.publish_state()

    # ══════════════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ══════════════════════════════════════════════════════════════════════════


    def program_command_callback(self,msg):
        self.ProgramCommand = msg.data

    def program_color_callback(self, msg):
        self.Programcolor = msg.data

        if(self.Programcolor == self.ColorRed):
            self.send_parameters(0.15, 1.2, 0.5, 0.05, 180.0, 270.0)
            self.mission_step = 0
        elif(self.Programcolor == self.ColorBlue):
            self.send_parameters(0.15, 1.2, 0.5, 0.05, 270.0, 359.0)
            self.mission_step = 0

    def program_game_callback(self, msg):
        self.ProgramGame = msg.data

    def chaircount_callback(self, msg):
        self.ChairCount = msg.data

    def sensor_callback(self, msg):
        self.sensors.update(msg.data)

    def odom_callback(self, msg):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.curr_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def rviz_goal_callback(self, msg):
        q = msg.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.go_to(msg.pose.position.x, msg.pose.position.y,
                   math.degrees(yaw),
                   pos_pid=[12.5, 0.01, 1.2],
                   yaw_pid=[500.0, 0.0, 25.0])

    def filter_callback(self, msg):
        marker = Marker()
        marker.header = msg.header
        marker.type = Marker.SPHERE_LIST
        marker.scale.x = marker.scale.y = 0.05
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.g = 1.0

        for deg in [28, -28, 0]:
            rad = np.deg2rad(deg)
            idx = int((rad - msg.angle_min) / msg.angle_increment)
            if 0 <= idx < len(msg.ranges):
                dist = msg.ranges[idx]
                if np.isfinite(dist):
                    if deg ==  28: self.Lidar_left_dist   = dist
                    if deg == -28: self.Lidar_right_dist  = dist
                    if deg ==   0: self.Lidar_center_dist = dist
                    p = Point()
                    p.x = dist * np.cos(rad)
                    p.y = dist * np.sin(rad)
                    marker.points.append(p)
        self.marker_pub.publish(marker)

    # ══════════════════════════════════════════════════════════════════════════
    # ACTUATOR / MOTION HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def go_to(self, x, y, yaw_deg, mode="DIRECT_STM32",
              speed_limit=0.6, pos_pid=None, yaw_pid=None):
        yaw_rad = math.radians(yaw_deg)
        self.target_x   = x
        self.target_y   = y
        self.target_yaw = yaw_rad
        self.control_mode = mode

        if pos_pid and len(pos_pid) == 3:
            self.set_pid_gains(6, *pos_pid)
        if yaw_pid and len(yaw_pid) == 3:
            self.set_pid_gains(5, *yaw_pid)

        if mode == "DIRECT_STM32":
            spd = Float32()
            spd.data = float(speed_limit)
            self.max_speed_pub.publish(spd)

            goal = Pose2D()
            goal.x = float(x); goal.y = float(y); goal.theta = float(yaw_rad)
            self.goal_pub.publish(goal)
            self.get_logger().info(f'🚀 Go to ({x:.2f}, {y:.2f}) {yaw_deg}° @ {speed_limit}m/s')
        else:
            self.nav.set_goal(x, y, yaw_rad, self.curr_x, self.curr_y,
                              mode="DIRECT", cruise_speed=speed_limit)
    def go_to_curve(self, x, y, yaw_deg=None,
                    curve_side="AUTO",
                    speed_limit=0.4,
                    curve_strength=0.3):
        """
        วิ่งเป็นเส้นโค้ง Bezier ไปยัง (x, y) โดยหน้าหุ่นยังหันอยู่ทิศเดิม
        ใช้ Mecanum สไลด์ผ่านเส้นโค้ง ไม่หมุนตัว
 
        Parameters:
          x, y            : ตำแหน่งเป้าหมาย (เมตร)
          yaw_deg         : None = คงทิศเดิม, ใส่ค่า = หันหน้าเมื่อถึง
          curve_side      : "AUTO" | "LEFT" | "RIGHT"
                            AUTO = โค้งไปทางที่ใกล้กว่า
          speed_limit     : ความเร็ว (m/s)
          curve_strength  : ความโค้ง 0.0=เส้นตรง, 0.5=โค้งมาก (default 0.3)
 
        ตัวอย่าง:
          # วิ่งโค้งขวาไป (1.0, 0.0) หน้าหันอยู่กับที่
          self.go_to_curve(1.0, 0.0, curve_side="RIGHT", speed_limit=0.5)
 
          # วิ่งโค้งซ้ายไป (0.5, 0.5) โค้งมาก
          self.go_to_curve(0.5, 0.5, curve_side="LEFT",
                           speed_limit=0.4, curve_strength=0.5)
        """
        # yaw_deg=None → คงทิศปัจจุบัน
        if yaw_deg is None:
            yaw_deg = math.degrees(self.curr_yaw)
 
        self.target_x   = x
        self.target_y   = y
        self.target_yaw = math.radians(yaw_deg)
        self.control_mode = "INTERNAL"
 
        self.nav.set_bezier_goal(
            x, y,
            self.curr_x, self.curr_y, self.curr_yaw,
            curve_side=curve_side,
            cruise_speed=speed_limit,
            curve_strength=curve_strength,
        )
    def stop(self):
        self.control_mode = "MANUAL"
        self.manual_vx = self.manual_vy = 0.0
        self.vel_pub.publish(Twist())
        self.get_logger().info('🛑 Stopped')
    def set_manual_velocity(self, vx, vy):
        self.control_mode = "MANUAL"
        self.manual_vx = float(vx)
        self.manual_vy = float(vy)
    def set_pid_gains(self, pid_id, p, i, d):
        msg = Float32MultiArray()
        msg.data = [float(p), float(i), float(d)]
        if pid_id == 5:
            self.yaw_pid_pub.publish(msg)
        elif pid_id == 6:
            self.pos_pid_pub.publish(msg)
    def Griper_R(self,state):
        if(state == 'active'):
            self.ControlServo(0,self.ServoGriper_R[1])
        elif(state == 'unactive'):
            self.ControlServo(0,self.ServoGriper_R[0])
    def Griper_L(self,state):
        if(state == 'active'):
            self.ControlServo(2,self.ServoGriper_L[1])
        elif(state == 'unactive'):
            self.ControlServo(2,self.ServoGriper_L[0])

    def ControlBottle_L(self,state):
        if(state == 'up'):
            self.StateControlAutomation[0] = 1
        elif(state == 'down'):
            self.StateControlAutomation[0] = 0
        self.ControlAutomation()

    def publish_state(self):
        msg = Int32MultiArray()
        msg.data = self.current_state
        self.cmd_pub.publish(msg)
# ══════════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    ctrl = SudSakhonMainController()

    '''
    # ── Setup เริ่มต้น ─────────────────────────────────────────────────────
    ctrl.slide_system.enable()
    ctrl.slide_system.down()
    ctrl.servo_system.BlockBottle.off()
    ctrl.bucket_system.in_pos()
    ctrl.bottle_system.disable()
    ctrl.bottle_system.out_pos()
    ctrl.servo_system.PullBox.off()
    ctrl.set_servo(4, 90)

    '''

    try:
        rclpy.spin(ctrl)
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop()
        ctrl.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()