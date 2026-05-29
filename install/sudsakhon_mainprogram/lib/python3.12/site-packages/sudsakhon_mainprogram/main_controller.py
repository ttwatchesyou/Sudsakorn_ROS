import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Pose2D
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32MultiArray, Int8MultiArray, Float32, Float32MultiArray
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from collections import Counter

import math
import time
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Int32
from scipy.interpolate import CubicSpline

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
        self.mission_step_pub = self.create_publisher(Float32, '/current_mission_step', 10)


        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Int32MultiArray, '/automation/sensors',  self.sensor_callback,   10)
        self.create_subscription(Odometry,       '/odom',          self.odom_callback,     10)
        self.create_subscription(PoseStamped,    '/goal_pose',     self.rviz_goal_callback,10)
        self.create_subscription(Int32,          '/chair_count',   self.chaircount_callback,10)
        self.create_subscription(LaserScan,      'scan',           self.filter_callback,   10)
        self.create_subscription(Int32,          '/Program/Color',   self.program_color_callback,10)
        self.create_subscription(Int32,          '/Program/Game',   self.program_game_callback,10)
        self.create_subscription(Int32,          '/Program/Command',   self.program_command_callback,10)
        self.create_subscription(String,          '/detected_objects',   self.object_callback,10)

        # ── State ─────────────────────────────────────────────────────────────
        self.ServoGriper_R = [110,55,80]
        self.ServoGriper_L = [110,60,90]
        self.ServoBoxPusher = [90,30]

        self.StateControlAutomation = [0,0,0,0]
        self.detected_list = []

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
        self.DetectedObjects = ''

        # ── Systems ───────────────────────────────────────────────────────────
        self.slide_system  = SlideSystem(self)
        self.bottle_system = BottleSystem(self)
        self.bucket_system = BucketSystem(self)
        self.servo_system  = ServoSystem(self)
        self.sensors       = SensorSystem()
        self.nav           = NavigationSystem(self)

        self.game_seleclted = ""
        
        self.Chair_params_RED    = [0.15, 1.2, 0.5, 0.05, 180.0, 270.0]
        self.Chair_params_BLUE   = [0.15, 1.2, 0.5, 0.05, 270.0, 90.0]

        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info('✅ SudSakhon Controller Ready')

        self.send_parameters(0.15, 1.2, 0.5, 0.05, 180.0, 270.0)

    def send_parameters(self, min_w, max_w, threshold, gap, min_deg, max_deg):
        msg = Float32MultiArray()
        msg.data = [float(min_w), float(max_w), float(threshold), float(gap), float(min_deg), float(max_deg)]
        self.chair_params.publish(msg)

    def ControlServo(self,Number,Control):
        msg = Int32MultiArray()
        msg.data = [Number,Control]
        self.control_servo.publish(msg)

    def ControlAutomation(self):
        msg = Int32MultiArray()
        msg.data = self.StateControlAutomation
        self.control_states.publish(msg)

    def step(self, current, expected):
        return self.mission_step == expected or self.mission_step == current

    def next_step(self, step_num):
        self.get_logger().info(f'➡️  Step {self.mission_step} → {step_num}')
        self.mission_step = step_num

    def arrived(self, tol=None):
        if self.control_mode == "INTERNAL":
            return self.nav.arrived

        elif self.control_mode == "SPIN":
            yaw_err = self.target_yaw - self.curr_yaw
            while yaw_err > math.pi: yaw_err -= 2.0 * math.pi
            while yaw_err < -math.pi: yaw_err += 2.0 * math.pi
            return abs(yaw_err) < self.yaw_tolerance

        t    = tol if tol else self.dist_tolerance
        dist = math.sqrt((self.target_x - self.curr_x)**2 + (self.target_y - self.curr_y)**2)
        yaw_err = abs(self.target_yaw - self.curr_yaw)

        while yaw_err > math.pi:
            yaw_err = abs(yaw_err - 2 * math.pi)

        out = dist < t and yaw_err < self.yaw_tolerance
        return out

    def lidar_approach(self, lidar_dist, direction='x+', stop_dist=0.30, slow_dist=0.50, cruise_speed=0.20, slow_speed=0.06, lateral_speed=0.06, lateral_detect_dist=0.80, align_kp=1.5):
        if lidar_dist is None:
            self.set_manual_velocity(0.0, 0.0)
            return False

        dir_map = {'x+': (( 1, 0), ( 0, 1)), 'x-': ((-1, 0), ( 0,-1)), 'y+': (( 0, 1), (-1, 0)), 'y-': (( 0,-1), ( 1, 0))}
        (fdx, fdy), (ldx, ldy) = dir_map.get(direction, (( 1, 0), ( 0, 1)))

        if lidar_dist <= stop_dist:
            self.set_manual_velocity(0.0, 0.0)
            return True

        if lidar_dist <= slow_dist:
            ratio   = (lidar_dist - stop_dist) / (slow_dist - stop_dist)
            ratio   = max(0.0, min(1.0, ratio))
            fwd_spd = slow_speed + (cruise_speed - slow_speed) * ratio
        else:
            fwd_spd = cruise_speed

        lat_spd = 0.0
        angular = 0.0
        L = self.Lidar_left_dist
        R = self.Lidar_right_dist

        left_missing  = (L is None) or (L > lateral_detect_dist)
        right_missing = (R is None) or (R > lateral_detect_dist)

        if not left_missing and not right_missing:
            diff    = L - R
            angular = -align_kp * diff
            angular = max(min(angular, 0.5), -0.5)
        elif right_missing and not left_missing: lat_spd = +lateral_speed
        elif left_missing and not right_missing: lat_spd = -lateral_speed

        vx = fdx * fwd_spd + ldx * lat_spd
        vy = fdy * fwd_spd + ldy * lat_spd

        msg = Twist()
        msg.linear.x  = float(vx)
        msg.linear.y  = float(vy)
        msg.angular.z = float(angular)
        self.control_mode = "MANUAL"
        self.manual_vx = float(vx)
        self.manual_vy = float(vy)
        self.vel_pub.publish(msg)
        return False

    def publish_wheelcontrol(self):
        if self.control_mode == "MANUAL":
            msg = Twist()
            msg.linear.x = self.manual_vx
            msg.linear.y = self.manual_vy
            self.vel_pub.publish(msg)

        elif self.control_mode == "SPIN":
            yaw_err = self.target_yaw - self.curr_yaw
            while yaw_err > math.pi: yaw_err -= 2.0 * math.pi
            while yaw_err < -math.pi: yaw_err += 2.0 * math.pi
            
            if abs(yaw_err) < self.yaw_tolerance:
                self.control_mode = "MANUAL"
                self.manual_vx = 0.0
                self.manual_vy = 0.0
                self.vel_pub.publish(Twist())
            else:
                msg = Twist()
                v_yaw = 2.5 * yaw_err 
                v_yaw = max(min(v_yaw, 2.0), -2.0) 
                msg.angular.z = float(v_yaw)
                self.vel_pub.publish(msg)
        
        elif self.control_mode == "SPLINE_TRACKING":
            self.update_spline_tracking()

        elif self.control_mode == "INTERNAL":
            cmd_vel, is_arrived = self.nav.calculate_velocity(self.curr_x, self.curr_y, self.curr_yaw)
            if is_arrived:
                self.control_mode = "MANUAL"
                self.manual_vx    = 0.0
                self.manual_vy    = 0.0
                self.vel_pub.publish(Twist())
            else:
                self.vel_pub.publish(cmd_vel)

        self.publish_state()

    def control_loop(self):
        step_msg = Float32()
        step_msg.data = float(self.mission_step)
        self.mission_step_pub.publish(step_msg)

        yaw_pid_Set=[145.0, 0.0, 200.0]

        if( self.sensors.SW_2 == 0 ):
            self.Griper_R('hold')
            self.Griper_L('hold')
            self.Box_Pusher('unactive')
            self.ControlBottle_R('down')
            self.ControlBottle_L('down')
            self.ControlBox('down')

            self.go_to(-0.0, -4.0, 0.0, mode="DIRECT_STM32", speed_limit=1.0, pos_pid=[1.25, 0.0, 0.0], yaw_pid=[125.0, 0.0, 200.0])

            #self.go_to_curve(-1.0, -1.0,yaw_deg=270, speed_limit=1.30, curve_strength=0.4, curve_side='AUTO',curve_kp_=1.5)
            #self.go_to(-0.0, -0.0, 0.0, mode="DIRECT_STM32", speed_limit=1.0, pos_pid=[1.25, 0.0, 0.0], yaw_pid=[125.0, 0.0, 200.0])

            #self.go_to(-0.0, -0.0, 0.0, mode="DIRECT_STM32", speed_limit=1.0, pos_pid=[1.25, 0.0, 0.0], yaw_pid=[125.0, 0.0, 200.0])
            #self.go_to(-0.0, -0.0, 0.0,speed_limit=1.0,pos_pid=[1.25, 0.0, 0.0])

        if( self.sensors.SW_1 == 0 ):


            self.go_to(-0.0, -0.0, 0.0, mode="DIRECT_STM32", speed_limit=1.0, pos_pid=[1.25, 0.0, 0.0], yaw_pid=[125.0, 0.0, 200.0])
            pass
            #self.go_to(-0.0, -0.0, 180.0, mode="DIRECT_STM32", speed_limit=1.0, pos_pid=[1.25, 0.0, 0.0], yaw_pid=[125.0, 0.0, 200.0])
            #self.go_to(-0.0, -0.0, 180.0,speed_limit=1.0,pos_pid=[1.25, 0.0, 0.0])

        if self.game_seleclted == "":
            if self.Programcolor == self.ColorRed:
                if self.ProgramGame == 1:
                    self.game_seleclted = "GameRed_1"

        if self.game_seleclted == "GameRed_1":
            if self.mission_step == 0:
                if self.sensors.SW_1 == 0 or self.ProgramCommand == self.ProgramCommand_Start:
                    self.Griper_R('hold')
                    self.Griper_L('hold')
                    self.Box_Pusher('unactive')
                    self.ControlBottle_R('down')
                    self.ControlBottle_L('down')
                    self.ControlBox('down')
                    self.ControlSlide('in')
                    self.set_pid_gains(5, 0.05, 0.0, 0.5)
                    self.get_logger().info('🔘 เริ่มภารกิจ!')
                    self.ChairCount_RUN = self.ChairCount
                    self.next_step(1)

            elif self.mission_step == 1 and self.arrived():
                self.go_to_curve(-2.30, -1.75, speed_limit=1.30, curve_strength=0.4, curve_side='AUTO',curve_kp_=1.5)
                if self.ChairCount_RUN == 1:
                    self.ControlBottle_R('down')
                    self.ControlBottle_L('up')
                    
                else:
                    self.ControlBottle_R('up')
                    self.ControlBottle_L('up')
                self.next_step(2)

            elif self.mission_step == 2 and self.arrived():
                self.ControlBox('up')

                if self.ChairCount_RUN == 1:
                    if self.sensors.SensorbottleL_B_UP == 0 or self.sensors.SensorbottleL_Check == 0:
                        self.Griper_R('hold')
                        self.Griper_L('active')
                        self.Box_Pusher('unactive')
                        self.next_step(2.5)
                else:
                    

                    if (self.sensors.SensorbottleL_B_UP == 0 or self.sensors.SensorbottleL_Check == 0) and (self.sensors.SensorbottleR_B_UP == 0 or self.sensors.SensorbottleR_Check == 0):
                        self.Griper_R('active')
                        self.Griper_L('active')
                        self.Box_Pusher('active')
                        
                        self.next_step(2.5)
                
                    

            elif self.mission_step == 2.5: 
                if self.lidar_approach(self.Lidar_center_dist, direction='x+', stop_dist=0.25, slow_dist=0.30, cruise_speed=0.20, slow_speed=0.06):
                    
                    self.next_step(2.6)
                    
            elif self.mission_step == 2.6: 
                
                

                # time.sleep(0.2)
                self.ControlBottle_R('stop')
                self.ControlBottle_L('stop')
                time.sleep(0.15)
                self.ControlSlide('out')
                time.sleep(0.8)
                self.Box_Pusher('unactive')
                if self.ChairCount_RUN == 1:
                    self.Griper_R('hold')
                    self.Griper_L('unactive')
                else:
                    self.Griper_R('unactive')
                    self.Griper_L('unactive')
                time.sleep(0.5)

                self.ControlSlide('in')
                time.sleep(0.3)
                self.next_step(3)

            elif self.mission_step == 3:
                self.go_to(-2.25, -4.0, 0.0,speed_limit=1.0,pos_pid=[1.10, 0.0, 0.0], yaw_pid=yaw_pid_Set)
                self.Griper_R('hold')
                self.Griper_L('hold')
                self.next_step(4)

            elif self.mission_step == 4 and self.arrived():
                if self.ChairCount_RUN == 1:
                    self.ControlBottle_R('up')
                    self.ControlBottle_L('up')
                else:
                    self.ControlBottle_R('down')
                    self.ControlBottle_L('up')

                self.ControlBox('up')
                self.next_step(4.5) 

            elif self.mission_step == 4.5: 
                if self.lidar_approach(self.Lidar_center_dist, direction='x+', stop_dist=0.21, slow_dist=0.30, cruise_speed=0.20, slow_speed=0.06):
                    self.next_step(4.6)
            
            elif self.mission_step == 4.6: 
                #self.ControlBox('up')
                if self.ChairCount_RUN == 1:
                    if (self.sensors.SensorbottleL_B_UP == 0 or self.sensors.SensorbottleL_Check == 0) and (self.sensors.SensorbottleR_B_UP == 0 or self.sensors.SensorbottleR_Check == 0):
                        self.Griper_R('active')
                        self.Griper_L('active')
                        self.Box_Pusher('active')

                        self.next_step(4.7)
                else:
                    if (self.sensors.SensorbottleL_B_UP == 0 or self.sensors.SensorbottleL_Check == 0):
                        self.Griper_R('hold')
                        self.Griper_L('active')
                        self.Box_Pusher('unactive')

                        self.next_step(4.7)

            elif self.mission_step == 4.7 :
                #time.sleep(0.2)
                self.ControlBottle_R('stop')
                self.ControlBottle_L('stop')
                time.sleep(0.15)
                self.ControlSlide('out')
                time.sleep(0.8)
                self.Box_Pusher('unactive')
                if self.ChairCount_RUN == 1:
                    self.Griper_R('unactive')
                    self.Griper_L('unactive')
                else:
                    self.Griper_R('unactive')
                    self.Griper_L('unactive')
                time.sleep(0.5)
                self.ControlSlide('in')
                time.sleep(0.3)
                self.next_step(5)


            elif self.mission_step == 5 :
                self.go_to(-2.25, -4.0, 0, speed_limit=1.0, pos_pid=[1.25, 0.0, 0.0], yaw_pid=yaw_pid_Set)
                self.next_step(5.5)

            elif self.mission_step == 5.5 :
                if self.arrived():
                    self.next_step(6)






            # 🛑 แก้ไขจุดนี้: ยกเลิก SPIN และใช้เทคนิค Shimmy แทน
            elif self.mission_step == 6:
                self.get_logger().info('➡️ [Step 6] ใช้เทคนิค Shimmy: ถอยออกมา 15cm พร้อมบังคับหมุน 180 องศา เพื่อปลดล็อค STM32')
                # ถอย Y ออกมานิดนึงเป็น -3.85 (จากเดิม -4.0) เพื่อให้ระยะ X,Y ไม่เป็นศูนย์ 
                # (สำคัญ: สังเกตว่าผมเพิ่ม yaw_pid เข้าไปด้วยเพื่อให้มั่นใจว่าบอร์ดมีแรงหมุน)
                self.go_to(-2.25, -4.0, 180.0, mode="DIRECT_STM32", speed_limit=1.0, pos_pid=[1.25, 0.0, 0.0], yaw_pid=[110.0, 0.0, 0.2])
                self.next_step(6.4) 

            elif self.mission_step == 6.4:
                if self.arrived():
                    self.get_logger().info('➡️ [Step 6.4] หมุนเสร็จแล้ว: วิ่งกลับไปเสียบจุดเดิมที่ -4.0')
                    self.detected_list = []
                    # กลับไปที่จุดเป้าหมายเดิมที่ -4.0 วางกล่อง
                    #self.go_to(-2.25, -4.0, 180.0, mode="DIRECT_STM32", speed_limit=1.0, pos_pid=[1.25, 0.0, 0.0], yaw_pid=[500.0, 0.0, 25.0])
                    self.next_step(6.5)

            elif self.mission_step == 6.5: 
                print(f"DetectedObjects: {self.DetectedObjects.split(',')[0]}") # ปรินต์เช็คค่าได้ปกติ
                # เอาข้อความที่ได้เก็บลง list
                self.detected_list.append(self.DetectedObjects.split(',')[0] )
                final_result = ''

                self.Griper_R('hold')
                self.Griper_L('hold')

                self.ControlBottle_R('up')
                self.ControlBottle_L('stop')

                #
                
                if self.lidar_approach(self.Lidar_center_dist,
                                    direction='x+',
                                    stop_dist=0.25,
                                    slow_dist=0.30,
                                    cruise_speed=0.20,
                                    slow_speed=0.06):

                    if len(self.detected_list) > 0:
                        total_count = len(self.detected_list)
                        threshold = total_count * 0.20  # 20% ของข้อมูลทั้งหมด
                        
                        counted_data = Counter(self.detected_list)
                        
                        # ตัดค่าว่างออกจากตัวนับเพื่อหาข้อความ (Object) ที่เยอะที่สุด
                        if '' in counted_data:
                            del counted_data['']
                        
                        if len(counted_data) > 0:
                            most_frequent_text = counted_data.most_common(1)[0][0]
                            max_count = counted_data.most_common(1)[0][1]
                            
                            # ถ้าข้อความที่เจอเยอะที่สุด มีจำนวนมากกว่า 20% ให้เปลี่ยนผลลัพธ์เป็นข้อความนั้น
                            if max_count > threshold:
                                final_result = most_frequent_text

                    # ถ้าไม่เข้าเงื่อนไข (เช่น มีแต่ค่าว่าง หรือข้อความไม่ถึง 20%) final_result ก็จะเป็น '' เหมือนเดิม
                    print(f"ผลลัพธ์ที่จะนำไปใช้ต่อคือ: '{final_result}'")

                    if final_result == 'bottle':
                        self.next_step(7.1) #ไป ลูป สั่งกล่องออก
                    elif final_result == 'box':
                        self.next_step(7.2) #ไป ลูป สั่งขวดออก
                    else:
                        self.next_step(7.2) #ไป ลูป สั่งขวดออก
            elif self.mission_step == 7.2:
                
                if (self.sensors.SensorbottleR_B_UP == 0 or self.sensors.SensorbottleR_Check == 0):
                    self.Griper_R('active')
                    self.Griper_L('hold')
                    self.next_step(7.21)
                
            elif self.mission_step == 7.21:
                #time.sleep(0.5)
                self.ControlBottle_R('stop')
                self.ControlBottle_L('stop')
                time.sleep(0.1)
                self.ControlSlide('out')
                time.sleep(0.8)
                self.Griper_R('unactive')
                self.Griper_L('hold')
                time.sleep(0.5)
                self.ControlSlide('in')
                time.sleep(0.3)

                self.next_step(8)

            elif self.mission_step == 8:
                self.go_to(-2.35, -1.85, 180,
                        speed_limit=1.0,
                        pos_pid=[1.10, 0.0, 0.0], yaw_pid=yaw_pid_Set)

                
                self.next_step(9)

            elif self.mission_step == 9:
                if self.arrived():
                    self.ControlBottle_R('stop')
                    self.ControlBottle_L('up')
                    self.detected_list = []
                    # กลับไปที่จุดเป้าหมายเดิมที่ -4.0 วางกล่อง
                    #self.go_to(-2.25, -4.0, 180.0, mode="DIRECT_STM32", speed_limit=1.0, pos_pid=[1.25, 0.0, 0.0], yaw_pid=[500.0, 0.0, 25.0])
                    self.next_step(9.5)

            elif self.mission_step == 9.5: 
                print(f"DetectedObjects: {self.DetectedObjects.split(',')[0]}") # ปรินต์เช็คค่าได้ปกติ
                # เอาข้อความที่ได้เก็บลง list
                self.detected_list.append(self.DetectedObjects.split(',')[0] )
                final_result = ''

                

                self.Griper_R('hold')
                self.Griper_L('hold')
                
                if self.lidar_approach(self.Lidar_center_dist,
                                    direction='x+',
                                    stop_dist=0.25,
                                    slow_dist=0.30,
                                    cruise_speed=0.20,
                                    slow_speed=0.06):

                    if len(self.detected_list) > 0:
                        total_count = len(self.detected_list)
                        threshold = total_count * 0.20  # 20% ของข้อมูลทั้งหมด
                        
                        counted_data = Counter(self.detected_list)
                        
                        # ตัดค่าว่างออกจากตัวนับเพื่อหาข้อความ (Object) ที่เยอะที่สุด
                        if '' in counted_data:
                            del counted_data['']
                        
                        if len(counted_data) > 0:
                            most_frequent_text = counted_data.most_common(1)[0][0]
                            max_count = counted_data.most_common(1)[0][1]
                            
                            # ถ้าข้อความที่เจอเยอะที่สุด มีจำนวนมากกว่า 20% ให้เปลี่ยนผลลัพธ์เป็นข้อความนั้น
                            if max_count > threshold:
                                final_result = most_frequent_text

                    # ถ้าไม่เข้าเงื่อนไข (เช่น มีแต่ค่าว่าง หรือข้อความไม่ถึง 20%) final_result ก็จะเป็น '' เหมือนเดิม
                    print(f"ผลลัพธ์ที่จะนำไปใช้ต่อคือ: '{final_result}'")

                    if final_result == 'bottle':
                        self.next_step(10.1) #ไป ลูป สั่งกล่องออก
                    elif final_result == 'box':
                        self.next_step(10.2) #ไป ลูป สั่งขวดออก
                    else:
                        self.next_step(10.2) #ไป ลูป สั่งขวดออก

            elif self.mission_step == 10.2:
                if (self.sensors.SensorbottleL_B_UP == 0 or self.sensors.SensorbottleL_Check == 0):
                    self.Griper_R('hold')
                    self.Griper_L('active')
                    self.next_step(10.21)

            elif self.mission_step == 10.21:
                #self.Griper_R('hold')
                #self.Griper_L('active')
                #time.sleep(0.5)
                self.ControlBottle_R('stop')
                self.ControlBottle_L('stop')
                time.sleep(0.1)
                self.ControlSlide('out')
                time.sleep(0.8)
                self.Griper_R('hold')
                self.Griper_L('unactive')
                time.sleep(0.5)
                self.ControlSlide('in')
                time.sleep(0.3)
                self.next_step(11)

            elif self.mission_step == 11:
                self.go_to(-2.3, -5.0, 180, speed_limit=1.0, pos_pid=[1.25, 0.0, 0.0], yaw_pid=yaw_pid_Set)
                self.next_step(11.1)

            elif self.mission_step == 11.1 and self.arrived(tol=0.3):
                self.mission_step = 11.2
            elif self.mission_step == 11.2:
                self.go_to_curve(-0.3, -8.1, speed_limit=1.30, curve_strength=0.53, curve_side='AUTO',curve_kp_=1.65)
                self.mission_step = 11.3

            elif self.mission_step == 11.3 and self.arrived(tol=0.3):
                self.go_to(-1.3, -8.2, 180, speed_limit=1.0, pos_pid=[1.25, 0.0, 0.0], yaw_pid=yaw_pid_Set)
                self.next_step(11.4)
            elif self.mission_step == 11.4 and self.arrived(tol=0.1):
                self.next_step(11.41)
                
            elif self.mission_step == 11.41:   
                if self.lidar_approach(self.Lidar_center_dist, direction='x+', stop_dist=0.40, slow_dist=0.30, cruise_speed=0.20, slow_speed=0.06):
                    print(f"เชคขอบไม้ลิฟ ดีเลย์ 1 วินาที") 
                    self.next_step(11.42)

            

            elif self.mission_step == 11.42 :

                print(self.Lidar_center_dist)
                print(f"self.Lidar_center_dist: '{self.Lidar_center_dist}'")

                if self.Lidar_center_dist >= 1:
                    print(f"เชคขอบไม้ลิฟ ดีเลย์ 1 วินาที")
                    time.sleep(1.0)
                    self.next_step(12)
            
            elif self.mission_step == 12:

                target_x = (self.curr_x) + (-0.8)
                target_y = self.curr_y

                #self.go_to(-2.375, -8.2, 180, speed_limit=0.6, pos_pid=[1.7, 0.0, 0.0], yaw_pid=yaw_pid_Set)
                self.go_to(target_x, target_y, 180, speed_limit=0.6, pos_pid=[1.7, 0.0, 0.0], yaw_pid=yaw_pid_Set)
                self.next_step(12.1)

            

            elif self.mission_step == 12.1 and self.arrived():

                target_x = (self.curr_x)
                target_y = (self.curr_y) + (+0.8)

                self.go_to(target_x, target_y, 180, speed_limit=0.6, pos_pid=[1.7, 0.0, 0.0], yaw_pid=yaw_pid_Set)
                #self.go_to(-2.375, -7.30, 180, speed_limit=0.6, pos_pid=[1.7, 0.0, 0.0], yaw_pid=yaw_pid_Set)
                self.detected_list = []
                self.next_step(12.2)

            
            
            elif self.mission_step == 12.2 and self.arrived(): 
               self.next_step(12.3)
            
            elif self.mission_step == 12.3:
                print(f"DetectedObjects: {self.DetectedObjects.split(',')[0]}") # ปรินต์เช็คค่าได้ปกติ
                # เอาข้อความที่ได้เก็บลง list
                self.detected_list.append(self.DetectedObjects.split(',')[0] )
                final_result = ''

                self.Griper_R('hold')
                self.Griper_L('hold')

                self.ControlBottle_R('up')
                self.ControlBottle_L('stop')
                
                if self.lidar_approach(self.Lidar_center_dist,
                                    direction='x+',
                                    stop_dist=0.25,
                                    slow_dist=0.30,
                                    cruise_speed=0.20,
                                    slow_speed=0.06):

                    if len(self.detected_list) > 0:
                        total_count = len(self.detected_list)
                        threshold = total_count * 0.20  # 20% ของข้อมูลทั้งหมด
                        
                        counted_data = Counter(self.detected_list)
                        
                        # ตัดค่าว่างออกจากตัวนับเพื่อหาข้อความ (Object) ที่เยอะที่สุด
                        if '' in counted_data:
                            del counted_data['']
                        
                        if len(counted_data) > 0:
                            most_frequent_text = counted_data.most_common(1)[0][0]
                            max_count = counted_data.most_common(1)[0][1]
                            
                            # ถ้าข้อความที่เจอเยอะที่สุด มีจำนวนมากกว่า 20% ให้เปลี่ยนผลลัพธ์เป็นข้อความนั้น
                            if max_count > threshold:
                                final_result = most_frequent_text

                    # ถ้าไม่เข้าเงื่อนไข (เช่น มีแต่ค่าว่าง หรือข้อความไม่ถึง 20%) final_result ก็จะเป็น '' เหมือนเดิม
                    print(f"ผลลัพธ์ที่จะนำไปใช้ต่อคือ: '{final_result}'")

                    if final_result == 'bottle':
                        self.next_step(13.1) #ไป ลูป สั่งกล่องออก
                    elif final_result == 'box':
                        self.next_step(13.2) #ไป ลูป สั่งขวดออก
                    else:
                        self.next_step(13.2) #ไป ลูป สั่งขวดออก


            elif self.mission_step == 13.2:
                if (self.sensors.SensorbottleR_B_UP == 0 or self.sensors.SensorbottleR_Check == 0):
                    self.Griper_R('active')
                    self.Griper_L('hold')
                    self.next_step(13.21)

            elif self.mission_step == 13.21:
                #self.Griper_R('hold')
                #self.Griper_L('active')
                #time.sleep(0.5)
                self.ControlBottle_R('stop')
                self.ControlBottle_L('stop')
                time.sleep(0.1)
                self.ControlSlide('out')
                time.sleep(0.8)
                self.Griper_R('unactive')
                self.Griper_L('hold')
                time.sleep(0.5)
                self.ControlSlide('in')
                time.sleep(0.3)
                self.next_step(14)
            
            elif self.mission_step == 14:
                target_x = (self.curr_x) + (+0.1)
                target_y = (self.curr_y) + (-0.8)
                self.go_to(target_x, target_y, 180, speed_limit=0.6, pos_pid=[1.7, 0.0, 0.0], yaw_pid=yaw_pid_Set)
                #self.go_to(-2.40, -8.2, 180, speed_limit=0.8, pos_pid=[1.25, 0.0, 0.0], yaw_pid=yaw_pid_Set)
                self.next_step(14.1)

            elif self.mission_step == 14.1 and self.arrived():

                target_x = (self.curr_x) + (+1.0)
                target_y = self.curr_y
                self.go_to(target_x, target_y, 180, speed_limit=0.6, pos_pid=[1.7, 0.0, 0.0], yaw_pid=yaw_pid_Set)
                #self.go_to(-1.4, -8.2, 180, speed_limit=0.8, pos_pid=[1.25, 0.0, 0.0], yaw_pid=yaw_pid_Set)
                self.next_step(14.2)
            
            elif self.mission_step == 14.2 and self.arrived():
                print(self.Lidar_center_dist)
                print(f"self.Lidar_center_dist: '{self.Lidar_center_dist}'")

                if self.Lidar_center_dist <= 1:
                    print(f"เชคขอบไม้ลิฟ ดีเลย์ 1 วินาที")
                    time.sleep(1.0)
                    self.next_step(15)
                #self.go_to(-1.4, -8.2, 180, speed_limit=1.0, pos_pid=[1.25, 0.0, 0.0], yaw_pid=yaw_pid_Set)
                #self.next_step(14.3)
            elif self.mission_step == 15:
                self.go_to(-0.2, -8.2, 270, speed_limit=1.0, pos_pid=[1.25, 0.0, 0.0], yaw_pid=yaw_pid_Set)
                self.next_step(15.1)

            elif self.mission_step == 15.1 and self.arrived():
                self.go_to(-0.1, -0.5, 270, speed_limit=1.35, pos_pid=[1.35, 0.0, 0.0], yaw_pid=[145.0, 0.0, 200.0])
                self.next_step(15.2)

            elif self.mission_step == 15.2 and self.arrived():   
                self.next_step(15.3)
                
            elif self.mission_step == 15.3: 
                '''
                 print(self.Lidar_back_dist)
                 if self.lidar_approach(self.Lidar_back_dist,
                        direction='x-',
                        stop_dist=0.35,
                        slow_dist=0.40,
                        cruise_speed=0.20,
                        slow_speed=0.06):
                    self.next_step(15.4)
                '''
            elif self.mission_step == 15.4:
                print(f"End Program......!")
                self.next_step(15.5)
            
            
                #self.next_step(11.5)
            
            # คำสั่งนี้อยู่นอกสุดของ if-elif block เพื่อให้อัปเดตมอเตอร์ตลอดเวลา
            
        self.publish_wheelcontrol()

    # ══════════════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ════════════════════════════════════════════════════════════════════════════

    def object_callback(self, msg):
        self.DetectedObjects = msg.data

    def program_command_callback(self,msg): self.ProgramCommand = msg.data
    def program_color_callback(self, msg):
        self.Programcolor = msg.data
        if(self.Programcolor == self.ColorRed):
            self.send_parameters(0.15, 1.2, 0.5, 0.05, 180.0, 270.0)
            self.mission_step = 0
        elif(self.Programcolor == self.ColorBlue):
            self.send_parameters(0.15, 1.2, 0.5, 0.05, 270.0, 359.0)
            self.mission_step = 0
    def program_game_callback(self, msg): self.ProgramGame = msg.data
    def chaircount_callback(self, msg): self.ChairCount = msg.data
    def sensor_callback(self, msg): self.sensors.update(msg.data)
    
    def odom_callback(self, msg):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.curr_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def rviz_goal_callback(self, msg):
        q = msg.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.go_to(msg.pose.position.x, msg.pose.position.y, math.degrees(yaw), pos_pid=[12.5, 0.01, 1.2], yaw_pid=[500.0, 0.0, 25.0])

    def filter_callback(self, msg):
        marker = Marker()
        marker.header = msg.header
        marker.type = Marker.SPHERE_LIST
        marker.scale.x = marker.scale.y = 0.05
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.g = 1.0
        for deg in [28, -28, 0, 180]:
            rad = np.deg2rad(deg)
            idx = int((rad - msg.angle_min) / msg.angle_increment)
            if 0 <= idx < len(msg.ranges):
                dist = msg.ranges[idx]
                if np.isfinite(dist):
                    if deg ==  28: self.Lidar_left_dist   = dist
                    if deg == -28: self.Lidar_right_dist  = dist
                    if deg ==   0: self.Lidar_center_dist = dist
                    if deg == 180: self.Lidar_back_dist   = dist
                    p = Point()
                    p.x = dist * np.cos(rad)
                    p.y = dist * np.sin(rad)
                    marker.points.append(p)
        self.marker_pub.publish(marker)

    # ══════════════════════════════════════════════════════════════════════════
    # ACTUATOR / MOTION HELPERS
    # ══════════════════════════════════════════════════════════════════════════
    def spin_to(self, yaw_deg):
        """ 🌪️ โหมดสั่งหมุนตัวอยู่กับที่โดยเฉพาะ """
        self.target_yaw = math.radians(yaw_deg)
        self.control_mode = "SPIN"
        
        spd = Float32()
        spd.data = 0.0
        self.max_speed_pub.publish(spd)

    def go_to(self, x, y, yaw_deg, mode="DIRECT_STM32", speed_limit=0.6, pos_pid=None, yaw_pid=None):
        yaw_rad = math.radians(yaw_deg)
        self.target_x   = x
        self.target_y   = y
        self.target_yaw = yaw_rad
        self.control_mode = mode

        if pos_pid and len(pos_pid) == 3: self.set_pid_gains(6, *pos_pid)
        if yaw_pid and len(yaw_pid) == 3: self.set_pid_gains(5, *yaw_pid)

        if mode == "DIRECT_STM32":
            spd = Float32()
            spd.data = float(speed_limit)
            self.max_speed_pub.publish(spd)

            goal = Pose2D()
            goal.x = float(x)
            goal.y = float(y)
            goal.theta = float(yaw_rad)
            self.goal_pub.publish(goal)
            self.get_logger().info(f'🚀 [DIRECT] Go to ({x:.2f}, {y:.2f}) {yaw_deg}° @ {speed_limit}m/s')
        else:
            if hasattr(self, 'nav') and hasattr(self.nav, 'arrived'):
                self.nav.arrived = False 
            self.nav.set_goal(x, y, yaw_rad, self.curr_x, self.curr_y, mode="DIRECT", cruise_speed=speed_limit)

    def go_to_curve(self, x, y, yaw_deg=None, curve_side="AUTO", speed_limit=0.4, curve_strength=0.3, curve_kp_=1.5):
        if yaw_deg is None: yaw_deg = math.degrees(self.curr_yaw)
        self.target_x   = x
        self.target_y   = y
        self.target_yaw = math.radians(yaw_deg)
        self.control_mode = "INTERNAL"
    
        self.get_logger().info(f"🚀 [CURVE] Go to ({x:.2f}, {y:.2f}) {yaw_deg}° @ {speed_limit}m/s")
        #self.nav.set_bezier_goal(x, y, self.curr_x, self.curr_y, self.curr_yaw, curve_side=curve_side, cruise_speed=speed_limit, curve_strength=curve_strength, curve_kp = curve_kp_)
        self.nav.set_bezier_goal_new(x, y, self.target_yaw, self.curr_x, self.curr_y, self.curr_yaw, curve_side=curve_side, cruise_speed=speed_limit, curve_strength=curve_strength, curve_kp = curve_kp_)

    def go_to_spline(self, waypoints, yaw_deg=None, speed_limit=1.0, pos_pid=None):
        if yaw_deg is None: yaw_deg = math.degrees(self.curr_yaw)
        pts = np.array(waypoints)
        t = np.linspace(0, 1, len(pts))
        cs_x = CubicSpline(t, pts[:, 0])
        cs_y = CubicSpline(t, pts[:, 1])
        t_smooth = np.linspace(0, 1, 50)
        self.spline_path_x = cs_x(t_smooth)
        self.spline_path_y = cs_y(t_smooth)
        self.spline_target_yaw = math.radians(yaw_deg)
        self.spline_speed = speed_limit
        self.spline_idx = 1 
        self.target_x = waypoints[-1][0]
        self.target_y = waypoints[-1][1]
        self.target_yaw = self.spline_target_yaw
        if pos_pid and len(pos_pid) == 3: self.set_pid_gains(6, *pos_pid)
        self.control_mode = "SPLINE_TRACKING"

    def update_spline_tracking(self):
        if self.control_mode != "SPLINE_TRACKING": return
        tx = self.spline_path_x[self.spline_idx]
        ty = self.spline_path_y[self.spline_idx]
        dx = tx - self.curr_x
        dy = ty - self.curr_y
        dist = math.hypot(dx, dy)
        lookahead_dist = 0.15
        if dist < lookahead_dist and self.spline_idx < len(self.spline_path_x) - 1:
            self.spline_idx += 1
            return 
        local_vx = dx * math.cos(self.curr_yaw) + dy * math.sin(self.curr_yaw)
        local_vy = -dx * math.sin(self.curr_yaw) + dy * math.cos(self.curr_yaw)
        if dist > 0:
            local_vx = (local_vx / dist) * self.spline_speed
            local_vy = (local_vy / dist) * self.spline_speed
        yaw_err = self.spline_target_yaw - self.curr_yaw
        while yaw_err > math.pi: yaw_err -= 2.0 * math.pi
        while yaw_err < -math.pi: yaw_err += 2.0 * math.pi
        v_yaw = 1.5 * yaw_err 
        msg = Twist()
        msg.linear.x = float(local_vx)
        msg.linear.y = float(local_vy)
        msg.angular.z = float(v_yaw)
        self.vel_pub.publish(msg)
    
    def stop(self):
        self.control_mode = "MANUAL"
        self.manual_vx = self.manual_vy = 0.0
        self.vel_pub.publish(Twist())
    def set_manual_velocity(self, vx, vy):
        self.control_mode = "MANUAL"
        self.manual_vx = float(vx)
        self.manual_vy = float(vy)
    def set_pid_gains(self, pid_id, p, i, d):
        msg = Float32MultiArray()
        msg.data = [float(p), float(i), float(d)]
        if pid_id == 5: self.yaw_pid_pub.publish(msg)
        elif pid_id == 6: self.pos_pid_pub.publish(msg)

    def Griper_R(self,state):
        if(state == 'active'): self.ControlServo(13,self.ServoGriper_R[1])
        elif(state == 'unactive'): self.ControlServo(13,self.ServoGriper_R[0])
        elif(state == 'hold'): self.ControlServo(13,self.ServoGriper_R[2])
    def Griper_L(self,state):
        if(state == 'active'): self.ControlServo(15,self.ServoGriper_L[1])
        elif(state == 'unactive'): self.ControlServo(15,self.ServoGriper_L[0])
        elif(state == 'hold'): self.ControlServo(15,self.ServoGriper_L[2])
    def Box_Pusher(self,state):
        if(state == 'unactive'): self.ControlServo(14,self.ServoBoxPusher[0])
        elif(state == 'active'): self.ControlServo(14,self.ServoBoxPusher[1])
    def ControlBottle_L(self,state):
        if(state == 'stop'): self.StateControlAutomation[0] = 0
        elif(state == 'up'): self.StateControlAutomation[0] = 1
        elif(state == 'down'): self.StateControlAutomation[0] = 2
        self.ControlAutomation()
    def ControlBottle_R(self,state):
        if(state == 'stop'): self.StateControlAutomation[1] = 0
        elif(state == 'up'): self.StateControlAutomation[1] = 1
        elif(state == 'down'): self.StateControlAutomation[1] = 2
        self.ControlAutomation()
    def ControlBox(self,state):
        if(state == 'up'): self.StateControlAutomation[2] = 1
        elif(state == 'down'): self.StateControlAutomation[2] = 2
        self.ControlAutomation()
    def ControlSlide(self,state):
        if(state == 'in'):
            self.StateControlAutomation[3] = 2
            self.StateControlAutomation[2] = 0
        elif(state == 'out'):
            self.StateControlAutomation[3] = 1
            self.StateControlAutomation[2] = 0
        self.ControlAutomation()
    def publish_state(self):
        msg = Int32MultiArray()
        msg.data = self.current_state
        self.cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    ctrl = SudSakhonMainController()
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