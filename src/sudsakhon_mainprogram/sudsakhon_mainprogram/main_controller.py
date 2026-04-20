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
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Int32 # นำเข้า message type ที่ต้องการ

# นำเข้าคลาสต่าง ๆ จากไฟล์ที่แยกไว้
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
        
        # ==========================================
        # 1. Publishers
        # ==========================================
        self.cmd_pub = self.create_publisher(Int32MultiArray, 'motor_commands', 10)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(Pose2D, '/cmd_goal', 10)
        self.max_speed_pub = self.create_publisher(Float32, '/cmd_max_speed', 10)

        # Publisher สำหรับการจูน PID
        self.pos_pid_pub = self.create_publisher(Float32MultiArray, '/cmd_pos_pid', 10)
        self.yaw_pid_pub = self.create_publisher(Float32MultiArray, '/cmd_yaw_pid', 10)

        self.marker_pub = self.create_publisher(Marker, 'detected_marker', 10)

        # ==========================================
        # 2. Subscribers
        # ==========================================
        self.sensor_sub = self.create_subscription(Int8MultiArray, 'sensor_states', self.sensor_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.rviz_goal_callback, 10)
        self.subscription = self.create_subscription( Int32, '/chair_count', self.chaircount_callback, 10)

        self.subscription = self.create_subscription(LaserScan,'scan',self.filter_callback,10)
        
        # ==========================================
        # 3. State & Systems
        # ==========================================
        self.ChairCount = 0
        self.StepChairCount = 0

        self.Lidar_left_dist = None  # มุม 20
        self.Lidar_right_dist = None # มุม -20
        self.Lidar_center_dist = None # มุม -20

        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_yaw = 0.0
        
        self.target_x_tmp = 0.0
        self.target_y_tmp = 0.0
        self.target_yaw_tmp = 0.0
        
        self.is_moving = False 
        self.is_started = False
        self.mission_step = 0 
        
        self.control_mode = "DIRECT_STM32" 
        self.manual_vx = 0.0
        self.manual_vy = 0.0

        self.dist_tolerance = 0.05  
        self.yaw_tolerance = math.radians(3.0) 
        
        self.current_state = [0, 0, 0, 0, 0, 0, 0]

        self.slide_system = SlideSystem(self)
        self.bottle_system = BottleSystem(self)
        self.bucket_system = BucketSystem(self)
        self.servo_system = ServoSystem(self)
        self.sensors = SensorSystem()
        self.nav = NavigationSystem(self) 

        self.counterboxout = 0
        self.statesensorbox = 0
        
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info('SudSakhon Controller Ready: สามารถส่ง PID พร้อม go_to ได้แล้ว')

    def chaircount_callback(self,msg):
        #self.get_logger().info(f'ได้รับจำนวนเก้าอี้: {msg.data}')
        self.ChairCount = msg.data
    def filter_callback(self, msg):
        # 1. เตรียม Marker (ลูกบอลสีแดงดวงใหญ่)
        marker = Marker()
        marker.header = msg.header
        marker.type = Marker.SPHERE_LIST
        marker.scale.x = 0.05 # ใหญ่สะใจ 30 ซม.
        marker.scale.y = 0.05
        marker.scale.z = 0.1
        marker.color.a = 1.0 
        marker.color.g = 1.0 # สีแดง

        # 2. เลือกมุมที่ต้องการ
        target_angles = [28, -28, 0]

        for deg in target_angles:
            rad = np.deg2rad(deg)
            idx = int((rad - msg.angle_min) / msg.angle_increment)

            if 0 <= idx < len(msg.ranges):
                dist = msg.ranges[idx]
                
                if np.isfinite(dist):
                    # --- ส่วนที่ดึงความยาวออกมา ---
                    if deg == target_angles[0]:
                        self.Lidar_left_dist = dist
                    elif deg == target_angles[1]:
                        self.Lidar_right_dist = dist
                    elif deg == target_angles[2]:
                        self.Lidar_center_dist = dist
                    
                    # (ใส่ค่าลง Marker เหมือนเดิม)
                    p = Point()
                    p.x = dist * np.cos(rad)
                    p.y = dist * np.sin(rad)
                    p.z = 0.0
                    marker.points.append(p)

        # ลอง Print ดูค่าความยาวที่ได้
        #self.get_logger().info(f'Distance L(20°): {self.left_dist:.2f} m, R(-20°): {self.right_dist:.2f} m')
        
        # ส่ง Marker ออกไป RViz
        self.marker_pub.publish(marker)

    def sensor_callback(self, msg):
        self.sensors.update(msg.data)

    def odom_callback(self, msg):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        t3 = +2.0 * (q.w * q.z + q.x * q.y)
        t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.curr_yaw = math.atan2(t3, t4)

    def rviz_goal_callback(self, msg):
        q = msg.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        # ตัวอย่างการส่งจาก RViz โดยใช้ PID มาตรฐาน
        self.go_to(msg.pose.position.x, msg.pose.position.y, math.degrees(yaw), 
                   pos_pid=[12.5, 0.01, 1.2], yaw_pid=[500.0, 0.0, 25.0])

    def set_pid_gains(self, id, p, i, d):
        """ส่งค่า PID ไปจูนที่ STM32"""
        msg = Float32MultiArray()
        msg.data = [float(p), float(i), float(d)]
        if id == 5:
            self.yaw_pid_pub.publish(msg)
        elif id == 6:
            self.pos_pid_pub.publish(msg)
        # ไม่ใส่ sleep ตรงนี้เพื่อให้โปรแกรมไหลลื่น แต่จะแยกไปส่งใน go_to แทน

    def stop(self):
        """สั่งหยุดหุ่นยนต์ทันที"""
        self.control_mode = "MANUAL"
        self.is_moving = False
        self.manual_vx = 0.0
        self.manual_vy = 0.0
        msg = Twist()
        self.vel_pub.publish(msg)
        self.get_logger().info('🛑 Robot Stopped')

    def set_manual_velocity(self, vx, vy):
        """สั่งความเร็วหุ่นยนต์โดยตรง (Manual)"""
        self.control_mode = "MANUAL"
        self.manual_vx = float(vx)
        self.manual_vy = float(vy)
        
        msg = Twist()
        msg.linear.x = self.manual_vx
        msg.linear.y = self.manual_vy
        self.vel_pub.publish(msg)


    def control_loop(self):
        """Loop หลักจัดการภารกิจ"""

        # --- 1. เช็คระยะห่างปัจจุบัน (ใช้ตัดสินใจว่าถึงหรือยัง) ---
        dist_err = math.sqrt((self.target_x_tmp - self.curr_x)**2 + (self.target_y_tmp - self.curr_y)**2)
        arrived = (dist_err < self.dist_tolerance)

        # --- 2. ลำดับภารกิจ (รันทีละ Step) ---

        # [START] กดปุ่มเริ่ม
        if self.sensors.sw1 == 0 and self.mission_step == 0:
            self.get_logger().info('🔘 เริ่มภารกิจ STEP 1')
            self.go_to(0.0, 0.0, 0.0, speed_limit=0.5, pos_pid=[13.5, 0.00002, 4.8])
            self.bucket_system.out_pos()
            self.slide_system.enable()
            self.slide_system.down()
            self.mission_step = 1

        # [Step 1 -> 2] ถึงจุดแรกแล้ว ไปจุดที่สองต่อ
        elif self.mission_step == 1 and arrived:
            self.get_logger().info('🏁 ไปจุดที่ 2 (Speed 1.4)')
            self.go_to(-0.50, 0.0, 0.0, speed_limit=1.6, pos_pid=[13.5, 0.00002, 6.5])
            self.mission_step = 2
        

        # [Step 2 -> 3] ถึงจุดสองแล้ว ไปจ่อหน้าโต๊ะ
        elif self.mission_step == 2 and arrived:
            self.get_logger().info('🏁 ไปจุดที่ 3 (จ่อหน้าโต๊ะ)')
            self.go_to(-0.42, -0.45, 0.0, speed_limit=1.4, pos_pid=[6.5, 0.00002, 3.0])
            self.mission_step = 3
        
        # [Step 3 -> 4] ถึงจุดจ่อแล้ว เปลี่ยนเป็นโหมดเดินหาขอบโต๊ะ (Sensor Mode)
        elif self.mission_step == 3 and arrived:
            self.mission_step = 4

        elif self.mission_step == 4:

            Lenght_Lidar = 0.9
            if self.Lidar_left_dist <= Lenght_Lidar and self.Lidar_right_dist <= Lenght_Lidar: # เข้าเงื่อนไข
                self.set_manual_velocity(0.0, 0.0) # ค่อยๆ คลานเข้าหาโต๊ะ
                self.get_logger().info(f'stop: {self.Lidar_left_dist:.2f} m, R(-20°): {self.Lidar_right_dist:.2f} m')
                self.mission_step = 5
                time.sleep(1)
            elif self.Lidar_left_dist <= Lenght_Lidar and self.Lidar_right_dist >= Lenght_Lidar: #วิ่งซ้าย
                self.set_manual_velocity(0.0, 0.05) # ค่อยๆ คลานเข้าหาโต๊ะ
                self.get_logger().info(f'left: {self.Lidar_left_dist:.2f} m, R(-20°): {self.Lidar_right_dist:.2f} m')
            elif self.Lidar_left_dist >= Lenght_Lidar and self.Lidar_right_dist <= Lenght_Lidar: #วิ่งขวา
                self.set_manual_velocity(0.0, -0.05) # ค่อยๆ คลานเข้าหาโต๊ะ
                self.get_logger().info(f'right: {self.Lidar_left_dist:.2f} m, R(-20°): {self.Lidar_right_dist:.2f} m')

        elif self.mission_step == 5:
            self.StepChairCount = self.ChairCount
            self.get_logger().info(str(self.StepChairCount) + " ตัว A" + str(self.mission_step))
            self.slide_system.enable()
            self.slide_system.up()
            self.mission_step = 6
            

        elif self.mission_step == 6:
            #print("speed = "+str(self.Lidar_center_dist))

            if self.sensors.table_l == 0 or self.sensors.table_r == 0:
                
                self.slide_system.enable()
                self.slide_system.up()

                self.stop()
                self.get_logger().info('🎯 เจอขอบโต๊ะ! -> ไป Step 5')
                self.get_logger().info(str(self.StepChairCount) + " ตัว B" + str(self.mission_step))
                #self.go_to(-0.57, -0.625, 0.0, speed_limit=0.3)

                self.set_manual_velocity(0.1, 0.0) # ค่อยๆ คลานเข้าหาโต๊ะ
                time.sleep(0.5)
                self.set_manual_velocity(0.0, 0.0) # ค่อยๆ คลานเข้าหาโต๊ะ
                
                if(self.StepChairCount <= 1):
                    self.mission_step = 7.1
                else:
                    self.mission_step = 7.2
                    self.counterboxout = 0
                    self.statesensorbox = 0
            else:
                if(self.Lidar_center_dist <= 0.38):
                    self.set_manual_velocity(0.06, 0.0)
                else:
                    self.set_manual_velocity(0.10, 0.0) # ค่อยๆ คลานเข้าหาโต๊ะ
        




        #case 1 เก้าอี้
        elif self.mission_step == 7.1:
            self.servo_system.PullBox.on()
            time.sleep(0.5)
            self.servo_system.PullBox.off()

            self.slide_system.enable()
            self.slide_system.down()
            self.mission_step = 7.11
        elif self.mission_step == 7.11:
            if self.sensors.bottle_detected == 0:
                self.bottle_system.disable()
                self.bottle_system.out_pos()
                time.sleep(0.5)
                self.servo_system.BlockBottle.on()
                self.set_manual_velocity(-0.25, 0.0) # ค่อยๆ คลานเข้าหาโต๊ะ
                time.sleep(0.75)
                self.set_manual_velocity(0.0, 0.0) # ค่อยๆ คลานเข้าหาโต๊ะ
                self.servo_system.BlockBottle.off()
                self.mission_step = 8
            else:
                self.bottle_system.enable()
                self.bottle_system.out_pos()





        #case 2 เก้าอี้

        elif self.mission_step == 7.2:
            if self.sensors.box_detected == 0:
                self.servo_system.PullBox.on()
                time.sleep(0.5)
                self.servo_system.PullBox.off()
                time.sleep(1.5)
                self.slide_system.enable()
                self.slide_system.up()
                self.mission_step = 7.21

        elif self.mission_step == 7.21:
            if self.sensors.box_detected == 0 and self.counterboxout >=1:
                time.sleep(0.5)
                self.servo_system.PullBox.on()
                time.sleep(0.5)
                self.servo_system.PullBox.off()

                print("---> Case Ok")
                #self.slide_system.disable()
                #self.slide_system.down()
                self.mission_step = 7.22
            else:
                #if(self.statesensorbox == 1)
                self.counterboxout = self.counterboxout + 1
                print("---> Up" + str(self.counterboxout))

        elif self.mission_step == 7.22:
            if self.sensors.bottle_detected == 0:
                self.bottle_system.disable()
                self.bottle_system.out_pos()
                time.sleep(0.5)
                self.servo_system.BlockBottle.on()
                self.mission_step = 7.23
            else:
                self.bottle_system.enable()
                self.bottle_system.out_pos()

        elif self.mission_step == 7.23:
            if self.sensors.table_l == 1:
                self.servo_system.BlockBottle.off()
                time.sleep(0.2)
                self.set_manual_velocity(0.0, -0.15)
                time.sleep(0.8)
                self.set_manual_velocity(0.0, 0.0)
                self.mission_step = 7.24
            else:

                self.set_manual_velocity(0.0, 0.15)

        elif self.mission_step == 7.24:
            if self.sensors.bottle_detected == 0:
                self.bottle_system.disable()
                self.bottle_system.out_pos()
                time.sleep(0.5)
                self.servo_system.BlockBottle.on()

                self.set_manual_velocity(-0.15, 0.0) # ค่อยๆ คลานเข้าหาโต๊ะ
                time.sleep(0.85)
                self.set_manual_velocity(0.0, 0.0) # ค่อยๆ คลานเข้าหาโต๊ะ

                self.servo_system.BlockBottle.off()
                self.mission_step = 8
            else:
                self.bottle_system.enable()
                self.bottle_system.out_pos()

        elif self.mission_step == 8:
            self.slide_system.enable()
            self.slide_system.down()
            self.go_to(-0.48, -0.45, 0.0, speed_limit=1.4, pos_pid=[6.5, 0.00002, 3.0])
            self.mission_step = 9
        elif self.mission_step == 9 and arrived:
            self.mission_step = 10

        elif self.mission_step == 10:
            self.go_to(-0.48, -0.95, 0.0, speed_limit=1.4, pos_pid=[6.5, 0.00002, 3.0])
            self.mission_step = 11
        elif self.mission_step == 11 and arrived:
            self.slide_system.enable()
            self.slide_system.down()
            self.mission_step = 12
            
        '''
        elif self.mission_step == 7.21:
            if self.sensors.bottle_detected == 0:
                self.bottle_system.disable()
                self.bottle_system.out_pos()
                time.sleep(0.2)
                self.servo_system.BlockBottle.on()

                
                self.set_manual_velocity(-0.15, 0.0) # ค่อยๆ คลานเข้าหาโต๊ะ
                time.sleep(0.75)
                self.set_manual_velocity(0.0, 0.0) # ค่อยๆ คลานเข้าหาโต๊ะ
                

                self.mission_step = 7.22
            else:
                self.bottle_system.enable()
                self.bottle_system.out_pos()
        '''
        '''
        elif self.mission_step == 7.22:
            if self.sensors.bottle_detected == 0:
                self.bottle_system.disable()
                self.bottle_system.out_pos()
                time.sleep(0.2)
                self.servo_system.BlockBottle.on()
                self.set_manual_velocity(-0.15, 0.0) # ค่อยๆ คลานเข้าหาโต๊ะ
                time.sleep(0.75)
                self.set_manual_velocity(0.0, 0.0) # ค่อยๆ คลานเข้าหาโต๊ะ
                self.mission_step = 7.22
            else:
                self.bottle_system.enable()
                self.bottle_system.out_pos()

        '''

        '''
        elif self.mission_step == 7.2:
            self.slide_system.disable()
            self.servo_BlockBottleDown("on")
            self.servo_system.PullBox.on()
            time.sleep(1.0)
            self.servo_system.BlockBottle.on()
            self.servo_system.PullBox.off()
            time.sleep(0.1)
            self.slide_system.enable()
            self.slide_system.up()
            time.sleep(1.5)
            self.servo_system.PullBox.on()
            time.sleep(1.0)
            self.servo_system.PullBox.off()
            time.sleep(0.2)
            self.slide_system.down()

            self.set_manual_velocity(-0.15, 0.0) # ค่อยๆ คลานเข้าหาโต๊ะ
            time.sleep(1.0)
            self.set_manual_velocity(0.0, 0.0) # ค่อยๆ คลานเข้าหาโต๊ะ
            self.servo_system.BlockBottle.off()
            self.servo_BlockBottleDown("off")
            self.bottle_system.disable()
            self.bottle_system.out_pos()

            time.sleep(0.3)
            self.set_manual_velocity(-0.0, 0.3) # ค่อยๆ คลานเข้าหาโต๊ะ
            time.sleep(0.5)
            self.set_manual_velocity(0.0, 0.0) # ค่อยๆ คลานเข้าหาโต๊ะ

            self.mission_step = 7.3
        
        elif self.mission_step == 7.3:
            #print("speed = "+str(self.Lidar_center_dist))
            if self.sensors.table_l == 0 or self.sensors.table_r == 0:
                self.stop()
                self.get_logger().info('🎯 เจอขอบโต๊ะ! -> ไป Step 5')
                self.get_logger().info(str(self.StepChairCount) + " ตัว B" + str(self.mission_step))

                self.mission_step = 8
                #self.go_to(-0.57, -0.625, 0.0, speed_limit=0.3)
            else:
                if(self.Lidar_center_dist <= 0.38):
                    self.set_manual_velocity(0.06, 0.0)
                else:
                    self.set_manual_velocity(0.10, 0.0) # ค่อยๆ คลานเข้าหาโต๊ะ
        
        elif self.mission_step == 8:
            #self.servo_system.BlockBottleDown.off()
            self.servo_system.BlockBottle.off()
            self.servo_BlockBottleDown("off")
            self.mission_step = 9

        '''

        # 2. จัดการการส่งความเร็วตามโหมด
        if self.control_mode == "MANUAL":
            msg = Twist()
            msg.linear.x = self.manual_vx
            msg.linear.y = self.manual_vy
            self.vel_pub.publish(msg)
            
        elif self.control_mode == "INTERNAL":
            cmd_vel, arrived_internal = self.nav.calculate_velocity(self.curr_x, self.curr_y, self.curr_yaw)
            self.vel_pub.publish(cmd_vel)
            arrived = arrived_internal

        '''
        # 1. เช็คสถานะ Arrived
        dist_err = math.sqrt((self.target_x_tmp - self.curr_x)**2 + (self.target_y_tmp - self.curr_y)**2)
        yaw_err = self.target_yaw_tmp - self.curr_yaw
        while yaw_err > math.pi: yaw_err -= 2*math.pi
        while yaw_err < -math.pi: yaw_err += 2*math.pi
        arrived = (dist_err < self.dist_tolerance) and (abs(yaw_err) < self.yaw_tolerance)
        
        # 1. เช็คสถานะ Arrived
        dist_err = math.sqrt((self.target_x_tmp - self.curr_x)**2 + (self.target_y_tmp - self.curr_y)**2)
        yaw_err = self.target_yaw_tmp - self.curr_yaw
        while yaw_err > math.pi: yaw_err -= 2*math.pi
        while yaw_err < -math.pi: yaw_err += 2*math.pi
        arrived = (dist_err < self.dist_tolerance) and (abs(yaw_err) < self.yaw_tolerance)
        
        # 2. จัดการการส่งความเร็วตามโหมด
        if self.control_mode == "MANUAL":
            msg = Twist()
            msg.linear.x = self.manual_vx
            msg.linear.y = self.manual_vy
            self.vel_pub.publish(msg)
            
        elif self.control_mode == "INTERNAL":
            cmd_vel, arrived_internal = self.nav.calculate_velocity(self.curr_x, self.curr_y, self.curr_yaw)
            self.vel_pub.publish(cmd_vel)
            arrived = arrived_internal

        # 3. จัดการ Mission Sequence
        if self.sensors.sw1 == 0 and not self.is_started:
            self.get_logger().info('🔘 SW1: เริ่มภารกิจ STEP 1')
            self.is_started = True
            self.mission_step = 1
            self.bucket_system.out_pos()
            
            # ส่งไปจุด 0,0 พร้อมจูน PID ทันที (อัปเดตค่า D เป็น 4.8 ตาม snippet)
            self.go_to(0.0, 0.0, 0.0, speed_limit=0.5, 
                       pos_pid=[12.5, 0.00002, 4.8], 
                       yaw_pid=[400.0, 0.0, 20.0])

        # ตรวจสอบสถานะการเข้าถึงจุดหมายสำหรับ Step 1 และ 2
        if self.is_started and self.is_moving and arrived:
            if self.mission_step == 1:
                self.get_logger().info('🏁 ถึงจุดที่ 1 -> ไปจุดที่ 2 (Speed 1.4)')
                self.is_moving = False 
                self.mission_step = 2
                self.go_to(-0.50, -0.0, 0.0, speed_limit=1.4, 
                           pos_pid=[12.5, 0.00002, 4.5])
            
            elif self.mission_step == 2:
                self.get_logger().info('🏁 ถึงจุดที่ 2 -> ไปจุดที่ 3 (เปลี่ยน PID เป็น D 6.5)')
                self.is_moving = False 
                
                self.go_to(-0.55, -0.625, 0.0, speed_limit=1.0, 
                           pos_pid=[12.5, 0.00002, 6.5])

                self.mission_step = 3
            elif self.mission_step == 3:
                self.get_logger().info('🏁 ถึงจุดที่ 3 -> ไปจุดที่ 4 (เปลี่ยน PID เป็น D 6.5)')
                self.is_moving = False 
                self.mission_step = 4

            elif self.mission_step == 5:
                self.get_logger().info('🏁 ถึงจุดที่ 5 -> ไปจุดที่ 6 (เปลี่ยน PID เป็น D 6.5)')
                self.go_to(-0.57, -1.18, 0.0, speed_limit=1.0, 
                           pos_pid=[12.5, 0.00002, 6.5])
                self.is_moving = True 
                self.mission_step = 6

            elif self.mission_step == 6:
                self.get_logger().info('🏁 ถึงจุดที่ 6 -> ไปจุดที่ 7 (เปลี่ยน PID เป็น D 6.5)')
                self.is_moving = False 
                self.mission_step = 7

            elif self.mission_step == 8:
                self.get_logger().info('🏁 ถึงจุดที่ 8 -> ไปจุดที่ 9 (เปลี่ยน PID เป็น D 6.5)')
                self.go_to(-0.57, -1.22, 180.0, speed_limit=1.0, 
                           pos_pid=[12.5, 0.00002, 6.5], 
                            yaw_pid=[150.0, 0.0, 20.0])
                self.is_moving = True 
                self.mission_step = 9

            elif self.mission_step == 9:
                self.get_logger().info('🏁 ถึงจุดที่ 9 -> ไปจุดที่ 10 (เปลี่ยน PID เป็น D 6.5)')
                self.is_moving = False 
                self.mission_step = 10


            elif self.mission_step == 11:
                self.get_logger().info('🏁 ถึงจุดที่ 11 -> ไปจุดที่ 12 (เปลี่ยน PID เป็น D 6.5)')
                self.go_to(-0.57, -0.635, 180.0, speed_limit=1.0, 
                           pos_pid=[12.5, 0.00002, 6.5], 
                            yaw_pid=[80.0, 0.0, 20.0])
                self.is_moving = True 
                self.mission_step = 12

            elif self.mission_step == 12:
                self.get_logger().info('🏁 ถึงจุดที่ 12 -> ไปจุดที่ 13 (เปลี่ยน PID เป็น D 6.5)')
                self.is_moving = False 
                self.mission_step = 13

        # Logic พิเศษสำหรับ Step 3: ตรวจสอบเซนเซอร์ขณะเคลื่อนที่
        elif self.is_started  and self.mission_step == 4:
            if self.sensors.table_l == 0 and self.sensors.table_r == 0:
                self.get_logger().info('🏁 ตรวจพบขอบโต๊ะ! หยุดหุ่นยนต์ -> Step 5')
                self.stop()

                self.go_to(-0.57, -0.625, 0.0, speed_limit=0.3, 
                           pos_pid=[12.5, 0.00002, 6.5])

                self.mission_step = 5
            else:
                # สั่งเคลื่อนที่แบบ Manual เพื่อประคองเข้าหาโต๊ะช้าๆ (ความเร็ว 0.1)
                self.set_manual_velocity(0.15, 0.0)


        elif self.is_started  and self.mission_step == 7:
            if self.sensors.table_l == 0 and self.sensors.table_r == 0:
                self.get_logger().info('🏁 ตรวจพบขอบโต๊ะ! หยุดหุ่นยนต์ -> Step 7')
                self.stop()

                self.go_to(-0.57, -1.23, 0.0, speed_limit=0.3, 
                           pos_pid=[12.5, 0.00002, 6.5])

                self.mission_step = 8
            else:
                # สั่งเคลื่อนที่แบบ Manual เพื่อประคองเข้าหาโต๊ะช้าๆ (ความเร็ว 0.1)
                self.set_manual_velocity(0.15, 0.0)


        elif self.is_started  and self.mission_step == 10:
            if self.sensors.table_l == 0 and self.sensors.table_r == 0:
                self.get_logger().info('🏁 ตรวจพบขอบโต๊ะ! หยุดหุ่นยนต์ -> Step 10')
                self.stop()

                self.go_to(-0.57, -1.23, 180.0, speed_limit=0.3, 
                           pos_pid=[12.5, 0.00002, 6.5])

                self.mission_step = 11
            else:
                # สั่งเคลื่อนที่แบบ Manual เพื่อประคองเข้าหาโต๊ะช้าๆ (ความเร็ว 0.1)
                self.set_manual_velocity(0.15, 0.0)


        elif self.is_started  and self.mission_step == 13:
            if self.sensors.table_l == 0 and self.sensors.table_r == 0:
                self.get_logger().info('🏁 ตรวจพบขอบโต๊ะ! หยุดหุ่นยนต์ -> Step 12')
                self.stop()

                self.go_to(-0.57, -1.23, 180.0, speed_limit=0.3, 
                           pos_pid=[12.5, 0.00002, 6.5])

                self.mission_step = 14
            else:
                # สั่งเคลื่อนที่แบบ Manual เพื่อประคองเข้าหาโต๊ะช้าๆ (ความเร็ว 0.1)
                self.set_manual_velocity(0.15, 0.0)

        '''
                    

        # 4. ส่งสถานะ Actuators
        self.publish_state()

    def publish_state(self):
        msg = Int32MultiArray()
        msg.data = self.current_state
        self.cmd_pub.publish(msg)

    def set_servo(self, channel, angle):
        self.current_state[5] = int(channel)
        self.current_state[6] = int(angle)
        self.publish_state()

    def servo_BlockBottleDown(self,pos):
        if(pos == "off"):
            self.set_servo(4,90)
        elif(pos == "on"):
            self.set_servo(4,20)


    def go_to(self, x, y, yaw_deg, mode="DIRECT_STM32", speed_limit=0.6, pos_pid=None, yaw_pid=None):
        """
        ฟังก์ชันสั่งเข้าโหมดคุมตำแหน่งอัตโนมัติ
        - pos_pid: list [p, i, d] ถ้าไม่ใส่จะใช้ค่าเดิมในบอร์ด
        - yaw_pid: list [p, i, d] ถ้าไม่ใส่จะใช้ค่าเดิมในบอร์ด
        """
        yaw_rad = math.radians(yaw_deg)
        self.target_x_tmp = x
        self.target_y_tmp = y
        self.target_yaw_tmp = yaw_rad
        self.is_moving = True 
        self.control_mode = mode
        
        # --- ส่งค่า PID ก่อนถ้ามีการระบุมา ---
        if pos_pid is not None and len(pos_pid) == 3:
            self.set_pid_gains(6, pos_pid[0], pos_pid[1], pos_pid[2])
            # self.get_logger().info(f'🔧 Update Pos PID: {pos_pid}')
            
        if yaw_pid is not None and len(yaw_pid) == 3:
            self.set_pid_gains(5, yaw_pid[0], yaw_pid[1], yaw_pid[2])
            # self.get_logger().info(f'🔧 Update Yaw PID: {yaw_pid}')

        if mode == "DIRECT_STM32":
            # ส่งความเร็วสูงสุด
            speed_msg = Float32()
            speed_msg.data = float(speed_limit)
            self.max_speed_pub.publish(speed_msg)
            
            # ส่งเป้าหมายพิกัด
            goal = Pose2D()
            goal.x = float(x)
            goal.y = float(y)
            goal.theta = float(yaw_rad)
            self.goal_pub.publish(goal)
            self.get_logger().info(f'🚀 Go to: ({x}, {y}) {yaw_deg}° | Max: {speed_limit}m/s')
        else:
            self.nav.set_goal(x, y, yaw_rad, self.curr_x, self.curr_y, mode="DIRECT", cruise_speed=speed_limit)

def main(args=None):
    rclpy.init(args=args)
    controller = SudSakhonMainController()
    
    # Setup เริ่มต้น
    controller.slide_system.enable()
    controller.slide_system.down()
    controller.servo_system.BlockBottle.off()
    #controller.servo_system.BlockBottleDown.off()
    
    controller.bucket_system.in_pos()
    controller.bottle_system.disable()
    controller.bottle_system.out_pos()

    #controller.servo_system.PullBox.on()
    #time.sleep(1)
    controller.servo_system.PullBox.off()

    controller.set_servo(4,90)

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()