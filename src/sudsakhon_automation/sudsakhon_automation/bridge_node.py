#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import serial
from std_msgs.msg import Int32MultiArray

class SudsakhonSerialBridge(Node):
    def __init__(self):
        super().__init__('sudsakhon_serial_bridge')
        
        # ประกาศ Parameters
        self.declare_parameter('port', '/dev/Controller_Automation')
        self.declare_parameter('baud', 115200)
        
        port_name = self.get_parameter('port').get_parameter_value().string_value
        baud_rate = self.get_parameter('baud').get_parameter_value().integer_value
        
        try:
            self.ser = serial.Serial(port_name, baud_rate, timeout=0.05)
            self.get_logger().info(f"Connected to STM32 on {port_name} at {baud_rate}")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to serial: {str(e)}")
            return

        # Publishers
        self.sensor_pub = self.create_publisher(Int32MultiArray, '/automation/sensors', 10)

        # Subscribers
        self.create_subscription(Int32MultiArray, '/automation/control_states', self.control_callback, 10)
        self.create_subscription(Int32MultiArray, '/automation/servo', self.servo_callback, 10)

        # Timers (แทน while loop ใน ROS 1)
        self.timer = self.create_timer(0.01, self.serial_read_callback)

    def control_callback(self, msg):
        """ รับค่า [bottleL, bottleR, box, slide] แล้วส่งลง Serial """
        if len(msg.data) >= 4:
            cmd = f"C,{msg.data[0]},{msg.data[1]},{msg.data[2]},{msg.data[3]}\n"
            self.ser.write(cmd.encode())

    # def control_callback(self, msg):
    #     """ รับค่า [bottleL, bottleR, box, slide] แล้วส่งลง Serial """
    #     if len(msg.data) >= 4:
    #         bottleL = msg.data[0]
    #         bottleR = msg.data[1]
    #         box = msg.data[2]
    #         slide = msg.data[3]

    #         is_box_up = (box == "up" or box == 1)
            
    #         is_slide_working = (slide != "stop" and slide != 0)
        
    #         is_sensor_triggered = getattr(self, 'sensor_triggered', False) 

    #         if is_box_up and (is_sensor_triggered or is_slide_working):
    #             box = "stop" if isinstance(msg.data[2], str) else 0
                
    #         cmd = f"C,{bottleL},{bottleR},{box},{slide}\n"
    #         self.ser.write(cmd.encode())

    def servo_callback(self, msg):
        """ รับค่า [channel, angle] แล้วส่งลง Serial """
        if len(msg.data) >= 2:
            cmd = f"V,{msg.data[0]},{msg.data[1]}\n"
            self.ser.write(cmd.encode())

    def serial_read_callback(self):
        """ อ่านข้อมูลสถานะเซนเซอร์จาก Serial """
        if self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line.startswith('S,'):
                    # แปลง S,1,0,1... เป็น Array ของตัวเลข
                    parts = line.split(',')[1:]
                    sensor_values = [int(p) for p in parts]
                    
                    msg = Int32MultiArray()
                    msg.data = sensor_values
                    self.sensor_pub.publish(msg)
            except Exception as e:
                # จัดการ error กรณี decode ผิดพลาดจากสัญญาณรบกวน
                pass

def main(args=None):
    rclpy.init(args=args)
    bridge = SudsakhonSerialBridge()
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(bridge, 'ser'):
            bridge.ser.close()
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()