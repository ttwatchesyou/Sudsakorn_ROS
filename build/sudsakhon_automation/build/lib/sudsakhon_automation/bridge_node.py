import rclpy
from rclpy.node import Node
import serial
from std_msgs.msg import Int32MultiArray, Int8MultiArray

class SudSakhonBridge(Node):
    def __init__(self):
        super().__init__('sudsakhon_bridge')
        
        # ตั้งค่าพอร์ตและ Baudrate
        self.declare_parameter('port', '/dev/Controller_Automation')
        self.declare_parameter('baud', 115200)
        
        port = self.get_parameter('port').value
        baud = self.get_parameter('baud').value

        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            self.get_logger().info(f'SudSakhon Bridge: เชื่อมต่อกับ STM32 ที่ {port} สำเร็จ (Servo Mode)')
        except Exception as e:
            self.get_logger().error(f'SudSakhon Bridge: ไม่สามารถเชื่อมต่อพอร์ตได้: {str(e)}')

        # Publisher: ส่งสถานะ Sensor กลับไปยัง ROS
        self.sensor_pub = self.create_publisher(Int8MultiArray, 'sensor_states', 10)
        
        # Subscriber: รับคำสั่งจาก ROS
        # ลำดับข้อมูลในอาร์เรย์: [Box, Bucket, Bottle, EnSlid, EnBott, PCA_CH, SERVO_ANGLE]
        self.subscription = self.create_subscription(
            Int32MultiArray, 'motor_commands', self.cmd_callback, 10)

        # Timer สำหรับอ่านค่าจาก Serial (50Hz)
        self.timer = self.create_timer(0.02, self.read_serial)

    def cmd_callback(self, msg):
        # ตรวจสอบว่ามีข้อมูลครบ 7 ตัวหรือไม่
        if len(msg.data) >= 7:
            # ตรวจสอบขอบเขตองศา (0-180) เพื่อความปลอดภัยก่อนส่ง
            angle = max(0, min(180, msg.data[6]))
            # สร้างโปรโตคอล CSV ส่งไปยัง STM32
            cmd_str = f"C,{msg.data[0]},{msg.data[1]},{msg.data[2]},{msg.data[3]},{msg.data[4]},{msg.data[5]},{angle}\n"
            self.ser.write(cmd_str.encode())
        else:
            self.get_logger().warn('ข้อมูล motor_commands ไม่ครบ 7 ตำแหน่ง (ต้องการ PCA_CH และ SERVO_ANGLE)')

    def read_serial(self):
        if hasattr(self, 'ser') and self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line.startswith("S,"):
                    raw_vals = line.split(',')[1:]
                    msg = Int8MultiArray()
                    msg.data = [int(v) for v in raw_vals]
                    self.sensor_pub.publish(msg)
            except Exception:
                pass

def main(args=None):
    rclpy.init(args=args)
    node = SudSakhonBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()