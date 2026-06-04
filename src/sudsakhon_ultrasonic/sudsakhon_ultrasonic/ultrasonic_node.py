import rclpy
from rclpy.node import Node
import serial
from std_msgs.msg import Float32MultiArray

class UltrasonicNode(Node):
    def __init__(self):
        super().__init__('sudsakhon_ultrasonic_node')
        
        self.publisher_ = self.create_publisher(Float32MultiArray, 'ultrasonic_data', 10)
        
        port_name = '/dev/Controller_Ultrasonic'
        baud_rate = 115200
        
        try:
            self.serial_port = serial.Serial(port_name, baud_rate, timeout=0.05)
            self.get_logger().info(f'Connected to {port_name} successfully!')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to Serial: {e}')
            self.serial_port = None
            
        self.timer = self.create_timer(0.01, self.read_serial_data)

    def read_serial_data(self):
        if self.serial_port and self.serial_port.in_waiting > 0:
            try:
                line = self.serial_port.readline().decode('utf-8').strip()
                data = line.split(',')
                
                if len(data) == 2:
                    msg = Float32MultiArray()
                    msg.data = [float(data[0]), float(data[1])]
                    
                    self.publisher_.publish(msg)
            except Exception as e:
                self.get_logger().warn(f'Data parsing error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = UltrasonicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'serial_port') and node.serial_port:
            node.serial_port.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()