import rclpy
from rclpy.node import Node
from flask import Flask, jsonify, request
from flask_cors import CORS 
from nav_msgs.msg import Odometry
import subprocess
import threading
import os
import json

# --- Configuration ---
PORT = 8001
STORAGE_FILE = os.path.expanduser('~/monitored_services.json')
DEFAULT_SERVICES = ["sudsakhon_tf.service", "sudsakhon_odom.service"]

app = Flask(__name__)
# อนุญาต CORS เพื่อให้ Next.js (Port 3000) คุยกับ Flask (Port 8001) ได้
CORS(app)

# Global Data
robot_state = {
    "linear_x": 0.0,
    "linear_y": 0.0,
    "angular_z": 0.0,
    "monitored_services": []
}

def load_services():
    if os.path.exists(STORAGE_FILE):
        try:
            with open(STORAGE_FILE, 'r') as f: return json.load(f)
        except: pass
    return list(DEFAULT_SERVICES)

robot_state["monitored_services"] = load_services()

# --- API ROUTES ---

# 1. แก้ปัญหา Not Found เวลาเข้า IP ตรงๆ
@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "robot": "Sudsakhon",
        "message": "Backend API is running"
    })

@app.route('/api/telemetry', methods=['GET'])
def get_telemetry():
    return jsonify(robot_state)

@app.route('/api/services', methods=['GET'])
def get_services():
    results = []
    for s in robot_state["monitored_services"]:
        try:
            status = subprocess.check_output(["systemctl", "is-active", s]).decode().strip()
        except: status = "inactive"
        results.append({"name": s, "status": status})
    return jsonify(results)

@app.route('/api/control', methods=['POST'])
def control():
    data = request.json
    service = data.get('service')
    action = data.get('action')
    
    if service not in robot_state["monitored_services"]:
        return jsonify({"status": "error", "message": "Service not found"}), 404

    # 🔥 วิธีแก้ API ค้าง: โยนงานสั่ง systemctl ไปทำข้างหลัง (Background Thread)
    def execute_cmd():
        try:
            subprocess.run(["sudo", "systemctl", "reset-failed", service], check=False)
            subprocess.run(["sudo", "systemctl", action, service], check=False)
        except Exception as e:
            print(f"Command Error: {e}")

    threading.Thread(target=execute_cmd).start()
    
    # ตอบกลับทันทีเพื่อให้ Frontend ไม่ขึ้น Pending นานจนค้าง
    return jsonify({"status": "success", "message": "Command received"})

# --- ROS 2 NODE ---
class SudsakhonBridge(Node):
    def __init__(self):
        super().__init__('sudsakhon_bridge_node')
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

    def odom_callback(self, msg):
        robot_state["linear_x"] = round(msg.twist.twist.linear.x, 2)
        robot_state["linear_y"] = round(msg.twist.twist.linear.y, 2)
        robot_state["angular_z"] = round(msg.twist.twist.angular.z, 2)

def main(args=None):
    # รัน Flask แบบ Threaded เพื่อรองรับหลาย Request พร้อมกัน
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True, use_reloader=False), daemon=True).start()
    
    rclpy.init(args=args)
    node = SudsakhonBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()