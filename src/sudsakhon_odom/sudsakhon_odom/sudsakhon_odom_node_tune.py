#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import tf2_ros
import serial
import threading
import time
import math
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Empty, Float32MultiArray, Int32MultiArray
from geometry_msgs.msg import Quaternion, TransformStamped, Pose2D, Twist
from flask import Flask, render_template_string
from flask_socketio import SocketIO

# ==========================================
# 1. CONFIGURATION
# ==========================================
SERIAL_PORT = '/dev/Controller_Base' 
BAUD_RATE = 115200
WEB_PORT = 5000

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# UI ที่เพิ่มการแสดงผล Speed Limit ปัจจุบัน
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SudSakhon Control Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
</head>
<body class="bg-slate-950 text-slate-100 font-sans">
    <div class="container mx-auto p-4 max-w-7xl">
        <header class="flex justify-between items-center mb-6 border-b border-slate-800 pb-4">
            <div>
                <h1 class="text-2xl font-black text-blue-500 tracking-tight uppercase">SUDSAKHON BRIDGE</h1>
                <p class="text-xs text-slate-500 uppercase font-bold text-emerald-400">ROS 2 Jazzy • Speed Monitoring Active</p>
            </div>
            <div id="status" class="px-4 py-1.5 rounded-full bg-red-500/10 text-red-500 border border-red-500/20 text-xs font-bold self-center tracking-widest uppercase">DISCONNECTED</div>
        </header>
        
        <div class="grid grid-cols-12 gap-6">
            <!-- Sidebar: Controls -->
            <div class="col-span-12 lg:col-span-4 space-y-4">
                
                <!-- Speed Limit Control (S) - เพิ่มการอัปเดตค่าจาก Topic -->
                <div class="bg-slate-900 border border-slate-800 p-5 rounded-2xl shadow-2xl border-l-4 border-l-blue-500">
                    <div class="flex justify-between items-center mb-4">
                        <h2 class="text-sm font-black text-blue-400 uppercase">Speed Limit (S)</h2>
                        <div id="speed-indicator" class="text-[10px] bg-blue-500/20 text-blue-400 px-2 py-0.5 rounded font-bold uppercase tracking-tighter">Current: 0.6 m/s</div>
                    </div>
                    <div class="space-y-3">
                        <div class="flex items-center gap-2">
                            <input id="max_speed" type="number" value="0.6" step="0.1" class="flex-1 bg-slate-800 p-3 rounded-lg text-white font-mono outline-none border border-slate-700 focus:border-blue-500 transition-colors" placeholder="m/s">
                            <button onclick="sendMaxSpeed()" class="bg-blue-600 hover:bg-blue-500 p-3 rounded-xl font-black text-xs transition-all text-white px-6">SET</button>
                        </div>
                        <p class="text-[9px] text-slate-500 italic">* อัปเดตอัตโนมัติเมื่อสั่งจาก Topic หรือ Main Controller</p>
                    </div>
                </div>

                <!-- Manual Control -->
                <div class="bg-slate-900 border border-slate-800 p-5 rounded-2xl shadow-2xl">
                    <h2 class="text-sm font-black mb-4 text-pink-400 flex items-center gap-2 uppercase">Manual Override (V)</h2>
                    <div class="grid grid-cols-3 gap-2">
                        <div></div>
                        <button onmousedown="manualMove(0.4, 0)" onmouseup="manualMove(0,0)" class="bg-slate-800 p-4 rounded-xl hover:bg-slate-700 active:bg-pink-600 transition-colors text-xl">▲</button>
                        <div></div>
                        <button onmousedown="manualMove(0, 0.4)" onmouseup="manualMove(0,0)" class="bg-slate-800 p-4 rounded-xl hover:bg-slate-700 active:bg-pink-600 transition-colors text-xl">◀</button>
                        <button onclick="manualMove(0,0)" class="bg-red-900/40 p-4 rounded-xl border border-red-500 text-red-500 font-bold uppercase text-[10px]">Stop</button>
                        <button onmousedown="manualMove(0, -0.4)" onmouseup="manualMove(0,0)" class="bg-slate-800 p-4 rounded-xl hover:bg-slate-700 active:bg-pink-600 transition-colors text-xl">▶</button>
                        <div></div>
                        <button onmousedown="manualMove(-0.4, 0)" onmouseup="manualMove(0,0)" class="bg-slate-800 p-4 rounded-xl hover:bg-slate-700 active:bg-pink-600 transition-colors text-xl">▼</button>
                        <div></div>
                    </div>
                </div>

                <!-- PID Tuning -->
                <div class="bg-slate-900 border border-slate-800 p-5 rounded-2xl shadow-2xl space-y-4">
                    <div>
                        <h2 class="text-[10px] font-black mb-2 text-yellow-500 uppercase">Position PID (ID 6)</h2>
                        <div class="flex gap-2">
                            <input id="kp_p" type="number" value="12.5" step="0.1" class="w-1/3 bg-slate-800 p-2 rounded text-center font-mono text-xs">
                            <input id="ki_p" type="number" value="0.01" step="0.001" class="w-1/3 bg-slate-800 p-2 rounded text-center font-mono text-xs">
                            <input id="kd_p" type="number" value="1.2" step="0.01" class="w-1/3 bg-slate-800 p-2 rounded text-center font-mono text-xs">
                        </div>
                        <button onclick="sendPID(6)" class="w-full mt-2 bg-yellow-600/20 text-yellow-500 border border-yellow-500/30 p-2 rounded-lg font-black text-[10px] uppercase">Update Position PID</button>
                    </div>
                    <div>
                        <h2 class="text-[10px] font-black mb-2 text-orange-500 uppercase">Yaw PID (ID 5)</h2>
                        <div class="flex gap-2">
                            <input id="kp_y" type="number" value="500.0" step="1" class="w-1/3 bg-slate-800 p-2 rounded text-center font-mono text-xs">
                            <input id="ki_y" type="number" value="0.0" step="0.1" class="w-1/3 bg-slate-800 p-2 rounded text-center font-mono text-xs">
                            <input id="kd_y" type="number" value="25.0" step="0.1" class="w-1/3 bg-slate-800 p-2 rounded text-center font-mono text-xs">
                        </div>
                        <button onclick="sendPID(5)" class="w-full mt-2 bg-orange-600/20 text-orange-500 border border-orange-500/30 p-2 rounded-lg font-black text-[10px] uppercase">Update Yaw PID</button>
                    </div>
                </div>
            </div>

            <!-- Main Content: Telemetry Readout -->
            <div class="col-span-12 lg:col-span-8 space-y-6">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="bg-slate-900 border border-slate-800 p-10 rounded-3xl flex flex-col justify-center items-center shadow-2xl border-t-4 border-t-blue-500">
                        <div class="text-[10px] text-slate-500 font-bold mb-2 tracking-[0.3em] uppercase">POSITION X (m)</div>
                        <div id="curX" class="text-8xl font-mono font-black text-blue-400">0.000</div>
                    </div>
                    
                    <div class="bg-slate-900 border border-slate-800 p-10 rounded-3xl flex flex-col justify-center items-center shadow-2xl border-t-4 border-t-emerald-500">
                        <div class="text-[10px] text-slate-500 font-bold mb-2 tracking-[0.3em] uppercase">POSITION Y (m)</div>
                        <div id="curY" class="text-8xl font-mono font-black text-emerald-400">0.000</div>
                    </div>

                    <div class="bg-slate-900 border border-slate-800 p-10 rounded-3xl flex flex-col justify-center items-center shadow-2xl border-t-4 border-t-purple-500">
                        <div class="text-[10px] text-slate-500 font-bold mb-2 tracking-[0.3em] uppercase">HEADING YAW (rad)</div>
                        <div id="curYaw" class="text-8xl font-mono font-black text-purple-400">0.000</div>
                    </div>

                    <div class="bg-slate-900 border border-slate-800 p-10 rounded-3xl flex flex-col justify-center items-center shadow-2xl border-t-4 border-t-orange-500">
                        <div class="text-[10px] text-slate-500 font-bold mb-2 tracking-[0.3em] uppercase">ROBOT SPEED (m/s)</div>
                        <div id="curSpeed" class="text-8xl font-mono font-black text-orange-400">0.000</div>
                    </div>
                </div>

                <!-- Status Panel -->
                <div class="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-xl">
                    <h2 class="text-[10px] font-black text-slate-500 mb-4 uppercase tracking-widest">System Messages</h2>
                    <div id="log" class="text-xs font-mono text-emerald-400/80 h-32 overflow-y-auto space-y-1">
                        <div>> Bridge system initialized...</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        const logEl = document.getElementById('log');

        function addLog(msg) {
            const div = document.createElement('div');
            div.innerText = `> ${msg}`;
            logEl.prepend(div);
            if (logEl.childNodes.length > 20) logEl.lastChild.remove();
        }
        
        socket.on('connect', () => {
            const st = document.getElementById('status');
            st.innerText = 'CONNECTED';
            st.className = 'px-4 py-1.5 rounded-full bg-emerald-500/10 text-emerald-500 border border-emerald-500/20 text-xs font-bold self-center tracking-widest uppercase';
            addLog('WebSocket Connection Established');
        });

        socket.on('disconnect', () => {
            const st = document.getElementById('status');
            st.innerText = 'DISCONNECTED';
            st.className = 'px-4 py-1.5 rounded-full bg-red-500/10 text-red-500 border border-red-500/20 text-xs font-bold self-center tracking-widest uppercase';
            addLog('WebSocket Connection Lost');
        });

        socket.on('robot_data', function(msg) {
            document.getElementById('curX').innerText = msg.x.toFixed(3);
            document.getElementById('curY').innerText = msg.y.toFixed(3);
            document.getElementById('curYaw').innerText = msg.yaw.toFixed(3);
            document.getElementById('curSpeed').innerText = (msg.speed || 0).toFixed(3);
        });

        // รับค่าอัปเดตความเร็วจาก Python (เมื่อ Topic เปลี่ยนค่า)
        socket.on('speed_update', function(msg) {
            const val = msg.max_speed;
            document.getElementById('max_speed').value = val;
            document.getElementById('speed-indicator').innerText = `Current: ${val} m/s`;
            addLog(`Max Speed updated from topic: ${val} m/s`);
        });

        function manualMove(vx, vy) { socket.emit('command', { type: 'V', data: `${vx}:${vy}` }); }
        function sendZero() { socket.emit('command', { type: 'Z', data: '' }); addLog('Reset Odometry command sent'); }
        
        function sendMaxSpeed() { 
            const s = document.getElementById('max_speed').value;
            socket.emit('command', { type: 'S', data: `${s}` }); 
            document.getElementById('speed-indicator').innerText = `Current: ${s} m/s`;
            addLog(`Max Speed manual update: ${s} m/s`);
        }

        function sendPID(id) {
            let p, i, d;
            if(id === 6) { p = document.getElementById('kp_p').value; i = document.getElementById('ki_p').value; d = document.getElementById('kd_p').value; }
            else { p = document.getElementById('kp_y').value; i = document.getElementById('ki_y').value; d = document.getElementById('kd_y').value; }
            socket.emit('command', { type: 'P', data: `${id}:${p}:${i}:${d}` });
            addLog(`PID Gain (ID:${id}) updated: P:${p} I:${i} D:${d}`);
        }
    </script>
</body>
</html>
"""

class RobotBridgeNode(Node):
    def __init__(self):
        super().__init__('robot_web_bridge')
        self.ser = None
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # ROS 2 Subscribers
        self.manual_vel_sub = self.create_subscription(Twist, '/cmd_vel_manual', self.manual_vel_callback, 10)
        self.vel_sub = self.create_subscription(Twist, '/cmd_vel', self.vel_callback, 10)
        self.goal_sub = self.create_subscription(Pose2D, '/cmd_goal', self.goal_callback, 10)
        self.reset_sub = self.create_subscription(Empty, '/cmd_reset_odom', lambda msg: self.send_to_robot('Z', ""), 10)
        self.pos_pid_sub = self.create_subscription(Float32MultiArray, '/cmd_pos_pid', self.pos_pid_callback, 10)
        self.yaw_pid_sub = self.create_subscription(Float32MultiArray, '/cmd_yaw_pid', self.yaw_pid_callback, 10)
        
        # Subscriber สำหรับ Max Speed (เพื่อ Monitor ดูการเปลี่ยนแปลง)
        self.speed_sub = self.create_subscription(Float32, '/cmd_max_speed', self.speed_limit_callback, 10)
        
        self.last_x, self.last_y, self.last_time = 0.0, 0.0, time.time()
        self.current_speed = 0.0
        
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            self.get_logger().info(f"BRIDGE: Connected to {SERIAL_PORT}")
        except Exception as e:
            self.get_logger().error(f"BRIDGE: Serial error: {e}")

        self.serial_thread = threading.Thread(target=self.serial_loop)
        self.serial_thread.daemon = True
        self.serial_thread.start()

    # --- Callbacks ---

    def speed_limit_callback(self, msg):
        """ รับค่าจาก Topic /cmd_max_speed แล้วส่งไปที่หน้าเว็บเพื่อแสดงผล """
        val = round(float(msg.data), 2)
        self.get_logger().info(f"MONITOR: Speed Limit changed to {val}")
        # ส่งไปให้ STM32
        self.send_to_robot('S', f"{val}")
        # ส่งไปอัปเดตหน้า Dashboard
        socketio.emit('speed_update', {'max_speed': val})

    def vel_callback(self, msg): self.send_to_robot('V', f"{round(msg.linear.x, 3)}:{round(msg.linear.y, 3)}")
    def manual_vel_callback(self, msg): self.send_to_robot('V', f"{msg.linear.x}:{msg.linear.y}")
    def goal_callback(self, msg): self.send_to_robot('G', f"{msg.x}:{msg.y}:{msg.theta}")
    def pos_pid_callback(self, msg): 
        if len(msg.data) >= 3: self.send_to_robot('P', f"6:{msg.data[0]}:{msg.data[1]}:{msg.data[2]}")
    def yaw_pid_callback(self, msg): 
        if len(msg.data) >= 3: self.send_to_robot('P', f"5:{msg.data[0]}:{msg.data[1]}:{msg.data[2]}")

    def send_to_robot(self, cmd_type, data):
        if self.ser and self.ser.is_open:
            full_cmd = f"{cmd_type}{data}\n"
            try:
                self.ser.write(full_cmd.encode())
                if cmd_type != 'V': # ไม่ Log ความเร็วเดินวนลูปเพื่อกัน Terminal ล้น
                    self.get_logger().info(f"SERIAL SEND: {full_cmd.strip()}")
            except Exception as e: self.get_logger().error(f"Serial write error: {e}")

    def serial_loop(self):
        while rclpy.ok():
            if self.ser and self.ser.in_waiting > 0:
                try:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith("POS:"):
                        parts = line.split(':')
                        if len(parts) >= 4:
                            x, y, yaw = float(parts[1]), float(parts[2]), float(parts[3])
                            now = time.time(); dt = now - self.last_time
                            if dt > 0: self.current_speed = math.sqrt((x - self.last_x)**2 + (y - self.last_y)**2) / dt
                            self.last_x, self.last_y, self.last_time = x, y, now
                            self.publish_odom_and_tf(x, y, yaw)
                            socketio.emit('robot_data', {'x': x, 'y': y, 'yaw': yaw, 'speed': self.current_speed})
                except Exception: pass
            time.sleep(0.005)

    def publish_odom_and_tf(self, x, y, yaw):
        now_msg = self.get_clock().now().to_msg(); q = self.euler_to_quaternion(0, 0, yaw)
        t = TransformStamped(); t.header.stamp = now_msg; t.header.frame_id = 'odom'; t.child_frame_id = 'base_link'
        t.transform.translation.x = x; t.transform.translation.y = y; t.transform.translation.z = 0.0; t.transform.rotation = q
        self.tf_broadcaster.sendTransform(t)
        odom = Odometry(); odom.header.stamp = now_msg; odom.header.frame_id = 'odom'; odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = x; odom.pose.pose.position.y = y; odom.pose.pose.orientation = q; self.odom_pub.publish(odom)

    def euler_to_quaternion(self, roll, pitch, yaw):
        cy, sy, cp, sp, cr, sr = math.cos(yaw*0.5), math.sin(yaw*0.5), math.cos(pitch*0.5), math.sin(pitch*0.5), math.cos(roll*0.5), math.sin(roll*0.5)
        q = Quaternion(); q.w = cr*cp*cy + sr*sp*sy; q.x = sr*cp*cy - cr*sp*sy; q.y = cr*sp*cy + sr*cp*sy; q.z = cr*cp*sy - sr*sp*cy; return q

ros_node = None
@app.route('/')
def index(): return render_template_string(HTML_TEMPLATE)
@socketio.on('command')
def handle_command(msg):
    global ros_node
    if ros_node: ros_node.send_to_robot(msg['type'], msg['data'])

def main():
    global ros_node; rclpy.init(); ros_node = RobotBridgeNode()
    ros_thread = threading.Thread(target=lambda: rclpy.spin(ros_node)); ros_thread.daemon = True; ros_thread.start()
    try: socketio.run(app, host='0.0.0.0', port=WEB_PORT, debug=False, log_output=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt: pass
    finally: ros_node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()