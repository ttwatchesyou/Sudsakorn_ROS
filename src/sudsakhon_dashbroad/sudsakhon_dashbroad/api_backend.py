#!/usr/bin/env python3
"""
Sudsakhon Bridge  v3
Flask REST API + ROS2 node
เพิ่ม: wheel speed คำนวณจาก Mecanum/Omni kinematics
"""

import math
import threading
import os
import json
import subprocess

import rclpy
from rclpy.node import Node
from flask import Flask, jsonify, request
from flask_cors import CORS

from std_msgs.msg import Float32MultiArray, Float32, Empty, Int32, Int32MultiArray, String
from geometry_msgs.msg import Pose2D, Twist
from nav_msgs.msg import Odometry

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PORT = 8001
STORAGE_FILE = os.path.expanduser("~/monitored_services.json")
DEFAULT_SERVICES = ["sudsakhon_tf.service", "sudsakhon_odom.service"]

# Mecanum wheel geometry (เมตร) — ปรับตามหุ่นจริง
# L = half wheelbase (ระยะ center → ซ้าย/ขวา)
# W = half track    (ระยะ center → หน้า/หลัง)
WHEEL_L = 0.15   # เมตร
WHEEL_W = 0.15   # เมตร

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────
robot_state = {
    # odometry (velocity)
    "linear_x":  0.0,
    "linear_y":  0.0,
    "angular_z": 0.0,
    # odometry (pose)
    "pose_x":    0.0,
    "pose_y":    0.0,
    "yaw":       0.0,
    # wheel speeds (คำนวณจาก mecanum inverse kinematics)
    "wheels": {
        "fl": {"speed": 0.0},   # front-left
        "fr": {"speed": 0.0},   # front-right
        "rl": {"speed": 0.0},   # rear-left
        "rr": {"speed": 0.0},   # rear-right
    },
    # program state
    "program_color": -1,
    "program_game":   0,
    # automation state
    "control_states": [0, 0, 0, 0],
    # services
    "monitored_services": [],
    # /detected_objects topic (String)
    "detected_objects":     "",          # ข้อความล่าสุด
    "detected_objects_log": [],          # history สูงสุด 200 รายการ [{ts, msg}]
}

_ros_node = None


# ─────────────────────────────────────────────────────────────────────────────
# MECANUM INVERSE KINEMATICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_wheel_speeds(vx: float, vy: float, wz: float) -> dict:
    """
    Mecanum/Omni inverse kinematics
    vx  = linear X (m/s, forward)
    vy  = linear Y (m/s, strafe left)
    wz  = angular Z (rad/s, CCW positive)

    สูตร standard mecanum:
      fl =  vx - vy - (L+W)*wz
      fr =  vx + vy + (L+W)*wz
      rl =  vx + vy - (L+W)*wz
      rr =  vx - vy + (L+W)*wz
    """
    lw = WHEEL_L + WHEEL_W
    fl = round( vx - vy - lw * wz, 3)
    fr = round( vx + vy + lw * wz, 3)
    rl = round( vx + vy - lw * wz, 3)
    rr = round( vx - vy + lw * wz, 3)
    return {
        "fl": {"speed": abs(fl), "raw": fl},
        "fr": {"speed": abs(fr), "raw": fr},
        "rl": {"speed": abs(rl), "raw": rl},
        "rr": {"speed": abs(rr), "raw": rr},
    }


# ─────────────────────────────────────────────────────────────────────────────
# SERVICE PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────
def load_services():
    if os.path.exists(STORAGE_FILE):
        try:
            with open(STORAGE_FILE) as f:
                data = json.load(f)
                return data if isinstance(data, list) else list(DEFAULT_SERVICES)
        except Exception:
            pass
    return list(DEFAULT_SERVICES)


def save_services():
    try:
        with open(STORAGE_FILE, "w") as f:
            json.dump(robot_state["monitored_services"], f)
    except Exception as e:
        print(f"[bridge] save_services error: {e}")


robot_state["monitored_services"] = load_services()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _node():
    if _ros_node is None:
        from flask import abort
        abort(503, description="ROS node not initialised yet")
    return _ros_node


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return jsonify({"status": "online", "robot": "Sudsakhon"})


@app.route("/api/telemetry", methods=["GET"])
def get_telemetry():
    return jsonify(robot_state)


@app.route("/api/detected_objects", methods=["GET"])
def get_detected_objects():
    """ส่ง detected_objects ล่าสุด + log ย้อนหลัง"""
    return jsonify({
        "latest": robot_state["detected_objects"],
        "log":    robot_state["detected_objects_log"],
    })


@app.route("/api/mission/status", methods=["GET"])
def get_mission_status():
    """ดึงสถานะภารกิจปัจจุบันและ Log การตรวจจับวัตถุ"""
    return jsonify({
        "current_game": robot_state["program_game"],
        "team_color": "RED" if robot_state["program_color"] == 0 else "BLUE" if robot_state["program_color"] == 1 else "NONE",
        # "latest": robot_state["detected_objects"],
        # "logs": robot_state["detected_objects_log"][-10:] # ส่งไปแค่ 10 รายการล่าสุดเพื่อความไว
    })


@app.route("/api/services", methods=["GET"])
def get_services():
    results = []
    for s in robot_state["monitored_services"]:
        try:
            status = subprocess.check_output(["systemctl", "is-active", s]).decode().strip()
        except Exception:
            status = "inactive"
        results.append({"name": s, "status": status})
    return jsonify(results)


@app.route("/api/logs/<service_name>", methods=["GET"])
def get_logs(service_name):
    try:
        result = subprocess.check_output(
            ["journalctl", "-u", service_name, "-n", "50", "--no-pager"]
        ).decode("utf-8")
        return jsonify({"logs": result})
    except Exception:
        return jsonify({"logs": "Could not fetch logs."})


@app.route("/api/add_service", methods=["POST"])
def add_service():
    new_svc = request.json.get("service")
    if new_svc and new_svc not in robot_state["monitored_services"]:
        robot_state["monitored_services"].append(new_svc)
        save_services()
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400


@app.route("/api/remove_service", methods=["POST"])
def remove_service():
    svc = request.json.get("service")
    if svc in robot_state["monitored_services"]:
        robot_state["monitored_services"].remove(svc)
        save_services()
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400


@app.route("/api/control", methods=["POST"])
def control():
    data = request.json
    service, action = data.get("service"), data.get("action")
    if service in robot_state["monitored_services"]:
        def execute():
            subprocess.run(["sudo", "systemctl", "reset-failed", service], check=False)
            subprocess.run(["sudo", "systemctl", action, service], check=False)
        threading.Thread(target=execute).start()
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 404


@app.route("/api/restart", methods=["POST"])
def restart_service():
    data = request.json
    service = data.get("service")
    if service in robot_state["monitored_services"]:
        def execute():
            subprocess.run(["sudo", "systemctl", "reset-failed", service], check=False)
            subprocess.run(["sudo", "systemctl", "restart", service], check=False)
        threading.Thread(target=execute).start()
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 404


# ── Tuner commands ────────────────────────────────────────────────────────────

@app.route("/api/cmd/goal", methods=["POST"])
def cmd_goal():
    d = request.json or {}
    try:
        x, y, theta = float(d["x"]), float(d["y"]), float(d["theta"])
    except (KeyError, ValueError) as e:
        return jsonify({"status": "error", "detail": str(e)}), 400
    _node().send_goal(x, y, theta)
    return jsonify({"status": "ok", "x": x, "y": y, "theta": theta})


@app.route("/api/cmd/stop", methods=["POST"])
def cmd_stop():
    d = request.json or {}
    try:
        x     = float(d.get("x",     robot_state["pose_x"]))
        y     = float(d.get("y",     robot_state["pose_y"]))
        theta = float(d.get("theta", robot_state["yaw"]))
    except (ValueError, TypeError) as e:
        return jsonify({"status": "error", "detail": str(e)}), 400
    _node().send_goal(x, y, theta)
    return jsonify({"status": "ok", "stopped_at": {"x": x, "y": y, "theta": theta}})


@app.route("/api/cmd/reset_odom", methods=["POST"])
def cmd_reset_odom():
    _node().send_reset()
    robot_state.update({"pose_x": 0.0, "pose_y": 0.0, "yaw": 0.0})
    robot_state["wheels"] = compute_wheel_speeds(0, 0, 0)
    return jsonify({"status": "ok"})


@app.route("/api/cmd/max_speed", methods=["POST"])
def cmd_max_speed():
    d = request.json or {}
    try:
        speed = float(d["speed"])
    except (KeyError, ValueError) as e:
        return jsonify({"status": "error", "detail": str(e)}), 400
    _node().send_max_speed(speed)
    return jsonify({"status": "ok", "speed": speed})


@app.route("/api/cmd/pos_pid", methods=["POST"])
def cmd_pos_pid():
    d = request.json or {}
    try:
        kp, ki, kd = float(d["kp"]), float(d["ki"]), float(d["kd"])
    except (KeyError, ValueError) as e:
        return jsonify({"status": "error", "detail": str(e)}), 400
    _node().send_pos_pid(kp, ki, kd)
    return jsonify({"status": "ok", "kp": kp, "ki": ki, "kd": kd})


@app.route("/api/cmd/yaw_profile", methods=["POST"])
def cmd_yaw_profile():
    d = request.json or {}
    try:
        max_rpm, brake_rad, fine_kp = float(d["max_rpm"]), float(d["brake_rad"]), float(d["fine_kp"])
    except (KeyError, ValueError) as e:
        return jsonify({"status": "error", "detail": str(e)}), 400
    _node().send_yaw_pid(max_rpm, brake_rad, fine_kp)
    return jsonify({"status": "ok"})


@app.route("/api/cmd/trap_profile", methods=["POST"])
def cmd_trap_profile():
    d = request.json or {}
    try:
        accel, decel, min_v = float(d["accel"]), float(d["decel"]), float(d["min_v"])
    except (KeyError, ValueError) as e:
        return jsonify({"status": "error", "detail": str(e)}), 400
    _node().send_trap_profile(accel, decel, min_v)
    return jsonify({"status": "ok"})


# ── Control center ────────────────────────────────────────────────────────────

@app.route("/api/cmd/program_color", methods=["POST"])
def cmd_program_color():
    d = request.json or {}
    try:
        color = int(d["color"])
        assert color in (0, 1)
    except (KeyError, ValueError, AssertionError):
        return jsonify({"status": "error", "detail": "color must be 0 or 1"}), 400
    robot_state["program_color"] = color
    _node().send_program_color(color)
    return jsonify({"status": "ok", "color": "RED" if color == 0 else "BLUE"})


@app.route("/api/cmd/program_game", methods=["POST"])
def cmd_program_game():
    d = request.json or {}
    try:
        game = int(d["game"])
        assert 1 <= game <= 10
    except (KeyError, ValueError, AssertionError):
        return jsonify({"status": "error", "detail": "game must be 1-10"}), 400
    robot_state["program_game"] = game
    _node().send_program_game(game)
    return jsonify({"status": "ok", "game": game})


@app.route("/api/cmd/program_command", methods=["POST"])
def cmd_program_command():
    d = request.json or {}
    try:
        command = int(d["command"])
        assert command in (1, 2)
    except (KeyError, ValueError, AssertionError):
        return jsonify({"status": "error", "detail": "command must be 1 (start) or 2 (reset)"}), 400
    _node().send_program_command(command)
    return jsonify({"status": "ok", "command": "START" if command == 1 else "RESET"})


@app.route("/api/cmd/estop", methods=["POST"])
def cmd_estop():
    _node().send_estop()
    robot_state["program_game"] = 0
    robot_state["wheels"] = compute_wheel_speeds(0, 0, 0)
    return jsonify({"status": "ok"})


@app.route("/api/cmd/teleop_vel", methods=["POST"])
def cmd_teleop_vel():
    d = request.json or {}
    try:
        vx = float(d.get("vx", 0.0))
        vy = float(d.get("vy", 0.0))
    except (ValueError, TypeError) as e:
        return jsonify({"status": "error", "detail": str(e)}), 400
    _node().send_teleop_vel(vx, vy)
    return jsonify({"status": "ok", "vx": vx, "vy": vy})


@app.route("/api/cmd/lift", methods=["POST"])
def cmd_lift():
    d = request.json or {}
    try:
        lift  = int(d["lift"])
        state = str(d["state"])
        assert lift in (1, 2) and state in ("up", "down")
    except (KeyError, ValueError, AssertionError):
        return jsonify({"status": "error", "detail": "lift must be 1|2, state must be up|down"}), 400
    idx   = lift - 1
    value = 1 if state == "up" else 0
    robot_state["control_states"][idx] = value
    _node().send_control_states(robot_state["control_states"])
    return jsonify({"status": "ok", "lift": lift, "state": state})


@app.route("/api/cmd/slider", methods=["POST"])
def cmd_slider():
    d = request.json or {}
    action = str(d.get("action", ""))
    if action not in ("in", "out"):
        return jsonify({"status": "error", "detail": "action must be 'in' or 'out'"}), 400
    angle = 0 if action == "in" else 180
    _node().send_servo(1, angle)
    return jsonify({"status": "ok", "action": action, "angle": angle})


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 NODE
# ─────────────────────────────────────────────────────────────────────────────
class SudsakhonBridge(Node):
    def __init__(self):
        super().__init__("sudsakhon_bridge_node")

        self.pub_pos_pid      = self.create_publisher(Float32MultiArray, "/cmd_pos_pid",               10)
        self.pub_yaw_pid      = self.create_publisher(Float32MultiArray, "/cmd_yaw_pid",               10)
        self.pub_trap         = self.create_publisher(Float32MultiArray, "/cmd_trap_profile",          10)
        self.pub_goal         = self.create_publisher(Pose2D,            "/cmd_goal",                  10)
        self.pub_max_speed    = self.create_publisher(Float32,           "/cmd_max_speed",             10)
        self.pub_reset        = self.create_publisher(Empty,             "/cmd_reset_odom",            10)
        self.pub_cmd_vel      = self.create_publisher(Twist,             "/cmd_vel",                   10)
        self.pub_prog_color   = self.create_publisher(Int32,             "/Program/Color",             10)
        self.pub_prog_game    = self.create_publisher(Int32,             "/Program/Game",              10)
        self.pub_prog_command = self.create_publisher(Int32,             "/Program/Command",           10)
        self.pub_ctrl_states  = self.create_publisher(Int32MultiArray,   "/automation/control_states", 10)
        self.pub_servo        = self.create_publisher(Int32MultiArray,   "/automation/servo",          10)

        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.create_subscription(String, "/detected_objects", self._detected_cb, 10)
        self.get_logger().info("SudsakhonBridge v3 ready")

    def _odom_cb(self, msg: Odometry):
        vx = round(msg.twist.twist.linear.x,  3)
        vy = round(msg.twist.twist.linear.y,  3)
        wz = round(msg.twist.twist.angular.z, 3)

        robot_state["linear_x"]  = vx
        robot_state["linear_y"]  = vy
        robot_state["angular_z"] = wz

        # pose
        px  = msg.pose.pose.position.x
        py  = msg.pose.pose.position.y
        qz  = msg.pose.pose.orientation.z
        qw  = msg.pose.pose.orientation.w
        yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
        robot_state["pose_x"] = round(px,  3)
        robot_state["pose_y"] = round(py,  3)
        robot_state["yaw"]    = round(yaw, 4)

        # ── คำนวณ wheel speeds จาก mecanum kinematics ──
        robot_state["wheels"] = compute_wheel_speeds(vx, vy, wz)

    def _detected_cb(self, msg: String):
        import datetime
        text = msg.data.strip()
        robot_state["detected_objects"] = text
        entry = {
            "ts":  datetime.datetime.now().strftime("%H:%M:%S"),
            "msg": text,
        }
        log = robot_state["detected_objects_log"]
        log.append(entry)
        if len(log) > 200:
            robot_state["detected_objects_log"] = log[-200:]

    # ── publishers ────────────────────────────────────────────────────────────
    def send_pos_pid(self, kp, ki, kd):
        msg = Float32MultiArray(); msg.data = [float(kp), float(ki), float(kd)]
        self.pub_pos_pid.publish(msg)

    def send_yaw_pid(self, max_rpm, brake_rad, fine_kp):
        msg = Float32MultiArray(); msg.data = [float(max_rpm), float(brake_rad), float(fine_kp)]
        self.pub_yaw_pid.publish(msg)

    def send_trap_profile(self, accel, decel, min_v):
        msg = Float32MultiArray(); msg.data = [float(accel), float(decel), float(min_v)]
        self.pub_trap.publish(msg)

    def send_goal(self, x, y, theta):
        msg = Pose2D(); msg.x = float(x); msg.y = float(y); msg.theta = float(theta)
        self.pub_goal.publish(msg)

    def send_max_speed(self, speed):
        msg = Float32(); msg.data = float(speed)
        self.pub_max_speed.publish(msg)

    def send_reset(self):
        self.pub_reset.publish(Empty())

    def send_program_color(self, color):
        msg = Int32(); msg.data = int(color); self.pub_prog_color.publish(msg)

    def send_program_game(self, game):
        msg = Int32(); msg.data = int(game); self.pub_prog_game.publish(msg)

    def send_program_command(self, command):
        msg = Int32(); msg.data = int(command); self.pub_prog_command.publish(msg)

    def send_estop(self):
        self.pub_cmd_vel.publish(Twist())
        msg = Int32(); msg.data = 0; self.pub_prog_game.publish(msg)

    def send_teleop_vel(self, vx, vy):
        msg = Twist(); msg.linear.x = float(vx); msg.linear.y = float(vy)
        self.pub_cmd_vel.publish(msg)

    def send_control_states(self, states):
        msg = Int32MultiArray(); msg.data = [int(s) for s in states]
        self.pub_ctrl_states.publish(msg)

    def send_servo(self, servo_id, angle):
        msg = Int32MultiArray(); msg.data = [int(servo_id), int(angle)]
        self.pub_servo.publish(msg)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    global _ros_node
    threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0", port=PORT,
            debug=False, threaded=True, use_reloader=False,
        ),
        daemon=True,
    ).start()
    print(f"[bridge] Flask listening on :{PORT}")
    rclpy.init(args=args)
    _ros_node = SudsakhonBridge()
    try:
        rclpy.spin(_ros_node)
    except KeyboardInterrupt:
        pass
    finally:
        _ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()