"""
yolo_depth_pub.py  —  simplified parameter set
ลด parameter เหลือเฉพาะที่ต้องปรับจริง ๆ สำหรับการตรวจจับ
"""

import os
import math
import time
import threading
import queue
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from std_msgs.msg import String, Bool

import cv2
import numpy as np
from openni import openni2
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────
#  Constants (ค่าที่ไม่จำเป็นต้องปรับ — hardcode ไว้ได้เลย)
# ─────────────────────────────────────────────────────────────
_DEPTH_BIN_WIDTH_MM       = 10
_CLOSEST_PERCENTILE       = 5.0
_DEPTH_SURFACE_PERCENTILE = 15.0
_DEPTH_CAND_RATIO_MIN     = 0.7
_DEPTH_CAND_RATIO_MAX     = 1.4
_MIN_CLUSTER_PIXELS       = 60
_MIN_CLUSTER_FRACTION     = 0.01
_BIN_NEIGHBOR_WIDTH       = 1
_TRIM_RATIO               = 0.2
_ROI_SHRINK_RATIO         = 0.12      # bottle ใช้ 0.06
_BOTTOM_BAND_RATIO        = 0.28
_BOTTOM_BAND_PERCENTILE   = 12.0
_EDGE_MAX_SIDE_PX         = 160
_EDGE_CANNY_LOW           = 50
_EDGE_CANNY_HIGH          = 160
_EDGE_DILATE_ITER         = 1
_EDGE_MIN_PIXELS          = 20
_COLOR_SHRINK_RATIO       = 0.15
_COLOR_MAX_SIDE_PX        = 96
_STABLE_KEY_GRID_PX       = 40
_BBOX_EMA_ALPHA           = 0.4
_VIEW_BBOX_EMA_ALPHA      = 0.25
_VIEW_BBOX_RESET_IOU      = 0.03
_CAMERA_BUFFER_SIZE       = 1
_CAMERA_GRAB_FLUSH        = 1
_QOS_DEPTH                = 1
_ANCHOR_RATIO_MIN         = 0.75
_ANCHOR_RATIO_MAX         = 1.35
_ANCHOR_MAX_ABS_DIFF_MM   = 450
_ANCHOR_SOFT_TOL_MM       = 120
_ANCHOR_BLEND_WEIGHT      = 0.65
_SIZE_CALIB_ALPHA         = 0.25
_SIZE_CALIB_MIN_SCALE     = 0.6
_SIZE_CALIB_MAX_SCALE     = 1.6


# ─────────────────────────────────────────────────────────────
#  High-Performance MJPEG Stream
# ─────────────────────────────────────────────────────────────
class _StreamState:
    def __init__(self):
        self._raw_q: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._jpeg: bytes = b""
        self._debug_lines: list[str] = []
        self._frame_count: int = 0
        self._encode_fps: float = 0.0
        self._client_lock = threading.Lock()
        self._clients: list[threading.Condition] = []
        self._jpeg_quality: int = 75
        threading.Thread(target=self._encode_loop, daemon=True).start()

    def push_raw(self, bgr: np.ndarray, debug_lines: list[str], quality: int = 75):
        self._jpeg_quality = quality
        with self._lock:
            self._debug_lines = debug_lines
        try:
            self._raw_q.put_nowait((bgr.copy(), debug_lines))
        except queue.Full:
            try:
                self._raw_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._raw_q.put_nowait((bgr.copy(), debug_lines))
            except queue.Full:
                pass

    def register_client(self) -> threading.Condition:
        cond = threading.Condition()
        with self._client_lock:
            self._clients.append(cond)
        return cond

    def unregister_client(self, cond: threading.Condition):
        with self._client_lock:
            try:
                self._clients.remove(cond)
            except ValueError:
                pass

    def client_count(self) -> int:
        with self._client_lock:
            return len(self._clients)

    def get_latest_jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg

    def get_debug(self):
        with self._lock:
            return list(self._debug_lines), self._frame_count, self._encode_fps

    def wait_for_new_frame(self, cond: threading.Condition, timeout: float = 2.0):
        with cond:
            cond.wait(timeout=timeout)
        with self._lock:
            return self._jpeg if self._jpeg else None

    def _encode_loop(self):
        t_last = time.monotonic()
        while True:
            try:
                bgr, dbg = self._raw_q.get(timeout=1.0)
            except queue.Empty:
                continue
            q = max(10, min(100, self._jpeg_quality))
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
            if not ok:
                continue
            now = time.monotonic()
            fps = 1.0 / max(now - t_last, 1e-6)
            t_last = now
            with self._lock:
                self._jpeg = buf.tobytes()
                self._debug_lines = dbg
                self._frame_count += 1
                self._encode_fps = fps
            with self._client_lock:
                for cond in list(self._clients):
                    with cond:
                        cond.notify_all()


_stream = _StreamState()


class _MjpegHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_page()
        elif self.path == "/stream.mjpg":
            self._serve_mjpeg()
        elif self.path == "/debug":
            self._serve_debug()
        else:
            self.send_error(404)

    def _serve_page(self):
        html = _build_html()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=--FRAME")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        cond = _stream.register_client()
        try:
            data = _stream.get_latest_jpeg()
            if data:
                self._send_jpeg(data)
            while True:
                data = _stream.wait_for_new_frame(cond)
                if data:
                    self._send_jpeg(data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            _stream.unregister_client(cond)

    def _send_jpeg(self, data: bytes):
        self.wfile.write(
            b"--FRAME\r\nContent-Type: image/jpeg\r\n"
            + f"Content-Length: {len(data)}\r\n\r\n".encode()
            + data + b"\r\n"
        )
        self.wfile.flush()

    def _serve_debug(self):
        lines, fc, fps = _stream.get_debug()
        body = "\n".join(lines) + f"\nframe={fc}  encode_fps={fps:.1f}"
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)


def _build_html() -> bytes:
    html = r"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>YOLO Depth — Live View</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0d0f14; color: #e2e8f0;
    font-family: 'JetBrains Mono','Fira Code',monospace;
    display: flex; flex-direction: column; align-items: center;
    min-height: 100vh; padding: 16px; }
  header { width:100%; max-width:900px; display:flex; align-items:center;
    gap:12px; padding:12px 0 20px; }
  .dot { width:10px; height:10px; border-radius:50%; background:#22c55e;
    animation:blink 1s step-start infinite; }
  @keyframes blink { 50%{opacity:0} }
  header h1 { font-size:1.1rem; letter-spacing:0.08em; color:#94a3b8; }
  .stats { font-size:0.72rem; color:#475569; margin-left:auto; display:flex; gap:14px; }
  .stats span { color:#38bdf8; }
  .stream-box { width:100%; max-width:900px; border:1px solid #1e293b;
    border-radius:8px; overflow:hidden; background:#020408; position:relative; }
  .stream-box img { width:100%; height:auto; display:block; }
  .overlay-label { position:absolute; top:10px; left:10px;
    background:rgba(0,0,0,0.55); padding:3px 8px; border-radius:4px;
    font-size:0.7rem; color:#64748b; letter-spacing:0.05em; }
  .debug-panel { width:100%; max-width:900px; margin-top:16px;
    background:#0a0c10; border:1px solid #1e293b; border-radius:8px; padding:14px 16px; }
  .debug-panel h2 { font-size:0.7rem; color:#475569; letter-spacing:0.12em;
    margin-bottom:10px; text-transform:uppercase; }
  #debug-text { font-size:0.8rem; line-height:1.7; color:#38bdf8;
    white-space:pre-wrap; min-height:60px; }
  .det-row { color:#4ade80; } .err-row { color:#f87171; } .info-row { color:#94a3b8; }
</style>
</head>
<body>
<header>
  <div class="dot"></div>
  <h1>YOLO DEPTH PUBLISHER — LIVE</h1>
  <div class="stats">
    encode: <span id="enc-fps">--</span> fps &nbsp;|&nbsp;
    browser: <span id="br-fps">--</span> fps
  </div>
</header>
<div class="stream-box">
  <img id="stream" src="/stream.mjpg" alt="live stream">
  <div class="overlay-label">MJPEG / LIVE</div>
</div>
<div class="debug-panel">
  <h2>📡 Debug Output</h2>
  <pre id="debug-text">Connecting...</pre>
</div>
<script>
  let lastTime = performance.now(), frameCount = 0;
  const img = document.getElementById('stream');
  const brFpsEl = document.getElementById('br-fps');
  const encFpsEl = document.getElementById('enc-fps');
  const debugEl = document.getElementById('debug-text');
  img.addEventListener('load', () => {
    frameCount++;
    const now = performance.now(), dt = (now - lastTime) / 1000;
    if (dt >= 1.0) { brFpsEl.textContent = (frameCount/dt).toFixed(1); frameCount=0; lastTime=now; }
  });
  img.addEventListener('error', () => { setTimeout(()=>{ img.src='/stream.mjpg?'+Date.now(); },1000); });
  async function pollDebug() {
    try {
      const r = await fetch('/debug', { cache:'no-store' });
      const text = await r.text();
      const match = text.match(/encode_fps=([\d]+\.[\d]+)/);
      if (match) encFpsEl.textContent = parseFloat(match[1]).toFixed(1);
      let html = '';
      for (const line of text.split('\n')) {
        if (line.startsWith('det:')) html += `<span class="det-row">${line}</span>\n`;
        else if (line.startsWith('ERR')||line.startsWith('WARN')) html += `<span class="err-row">${line}</span>\n`;
        else html += `<span class="info-row">${line}</span>\n`;
      }
      debugEl.innerHTML = html;
    } catch(e) { debugEl.textContent = 'Connection lost — retrying...'; }
    setTimeout(pollDebug, 500);
  }
  pollDebug();
</script>
</body>
</html>"""
    return html.encode()


def start_mjpeg_server(host: str = "0.0.0.0", port: int = 9090):
    server = ThreadingHTTPServer((host, port), _MjpegHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ─────────────────────────────────────────────────────────────
#  ROS 2 Node
# ─────────────────────────────────────────────────────────────
class YoloDepthPublisher(Node):
    def __init__(self):
        super().__init__("yolo_depth_publisher")

        self._param_cache = {}
        self._declare_parameters()
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # Publisher
        self.publisher = self.create_publisher(String, "/detected_objects", _QOS_DEPTH)

        # ─── ADDED: Subscriber for Enable/Disable YOLO ───
        self._detection_enabled = True
        self.enable_sub = self.create_subscription(
            Bool, 
            "/detected_objects_enable", 
            self._enable_callback, 
            10
        )

        target_fps = float(self._p("target_fps"))
        period_s = max(0.001, 1.0 / target_fps) if target_fps > 0 else 0.1
        self.timer = self.create_timer(period_s, self.process_frame)

        # Camera
        self.cap = cv2.VideoCapture(int(self._p("camera_index")))
        self._configure_camera()

        # YOLO
        model_path = str(self._p("model_path"))
        if not model_path:
            raise RuntimeError("model_path is empty")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        self.model = YOLO(model_path)
        try:
            self.model.fuse()
        except Exception as exc:
            self.get_logger().warn(f"Could not fuse model: {exc}")
        self.target_class_ids = self._resolve_target_class_ids()

        # Astra depth camera
        self._init_openni()
        self.dev = openni2.Device.open_any()
        self.depth_stream = self.dev.create_depth_stream()
        self.depth_stream.start()
        try:
            self.dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        except Exception as exc:
            self.get_logger().warn(f"Depth-color registration failed: {exc}")

        # State
        self._last_depth_by_key: dict = {}
        self._size_scale_by_label: dict = {}
        self._stable_depth: dict = {}
        self._stable_bbox: dict = {}
        self._last_box_label = None
        self._last_box_bbox = None
        self._last_box_time_s = 0.0
        self._view_bbox: dict = {}

        # Web stream
        stream_port = int(self._p("stream_port"))
        start_mjpeg_server(port=stream_port)
        self.get_logger().info(f"MJPEG stream → http://0.0.0.0:{stream_port}/")

    # ── callback for enable/disable ────────────────────────────────

    def _enable_callback(self, msg: Bool):
        if self._detection_enabled != msg.data:
            self._detection_enabled = msg.data
            state_str = "ENABLED" if self._detection_enabled else "DISABLED"
            self.get_logger().info(f"YOLO Detection is now {state_str}")

    # ── parameter helpers ──────────────────────────────────────────

    def _on_set_parameters(self, params):
        for p in params:
            self._param_cache[p.name] = p.value
        return SetParametersResult(successful=True)

    def _p(self, name, default=None):
        if name in self._param_cache:
            return self._param_cache[name]
        v = self.get_parameter(name).value
        self._param_cache[name] = v
        return v if v is not None else default

    def _declare_parameters(self):
        # ── 1. Hardware / paths ────────────────────────────────────
        self.declare_parameter("openni_lib_path", "")          # path ของ libOpenNI2 (ถ้าไม่ได้ default)
        self.declare_parameter("camera_index",    0)            # index กล้อง RGB (ls /dev/video*)
        #self.declare_parameter("model_path",      "/home/ubuntu/oldmodel/best500.pt")
        self.declare_parameter("model_path",      "/home/ubuntu/oldmodel/new/best.pt")

        # ── 2. Camera resolution & FPS ────────────────────────────
        self.declare_parameter("frame_width",  640)
        self.declare_parameter("frame_height", 480)
        self.declare_parameter("target_fps",   30.0)           # FPS ของ ROS timer loop
        self.declare_parameter("camera_fps",   30.0)           # FPS ที่ขอจากกล้อง

        # ── 3. YOLO inference ─────────────────────────────────────
        self.declare_parameter("imgsz",   640)                  # ขนาดภาพที่ส่งเข้า YOLO
        self.declare_parameter("conf",    0.35)                 # confidence threshold ขั้นต่ำ
        self.declare_parameter("device",  "cpu")                # "cpu" | "cuda" | "0"
        self.declare_parameter("max_det", 300)                  # จำนวน detection สูงสุดต่อ frame

        # ── 4. Class ที่ต้องการตรวจจับ ─────────────────────────────
        self.declare_parameter("box_class_name",    "box")
        self.declare_parameter("bottle_class_name", "bottle")
        self.declare_parameter("allow_other_classes", False)    # True = detect class อื่นด้วย

        # ── 5. Confidence threshold แยกตาม class ──────────────────
        self.declare_parameter("box_conf_threshold",    0.45)
        self.declare_parameter("bottle_conf_threshold", 0.35)

        # ── 6. พื้นที่ bbox ขั้นต่ำ (กรอง noise เล็ก ๆ) ────────────
        self.declare_parameter("box_min_area_px",    1200)      # pixel² ขั้นต่ำของ box
        self.declare_parameter("bottle_min_area_px",  450)      # pixel² ขั้นต่ำของ bottle

        # ── 7. Aspect ratio ที่ยอมรับ ─────────────────────────────
        self.declare_parameter("box_aspect_ratio_min",    0.3)
        self.declare_parameter("box_aspect_ratio_max",    4.0)
        self.declare_parameter("bottle_aspect_ratio_min", 0.1)
        self.declare_parameter("bottle_aspect_ratio_max", 1.2)

        # ── 8. กรอง region กลางภาพ ────────────────────────────────
        self.declare_parameter("enable_center_region_filter", True)
        self.declare_parameter("center_region_width_ratio",   0.5)  # 0.5 = กลาง 50%
        self.declare_parameter("draw_center_region_guides",   True)

        # ── 9. Color filter (กล่องขาว) ────────────────────────────
        self.declare_parameter("enable_white_box_color_filter", True)
        self.declare_parameter("white_ratio_min", 0.25)         # สัดส่วน pixel ขาวขั้นต่ำ

        # ── 10. Depth range ที่สนใจ ────────────────────────────────
        self.declare_parameter("min_depth_mm", 1)               # ระยะใกล้สุด (mm)
        self.declare_parameter("max_depth_mm", 3000)            # ระยะไกลสุด (mm)

        # ── 11. Camera FOV (ใช้คำนวณขนาดวัตถุ) ───────────────────
        self.declare_parameter("hfov_deg", 60.0)
        self.declare_parameter("vfov_deg", 49.5)

        # ── 12. Size-based distance fusion (ขวด) ──────────────────
        self.declare_parameter("enable_bottle_size_distance", True)
        self.declare_parameter("known_bottle_width_mm",  70.0)  # ความกว้างจริงของขวด (mm)
        self.declare_parameter("known_bottle_height_mm", 200.0) # ความสูงจริงของขวด (mm)

        # ── 13. Depth stability ────────────────────────────────────
        self.declare_parameter("ema_alpha",    0.35)            # smoothing (0=ไม่เปลี่ยน, 1=ไม่ smooth)
        self.declare_parameter("max_jump_mm",  250.0)           # กระโดดระยะเกินนี้ถือว่า noise

        # ── 14. Box anti-flicker ───────────────────────────────────
        self.declare_parameter("box_hysteresis_ms", 1500.0)     # เวลาผ่อนปรน threshold หลังเห็น box
        self.declare_parameter("box_hold_ms",         450.0)    # เวลา hold bbox เมื่อ YOLO miss

        # ── 15. Bottle-anchor correction ──────────────────────────
        self.declare_parameter("enable_bottle_anchor",    True)
        self.declare_parameter("anchor_y_tolerance_px",  55)    # pixel ที่ยอมให้ box กับ bottle ต่างกัน

        # ── 16. Size calibration (ปิดไว้ก่อน) ─────────────────────
        self.declare_parameter("enable_size_calibration", False)
        self.declare_parameter("known_box_width_mm",  618.0)    # ความกว้างจริงของกล่อง (mm)
        self.declare_parameter("known_box_height_mm", 398.0)    # ความสูงจริงของกล่อง (mm)

        # ── 17. Web stream ─────────────────────────────────────────
        self.declare_parameter("stream_port",         9090)
        self.declare_parameter("stream_jpeg_quality",   75)     # 50–90 (ต่ำ=เร็ว, สูง=คมชัด)

    def _configure_camera(self):
        w = int(self._p("frame_width"))
        h = int(self._p("frame_height"))
        fps = float(self._p("camera_fps"))
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, _CAMERA_BUFFER_SIZE)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            if fps > 0:
                self.cap.set(cv2.CAP_PROP_FPS, fps)
        except Exception as exc:
            self.get_logger().warn(f"Camera config error: {exc}")

    def _init_openni(self):
        lib_path = str(self._p("openni_lib_path") or "")
        candidates = [lib_path] if lib_path else []
        env = os.getenv("OPENNI2_REDIST")
        if env:
            candidates.append(env)
        candidates += [
            "/home/ubuntu/ros2_astra_camera/astra_camera/openni2_redist/x64",
            "/home/ubuntu/ros2_astra_camera/astra_camera/openni2_redist/arm64",
            "/home/ubuntu/ros2_astra_camera/astra_camera/openni2_redist/arm",
        ]
        errors = []
        for path in candidates:
            if not path or not Path(path).exists():
                continue
            try:
                openni2.initialize(path)
                self.get_logger().info(f"OpenNI initialized: {path}")
                return
            except Exception as exc:
                errors.append(f"{path}: {exc}")
        try:
            openni2.initialize()
            self.get_logger().info("OpenNI initialized (default path)")
        except Exception as exc:
            raise RuntimeError(f"OpenNI init failed.\n" + "\n".join(errors)) from exc

    def _resolve_target_class_ids(self):
        box_name    = str(self._p("box_class_name")).strip().lower()
        bottle_name = str(self._p("bottle_class_name")).strip().lower()
        target = {n for n in [box_name, bottle_name] if n}
        names = self.model.names
        pairs = names.items() if isinstance(names, dict) else enumerate(names)
        ids = [int(cid) for cid, cn in pairs if str(cn).strip().lower() in target]
        if not ids:
            self.get_logger().warn("No matching class IDs — detecting all classes")
            return None
        return ids

    # ── helper methods ─────────────────────────────────────────────

    def _white_ratio(self, frame, x1, y1, x2, y2) -> float:
        shrink = _COLOR_SHRINK_RATIO
        w, h = max(0, x2-x1), max(0, y2-y1)
        sx, sy = int(w*shrink), int(h*shrink)
        rx1 = max(0, x1+sx); ry1 = max(0, y1+sy)
        rx2 = min(frame.shape[1], x2-sx); ry2 = min(frame.shape[0], y2-sy)
        if rx2 <= rx1 or ry2 <= ry1:
            rx1, ry1, rx2, ry2 = max(0,x1), max(0,y1), min(frame.shape[1],x2), min(frame.shape[0],y2)
        if rx2 <= rx1 or ry2 <= ry1:
            return 0.0
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return 0.0
        ms = max(16, _COLOR_MAX_SIDE_PX)
        rh, rw = roi.shape[:2]
        if max(rh, rw) > ms:
            sc = ms / max(rh, rw)
            roi = cv2.resize(roi, (max(1,int(rw*sc)), max(1,int(rh*sc))))
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = (hsv[:,:,1] <= 70) & (hsv[:,:,2] >= 130)
        return float(mask.mean())

    @staticmethod
    def _iou(a, b) -> float:
        ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
        iw = max(0, min(ax2,bx2)-max(ax1,bx1))
        ih = max(0, min(ay2,by2)-max(ay1,by1))
        inter = iw*ih
        if inter <= 0: return 0.0
        denom = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter/denom if denom > 0 else 0.0

    def _passes_filters(self, frame, label, conf, x1, y1, x2, y2) -> bool:
        ll = label.strip().lower()
        box_name    = str(self._p("box_class_name")).strip().lower()
        bottle_name = str(self._p("bottle_class_name")).strip().lower()
        bw = max(0, x2-x1); bh = max(0, y2-y1)
        area = bw*bh
        ar   = bw / (bh + 1e-6)

        if ll == box_name:
            # Hysteresis: ผ่อนปรน threshold สักครู่หลังเห็น box ล่าสุด
            use_keep = False
            if self._last_box_bbox is not None:
                age_ms = (time.monotonic() - self._last_box_time_s) * 1000
                hyst_ms = float(self._p("box_hysteresis_ms"))
                if age_ms <= hyst_ms:
                    if self._iou((x1,y1,x2,y2), self._last_box_bbox) >= 0.15:
                        use_keep = True

            thr = float(self._p("box_conf_threshold"))
            if use_keep:
                thr = min(thr, 0.35)
            if conf < thr: return False

            ar_min = float(self._p("box_aspect_ratio_min"))
            ar_max = float(self._p("box_aspect_ratio_max"))
            if area < int(self._p("box_min_area_px")): return False
            if ar < ar_min or ar > ar_max: return False
            if y2 > int(frame.shape[0] * 0.85): return False  # ตัด edge ล่างสุด

            if bool(self._p("enable_white_box_color_filter")):
                wr = self._white_ratio(frame, x1, y1, x2, y2)
                min_wr = float(self._p("white_ratio_min"))
                if use_keep: min_wr = min(min_wr, 0.18)
                if wr < min_wr: return False

            self._last_box_label  = label
            self._last_box_bbox   = (x1, y1, x2, y2)
            self._last_box_time_s = time.monotonic()
            return True

        if ll == bottle_name:
            if conf < float(self._p("bottle_conf_threshold")): return False
            if area < int(self._p("bottle_min_area_px")): return False
            ar_min = float(self._p("bottle_aspect_ratio_min"))
            ar_max = float(self._p("bottle_aspect_ratio_max"))
            if ar < ar_min or ar > ar_max: return False
            return True

        return bool(self._p("allow_other_classes"))

    def _stable_filter(self, key, val_mm: float) -> float:
        prev = self._stable_depth.get(key)
        if prev is None:
            self._stable_depth[key] = val_mm; return val_mm
        max_jump = float(self._p("max_jump_mm"))
        if max_jump > 0 and abs(val_mm - prev) > max_jump:
            return prev
        alpha = max(0.0, min(float(self._p("ema_alpha")), 1.0))
        v = prev*(1-alpha) + val_mm*alpha
        self._stable_depth[key] = v
        return v

    def _size_distance_mm(self, label, bw_px, bh_px, fw, fh) -> int:
        if bw_px <= 0 or bh_px <= 0: return 0
        ll = label.strip().lower()
        bottle_name = str(self._p("bottle_class_name")).strip().lower()
        if ll != bottle_name or not bool(self._p("enable_bottle_size_distance")):
            return 0
        rw = float(self._p("known_bottle_width_mm"))
        rh = float(self._p("known_bottle_height_mm"))
        if rw <= 0 or rh <= 0: return 0
        hfov = math.radians(float(self._p("hfov_deg")))
        vfov = math.radians(float(self._p("vfov_deg")))
        fx = fw / (2*math.tan(hfov/2))
        fy = fh / (2*math.tan(vfov/2))
        est = []
        if bw_px > 0: est.append(rw*fx/bw_px)
        if bh_px > 0: est.append(rh*fy/bh_px)
        return max(0, int(np.median(est))) if est else 0

    def _estimate_depth(self, depth, frame, label, x1, y1, x2, y2):
        fh, fw = depth.shape[:2]
        ll = str(label).strip().lower()
        bottle_name = str(self._p("bottle_class_name")).strip().lower()
        x1=max(0,min(int(x1),fw-1)); x2=max(0,min(int(x2),fw))
        y1=max(0,min(int(y1),fh-1)); y2=max(0,min(int(y2),fh))
        if x2<=x1 or y2<=y1: return 0, 0

        mn = int(self._p("min_depth_mm")); mx = int(self._p("max_depth_mm"))

        # Shrink ROI
        sr = _ROI_SHRINK_RATIO if ll != bottle_name else 0.06
        w,h = x2-x1, y2-y1
        rx1=x1+int(w*sr); rx2=x2-int(w*sr)
        ry1=y1+int(h*sr); ry2=y2-int(h*sr)
        if rx2<=rx1 or ry2<=ry1: rx1,rx2,ry1,ry2=x1,x2,y1,y2
        patch = depth[ry1:ry2, rx1:rx2]
        valid_roi = patch[(patch>=mn)&(patch<=mx)]
        patch_area = patch.shape[0]*patch.shape[1]
        roi_ok = patch_area>0 and valid_roi.size/patch_area>=0.02 and valid_roi.size>=12

        # Center patch
        cx=(x1+x2)//2; cy=(y1+y2)//2; r=8
        cp = depth[max(0,cy-r):min(fh,cy+r+1), max(0,cx-r):min(fw,cx+r+1)]
        vc = cp[(cp>=mn)&(cp<=mx)]
        center_ok = cp.size>0 and vc.size/cp.size>=0.02 and vc.size>=12

        candidates = []; quality = 0
        raw_cluster=None; raw_cluster_dense=False
        raw_surface=None; raw_center=None; raw_bottom=None; raw_edge=None

        # A: dense cluster
        if roi_ok:
            q_arr = (valid_roi.astype(np.int32) // _DEPTH_BIN_WIDTH_MM)
            counts = np.bincount(q_arr)
            min_cl = max(_MIN_CLUSTER_PIXELS, int(valid_roi.size*_MIN_CLUSTER_FRACTION))
            cand_bins = np.flatnonzero(counts >= min_cl)
            if cand_bins.size == 0:
                weak = max(12, int(min_cl*0.35))
                weak_bins = np.flatnonzero(counts >= weak)
                chosen = int(weak_bins.min()) if weak_bins.size else None
            else:
                chosen = int(cand_bins.min())
            if chosen is not None:
                lo=max(0,chosen-_BIN_NEIGHBOR_WIDTH); hi=chosen+_BIN_NEIGHBOR_WIDTH
                band = valid_roi[(q_arr>=lo)&(q_arr<=hi)]
                if band.size == 0: band = valid_roi
                pct_cut = np.percentile(band, _CLOSEST_PERCENTILE)
                near = band[band<=pct_cut]
                if near.size >= max(12, int(min_cl*0.25)): band = near
                s = np.sort(band.astype(np.float32))
                cut = int(s.size*_TRIM_RATIO)
                trimmed = s[cut:-cut] if cut>0 and s.size-2*cut>0 else s
                raw = float(np.mean(trimmed))
                if np.isfinite(raw) and raw>0:
                    candidates.append(raw)
                    quality = max(quality, 2 if cand_bins.size else 1)
                    raw_cluster=raw; raw_cluster_dense=bool(cand_bins.size)

        # B: surface percentile
        if roi_ok:
            raw = float(np.percentile(valid_roi.astype(np.float32), _DEPTH_SURFACE_PERCENTILE))
            if np.isfinite(raw) and raw>0:
                candidates.append(raw); quality=max(quality,1); raw_surface=raw

        # C: center patch median
        if center_ok:
            raw = float(np.median(vc.astype(np.float32)))
            if np.isfinite(raw) and raw>0:
                candidates.append(raw); quality=max(quality,2); raw_center=raw

        # D: bottom band
        if roi_ok:
            bh2 = int(round(patch.shape[0]*_BOTTOM_BAND_RATIO))
            if bh2 >= 1:
                bot = patch[max(0,patch.shape[0]-bh2):, :]
                vb  = bot[(bot>=mn)&(bot<=mx)]
                if vb.size >= 12:
                    raw = float(np.percentile(vb.astype(np.float32), _BOTTOM_BAND_PERCENTILE))
                    if np.isfinite(raw) and raw>0:
                        candidates.append(raw); quality=max(quality,1); raw_bottom=raw

        # E: edge-based
        if roi_ok and frame is not None:
            roi_bgr = frame[ry1:ry2, rx1:rx2]
            if roi_bgr.size and roi_bgr.shape[:2]==patch.shape[:2]:
                ms = max(32, _EDGE_MAX_SIDE_PX)
                rh2,rw2 = roi_bgr.shape[:2]
                if max(rh2,rw2)>ms:
                    sc=ms/max(rh2,rw2)
                    roi_bgr = cv2.resize(roi_bgr,(max(1,int(rw2*sc)),max(1,int(rh2*sc))))
                    depth_s = cv2.resize(patch,(roi_bgr.shape[1],roi_bgr.shape[0]),interpolation=cv2.INTER_NEAREST)
                else:
                    depth_s = patch
                gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                edges= cv2.Canny(gray, _EDGE_CANNY_LOW, _EDGE_CANNY_HIGH)
                if _EDGE_DILATE_ITER>0:
                    edges=cv2.dilate(edges,np.ones((3,3),np.uint8),iterations=_EDGE_DILATE_ITER)
                ed = depth_s[(edges>0)&(depth_s>=mn)&(depth_s<=mx)]
                if ll in {"box","bottle"} and ed.size>=_EDGE_MIN_PIXELS:
                    raw=float(np.percentile(ed.astype(np.float32), max(_CLOSEST_PERCENTILE,5.0)))
                    if np.isfinite(raw) and raw>0:
                        candidates.append(raw); quality=max(quality,1); raw_edge=raw

        if not candidates: return 0, 0

        # Pick base
        if   ll==bottle_name and raw_edge   is not None: base=float(raw_edge)
        elif raw_cluster is not None and raw_cluster_dense: base=float(raw_cluster)
        elif raw_center  is not None: base=float(raw_center)
        elif raw_cluster is not None: base=float(raw_cluster)
        elif raw_surface is not None: base=float(raw_surface)
        elif raw_bottom  is not None: base=float(raw_bottom)
        elif raw_edge    is not None: base=float(raw_edge)
        else: base=float(np.median(candidates))

        filtered = [c for c in candidates if base>0 and _DEPTH_CAND_RATIO_MIN<=c/base<=_DEPTH_CAND_RATIO_MAX] or candidates
        raw_mm = float(np.mean(filtered))
        if not np.isfinite(raw_mm) or raw_mm<=0: return 0, 0
        return max(0, int(raw_mm)), quality

    def _size_calibration(self, label, dist_mm, obj_w, obj_h):
        if not bool(self._p("enable_size_calibration")) or dist_mm<=0 or obj_w<=0 or obj_h<=0:
            return dist_mm, obj_w, obj_h
        kw = float(self._p("known_box_width_mm")); kh = float(self._p("known_box_height_mm"))
        if kw<=0 or kh<=0: return dist_mm, obj_w, obj_h
        kl,ks = max(kw,kh), min(kw,kh)
        ml,ms = max(float(obj_w),float(obj_h)), min(float(obj_w),float(obj_h))
        if ml<=0 or ms<=0: return dist_mm, obj_w, obj_h
        se = float(np.median([kl/ml, ks/ms]))
        se = max(_SIZE_CALIB_MIN_SCALE, min(_SIZE_CALIB_MAX_SCALE, se))
        old = self._size_scale_by_label.get(label, 1.0)
        sc  = old*(1-_SIZE_CALIB_ALPHA) + se*_SIZE_CALIB_ALPHA
        self._size_scale_by_label[label] = sc
        return int(dist_mm*sc), int(obj_w*sc), int(obj_h*sc)

    def _center_bounds(self, fw):
        if not bool(self._p("enable_center_region_filter")): return 0, fw
        ratio = max(0.0, min(float(self._p("center_region_width_ratio")), 1.0))
        if ratio<=0 or ratio>=0.999: return 0, fw
        side = (1-ratio)/2
        lx = max(0, int(round(fw*side))); rx = fw-lx
        return lx, rx

    # ── main loop ─────────────────────────────────────────────────

    def process_frame(self):
        # Flush stale camera buffer
        for _ in range(_CAMERA_GRAB_FLUSH):
            try: self.cap.grab()
            except Exception: break

        ret, frame = self.cap.read()
        if not ret: return

        fw = int(self._p("frame_width")); fh = int(self._p("frame_height"))
        if frame.shape[1]!=fw or frame.shape[0]!=fh:
            frame = cv2.resize(frame, (fw, fh))

        # Depth frame
        d_frame = self.depth_stream.read_frame()
        d_data  = d_frame.get_buffer_as_uint16()
        dw = int(getattr(d_frame,"width",fw)); dh = int(getattr(d_frame,"height",fh))
        depth = np.frombuffer(d_data, dtype=np.uint16).reshape(dh, dw)
        if depth.shape[1]!=fw or depth.shape[0]!=fh:
            depth = cv2.resize(depth,(fw,fh),interpolation=cv2.INTER_NEAREST)

        objects = []
        debug_lines = []

        box_name    = str(self._p("box_class_name")).strip().lower()
        bottle_name = str(self._p("bottle_class_name")).strip().lower()
        cl_left, cl_right = self._center_bounds(fw)

        if self._detection_enabled:
            # ─── YOLO IS ENABLED ───
            results = self.model.predict(
                frame,
                imgsz   = int(self._p("imgsz")),
                conf    = float(self._p("conf")),
                device  = str(self._p("device")),
                half    = False,
                classes = self.target_class_ids,
                verbose = False,
                max_det = int(self._p("max_det")),
            )

            dets = []
            for r in results:
                if r.boxes is None: continue
                for box in r.boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    conf_v = float(box.conf[0]) if box.conf is not None else 0.0
                    cls_id = int(box.cls[0])
                    names  = self.model.names
                    label  = names.get(cls_id,str(cls_id)) if isinstance(names,dict) else (names[cls_id] if 0<=cls_id<len(names) else str(cls_id))
                    cx = (max(0,x1)+min(fw,x2))//2
                    if not (cl_left <= cx < cl_right): continue
                    if not self._passes_filters(frame, label, conf_v, x1, y1, x2, y2): continue

                    bw_px = max(0,x2-x1); bh_px = max(0,y2-y1)
                    # Smooth bbox size
                    sk = (label.strip().lower(), cx//_STABLE_KEY_GRID_PX, (y1+y2)//2//_STABLE_KEY_GRID_PX)
                    prev_b = self._stable_bbox.get(sk)
                    if prev_b is None:
                        self._stable_bbox[sk] = (float(bw_px), float(bh_px))
                    else:
                        a = max(0.0,min(_BBOX_EMA_ALPHA,1.0))
                        self._stable_bbox[sk] = (prev_b[0]*(1-a)+bw_px*a, prev_b[1]*(1-a)+bh_px*a)
                    bw_s = int(max(1, round(self._stable_bbox[sk][0])))
                    bh_s = int(max(1, round(self._stable_bbox[sk][1])))

                    dd, dq = self._estimate_depth(depth, frame, label, x1, y1, x2, y2)
                    ds     = self._size_distance_mm(label, bw_s, bh_s, fw, fh)
                    dets.append(dict(label=label,ll=label.strip().lower(),
                                     x1=x1,y1=y1,x2=x2,y2=y2,bw=bw_s,bh=bh_s,
                                     conf=conf_v,sk=sk,dd=dd,dq=dq,ds=ds))

            # Box hold (anti-blink)
            if not any(d["ll"]==box_name for d in dets) and self._last_box_bbox is not None:
                age_ms = (time.monotonic()-self._last_box_time_s)*1000
                if age_ms <= float(self._p("box_hold_ms")):
                    hx1,hy1,hx2,hy2 = [max(0,min(v,fw if i%2 else fh)) for i,v in enumerate(self._last_box_bbox)]
                    if hx2>hx1 and hy2>hy1:
                        cx=(hx1+hx2)//2; cy=(hy1+hy2)//2
                        if cl_left<=cx<cl_right:
                            sk=(box_name,cx//_STABLE_KEY_GRID_PX,cy//_STABLE_KEY_GRID_PX)
                            bw_px=hx2-hx1; bh_px=hy2-hy1
                            dd,dq = self._estimate_depth(depth,frame,self._last_box_label or box_name,hx1,hy1,hx2,hy2)
                            ds    = 0
                            dets.append(dict(label=self._last_box_label or box_name, ll=box_name,
                                             x1=hx1,y1=hy1,x2=hx2,y2=hy2,bw=bw_px,bh=bh_px,
                                             conf=0.0,sk=sk,dd=dd,dq=dq,ds=ds,held=True))

            # Bottle anchor
            anchor_depth=0; anchor_y2=0
            if bool(self._p("enable_bottle_anchor")):
                b_cands=[]; b_y2s=[]
                for d in dets:
                    if d["ll"]!=bottle_name: continue
                    dd,ds,dq=d["dd"],d["ds"],d["dq"]
                    cand=0
                    if dd>0 and ds>0:
                        cand = ds if dd/ds<0.55 or dd/ds>1.75 else (dd if dq>=2 else ds)
                    elif dd>0 and dq>=2: cand=dd
                    elif ds>0: cand=ds
                    elif dd>0: cand=dd
                    if cand>0: b_cands.append(cand); b_y2s.append(d["y2"])
                if b_cands:
                    anchor_depth=int(np.median(b_cands))
                    anchor_y2  =int(np.median(b_y2s))

            debug_lines.append(f"info: dets={len(dets)} anchor={anchor_depth}mm")

            for d in dets:
                label=d["label"]; ll=d["ll"]
                x1,y1,x2,y2=d["x1"],d["y1"],d["x2"],d["y2"]
                bw,bh=d["bw"],d["bh"]
                dd,dq,ds=d["dd"],d["dq"],d["ds"]
                sk=d["sk"]

                # Fuse depth + size
                if dd<=0 or dq<=0: dist=ds
                elif ds<=0:         dist=dd
                else:
                    ratio=dd/ds if ds>0 else 1.0
                    oor = ratio<0.55 or ratio>1.75
                    if dq>=1:
                        dist = (ds if ll==bottle_name and ds>0 else dd) if oor else int(0.75*dd+0.25*ds)
                    else:
                        dist = ds if oor else int(0.75*dd+0.25*ds)

                # Anchor correction
                if anchor_depth>0 and ll==box_name:
                    ytol=int(self._p("anchor_y_tolerance_px"))
                    if anchor_y2>0 and abs(y2-anchor_y2)<=ytol:
                        if dist>0:
                            ratio_a = dist/anchor_depth if anchor_depth>0 else 1.0
                            if ratio_a<_ANCHOR_RATIO_MIN or ratio_a>_ANCHOR_RATIO_MAX or abs(dist-anchor_depth)>_ANCHOR_MAX_ABS_DIFF_MM:
                                dist=anchor_depth; dq=max(dq,2)
                            if abs(dist-anchor_depth)>_ANCHOR_SOFT_TOL_MM:
                                dist=anchor_depth; dq=max(dq,2)
                            else:
                                dist=int(round((1-_ANCHOR_BLEND_WEIGHT)*dist+_ANCHOR_BLEND_WEIGHT*anchor_depth)); dq=max(dq,2)
                        else:
                            dist=anchor_depth; dq=max(dq,2)

                # Stability filter
                if dist>0:
                    if dq>=2: dist=int(self._stable_filter(sk,dist))
                    else:
                        prev=self._stable_depth.get(sk)
                        if prev is not None: dist=int(prev)

                if dist<=0:
                    dist=int(self._last_depth_by_key.get(sk,0))
                if dist>0:
                    self._last_depth_by_key[sk]=dist

                if dist<=0: continue

                # Compute physical size
                hfov=math.radians(float(self._p("hfov_deg")))
                vfov=math.radians(float(self._p("vfov_deg")))
                obj_w=int(bw*(2*dist*np.tan(hfov/2)/fw))
                obj_h=int(bh*(2*dist*np.tan(vfov/2)/fh))
                dist,obj_w,obj_h=self._size_calibration(label,dist,obj_w,obj_h)

                objects.append(f"{label},{dist},{obj_w},{obj_h}")
                debug_lines.append(f"det: {label}  dist={dist}mm  {obj_w}x{obj_h}mm  dq={dq}  box=({x1},{y1},{x2},{y2})")

                # Draw bounding box
                vb = self._view_bbox.get(sk)
                a  = max(0.0,min(_VIEW_BBOX_EMA_ALPHA,1.0))
                if vb is None:
                    self._view_bbox[sk]=(float(x1),float(y1),float(x2),float(y2))
                else:
                    px1,py1,px2,py2=vb
                    if self._iou((x1,y1,x2,y2),(int(px1),int(py1),int(px2),int(py2)))<_VIEW_BBOX_RESET_IOU:
                        self._view_bbox[sk]=(float(x1),float(y1),float(x2),float(y2))
                    else:
                        self._view_bbox[sk]=(px1*(1-a)+x1*a, py1*(1-a)+y1*a, px2*(1-a)+x2*a, py2*(1-a)+y2*a)
                dx1,dy1,dx2,dy2=[max(0,min(int(round(v)),fw-1 if i%2==0 else fh-1)) for i,v in enumerate(self._view_bbox[sk])]
                dx2=max(dx1+1,min(dx2,fw)); dy2=max(dy1+1,min(dy2,fh))
                cv2.rectangle(frame,(dx1,dy1),(dx2,dy2),(0,255,0),2)
                txt=f"{label} {dist}mm {obj_w}x{obj_h}mm"
                (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
                ty=max(0,dy1-th-10)
                cv2.rectangle(frame,(dx1,ty),(dx1+tw+8,ty+th+8),(0,255,0),-1)
                cv2.putText(frame,txt,(dx1+4,ty+th+2),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

            # Publish detected objects
            msg = String()
            msg.data = "|".join(objects)
            self.publisher.publish(msg)

        else:
            # ─── YOLO IS DISABLED ───
            debug_lines.append("info: Detection PAUSED by /detected_objects_enable")
            # Draw red warning text on the camera feed
            cv2.putText(frame, "DETECTION PAUSED", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Publish empty string to clear any lingering UI/state on the receiving end
            msg = String()
            msg.data = ""
            self.publisher.publish(msg)

        if self._detection_enabled:

            # Center guides (draw regardless of detection state if enabled)
            if bool(self._p("draw_center_region_guides")) and (cl_right-cl_left)<fw:
                cv2.line(frame,(max(0,cl_left),0),(max(0,cl_left),fh-1),(255,255,0),2)
                cv2.line(frame,(min(fw-1,cl_right),0),(min(fw-1,cl_right),fh-1),(255,255,0),2)

        # Push to web stream (skip encode when no viewer)
        if _stream.client_count()>0:
            q=max(10,min(100,int(self._p("stream_jpeg_quality"))))
            _stream.push_raw(frame, debug_lines, q)
        else:
            with _stream._lock:
                _stream._debug_lines=debug_lines

    def destroy_node(self):
        if hasattr(self,"cap") and self.cap: self.cap.release()
        if hasattr(self,"depth_stream") and self.depth_stream:
            try: self.depth_stream.stop()
            except Exception: pass
        try: openni2.unload()
        except Exception: pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloDepthPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()