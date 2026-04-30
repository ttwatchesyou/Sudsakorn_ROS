import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32, Float32MultiArray
from visualization_msgs.msg import Marker 
from geometry_msgs.msg import Point
import numpy as np
import math
import time

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
FIXED_FRAME      = 'laser_chair' 
MAX_RANGE        = 2.0       
MIN_DIST         = 0.10      
MATCH_RADIUS     = 0.40      
LOST_TIMEOUT     = 5.0
MAX_CLUSTER_PTS  = 90

class ChairTrack:
    _next_id = 1
    def __init__(self, centroid, width, dist, angle_deg, chairs_count, p1, p2):
        self.id = ChairTrack._next_id
        ChairTrack._next_id += 1
        self.centroid = centroid.copy()
        self.width, self.dist, self.angle_deg = width, dist, angle_deg
        self.chairs_count = chairs_count  
        self.p1, self.p2 = p1.copy(), p2.copy()
        self.visible, self.last_seen = True, time.time()

    def update(self, centroid, width, dist, angle_deg, chairs_count, p1, p2):
        self.centroid = centroid.copy()
        self.width, self.dist, self.angle_deg = width, dist, angle_deg
        self.chairs_count = chairs_count  
        self.p1, self.p2 = p1.copy(), p2.copy()
        self.visible, self.last_seen = True, time.time()

class ChairSubscriber(Node):
    def __init__(self):
        super().__init__('chair_subscriber')

        # ค่าเริ่มต้น: 270 (-90) ถึง 90 องศา (ด้านหน้า)
        self.declare_parameter('filter_min_deg', 270.0) 
        self.declare_parameter('filter_max_deg', 90.0)  
        self.declare_parameter('scan_topic', '/sudsakhon_scan')
        self.declare_parameter('min_width', 0.15)    
        self.declare_parameter('max_width', 1.20)    
        self.declare_parameter('two_chair_threshold', 0.50) 
        self.declare_parameter('cluster_gap', 0.05)  
        self.declare_parameter('min_pts', 4)         

        # ── Publishers ──
        self.filter_pub = self.create_publisher(LaserScan, '/filtered_scan', 10)
        self.marker_pub = self.create_publisher(Marker, '/filter_visual', 10)
        self.count_pub = self.create_publisher(Int32, '/chair_count', 10)

        # ── Subscribers ──
        self.sub = self.create_subscription(LaserScan, self.get_parameter('scan_topic').value, self._scan_cb, 10)
        self.param_sub = self.create_subscription(Float32MultiArray, '/chair_params', self._param_cb, 10)

        self._tracks = []

        print(f"\n" + "="*50)
        print(f" CHAIR TRACKER STARTED | Frame: {FIXED_FRAME}")
        print(f" Control via Topic: /chair_params")
        print(f" Data: [min_w, max_w, threshold, gap, min_deg, max_deg]")
        print(f"="*50 + "\n")

    def normalize_angle(self, angle):
        """ ทำมุมให้อยู่ในช่วง -pi ถึง pi """
        return math.atan2(math.sin(angle), math.cos(angle))

    def _param_cb(self, msg: Float32MultiArray):
        """ แก้ไขลำดับการเช็ค IF เพื่อให้รับ 6 ค่าได้ถูกต้อง """
        data = msg.data
        params = []
        
        # กรณีส่งมา 6 ค่าขึ้นไป (รวมมุม)
        if len(data) >= 6:
            params = [
                Parameter('min_width', Parameter.Type.DOUBLE, float(data[0])),
                Parameter('max_width', Parameter.Type.DOUBLE, float(data[1])),
                Parameter('two_chair_threshold', Parameter.Type.DOUBLE, float(data[2])),
                Parameter('cluster_gap', Parameter.Type.DOUBLE, float(data[3])),
                Parameter('filter_min_deg', Parameter.Type.DOUBLE, float(data[4])),
                Parameter('filter_max_deg', Parameter.Type.DOUBLE, float(data[5]))
            ]
            print(f"\n[UPDATED] Params + Angles: {data[4]:.1f}° to {data[5]:.1f}°")
        
        # กรณีส่งมาแค่ 4-5 ค่า (เฉพาะตรรกะเก้าอี้)
        elif len(data) >= 4:
            params = [
                Parameter('min_width', Parameter.Type.DOUBLE, float(data[0])),
                Parameter('max_width', Parameter.Type.DOUBLE, float(data[1])),
                Parameter('two_chair_threshold', Parameter.Type.DOUBLE, float(data[2])),
                Parameter('cluster_gap', Parameter.Type.DOUBLE, float(data[3]))
            ]
            print(f"\n[UPDATED] Chair Logic Only: w={data[0]}..{data[1]}")
        
        if params:
            self.set_parameters(params)
        else:
            print("\n[ERROR] /chair_params needs at least 4 or 6 values.")

    def _scan_cb(self, msg: LaserScan):
        # 1. ดึงพารามิเตอร์และ Normalize มุม
        min_rad = self.normalize_angle(math.radians(self.get_parameter('filter_min_deg').value))
        max_rad = self.normalize_angle(math.radians(self.get_parameter('filter_max_deg').value))

        filtered_ranges = [float('inf')] * len(msg.ranges)
        pts = []
        
        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r) or r < msg.range_min or r > MAX_RANGE:
                continue

            # 2. คำนวณมุมและ Normalize ให้อยู่ในระนาบเดียวกับ min/max_rad
            raw_angle = msg.angle_min + i * msg.angle_increment
            angle_rad = self.normalize_angle(raw_angle) 

            # 3. เช็คมุมแบบ Cross-zero (เช่น 270 ถึง 90)
            if min_rad <= max_rad:
                is_in_view = (min_rad <= angle_rad <= max_rad)
            else:
                is_in_view = (angle_rad >= min_rad or angle_rad <= max_rad)

            if is_in_view:
                filtered_ranges[i] = r
                pts.append(np.array([r * math.cos(raw_angle), r * math.sin(raw_angle)]))

        # 4. Publish Filtered Scan
        msg.ranges = filtered_ranges
        msg.header.frame_id = FIXED_FRAME
        self.filter_pub.publish(msg)

        # 5. Visualizer
        self._publish_filter_marker(min_rad, max_rad)

        # 6. Tracking Logic
        if pts:
            clusters = self._cluster(pts)
            detections = self._classify(clusters)
            self._update_tracks(detections)
        else:
            for t in self._tracks: t.visible = False
            
        self._publish_chair_marker()
        self._publish_chair_count()
        self._print_status()

    def _publish_filter_marker(self, min_rad, max_rad):
        marker = Marker()
        marker.header.frame_id = FIXED_FRAME
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "bounds"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        
        p_center = Point(x=0.0, y=0.0, z=0.0)
        p_min = Point(x=MAX_RANGE * math.cos(min_rad), y=MAX_RANGE * math.sin(min_rad), z=0.0)
        p_max = Point(x=MAX_RANGE * math.cos(max_rad), y=MAX_RANGE * math.sin(max_rad), z=0.0)

        marker.points = [p_center, p_min, p_center, p_max]
        marker.scale.x = 0.02 
        marker.color.a, marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 1.0, 1.0 
        self.marker_pub.publish(marker)

    def _cluster(self, pts):
        gap = self.get_parameter('cluster_gap').value
        min_pts = self.get_parameter('min_pts').value
        if not pts: return []
        clusters, curr = [], [pts[0]]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - curr[-1]) < gap:
                curr.append(pts[i])
            else:
                if len(curr) >= min_pts: clusters.append(np.array(curr))
                curr = [pts[i]]
        if len(curr) >= min_pts: clusters.append(np.array(curr))
        return clusters

    def _classify(self, clusters):
        min_w = self.get_parameter('min_width').value
        max_w = self.get_parameter('max_width').value
        min_pts = self.get_parameter('min_pts').value
        two_chair_thresh = self.get_parameter('two_chair_threshold').value 
        shapes = []
        for cluster in clusters:
            p1, p2 = cluster[0], cluster[-1]
            length = np.linalg.norm(p1 - p2)  
            if not (min_pts <= len(cluster) <= MAX_CLUSTER_PTS): continue
            avg_dist = float(np.mean(np.linalg.norm(cluster, axis=1)))
            if avg_dist < MIN_DIST: continue
            if min_w <= length <= max_w:
                cx, cy = float(np.mean(cluster[:, 0])), float(np.mean(cluster[:, 1]))
                count = 2 if length >= two_chair_thresh else 1
                shapes.append({
                    'width': length, 'centroid': np.array([cx, cy]), 'dist': avg_dist, 
                    'angle_deg': math.degrees(math.atan2(cy, cx)), 'chairs_count': count, 'p1': p1, 'p2': p2
                })
        return sorted(shapes, key=lambda x: x['dist'])[:2] 

    def _update_tracks(self, detections):
        now = time.time()
        for t in self._tracks: t.visible = False
        for det in detections:
            best_track, best_dist = None, MATCH_RADIUS
            for t in self._tracks:
                d = np.linalg.norm(det['centroid'] - t.centroid)
                if d < best_dist: best_dist, best_track = d, t
            if best_track:
                best_track.update(det['centroid'], det['width'], det['dist'], det['angle_deg'], det['chairs_count'], det['p1'], det['p2'])
            else:
                self._tracks.append(ChairTrack(det['centroid'], det['width'], det['dist'], det['angle_deg'], det['chairs_count'], det['p1'], det['p2']))
        self._tracks = [t for t in self._tracks if t.visible or (now - t.last_seen) < LOST_TIMEOUT]

    def _publish_chair_count(self):
        visible_tracks = [t for t in self._tracks if t.visible]
        total_chairs = sum(t.chairs_count for t in visible_tracks)
        count_msg = Int32()
        count_msg.data = total_chairs
        self.count_pub.publish(count_msg)

    def _publish_chair_marker(self):
        marker = Marker()
        marker.header.frame_id = FIXED_FRAME
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "chairs"
        marker.id = 1
        marker.type = Marker.LINE_LIST
        visible_tracks = [t for t in self._tracks if t.visible]
        if not visible_tracks:
            marker.action = Marker.DELETE
            self.marker_pub.publish(marker)
            return
        marker.action = Marker.ADD
        marker.scale.x = 0.04 
        marker.color.a, marker.color.r, marker.color.g, marker.color.b = 1.0, 0.2, 0.8, 0.2
        for t in visible_tracks:
            marker.points.append(Point(x=float(t.p1[0]), y=float(t.p1[1]), z=0.0))
            marker.points.append(Point(x=float(t.p2[0]), y=float(t.p2[1]), z=0.0))
        self.marker_pub.publish(marker)

    def _print_status(self):
        visible = [t for t in self._tracks if t.visible]
        total_chairs = sum(t.chairs_count for t in visible)
        print("\033[H\033[J", end="") 
        print(f"--- FRAME: {FIXED_FRAME} | SCAN: {self.get_parameter('scan_topic').value} ---")
        print(f"Total Chairs Detected: {total_chairs}")
        print(f"FOV: {self.get_parameter('filter_min_deg').value}° to {self.get_parameter('filter_max_deg').value}°")
        print(f"Logic: w={self.get_parameter('min_width').value}m-{self.get_parameter('max_width').value}m | Gap: {self.get_parameter('cluster_gap').value}m")
        print("-" * 65)
        for t in visible:
            print(f"ID C{t.id}: Dist={t.dist:.2f}m | Angle={t.angle_deg:+.1f}° | Width={t.width:.2f}m | Count={t.chairs_count}")

def main():
    rclpy.init()
    node = ChairSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()