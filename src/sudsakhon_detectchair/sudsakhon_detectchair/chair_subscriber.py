import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter # นำเข้าสำหรับจัดการ Parameter
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32, Float32MultiArray # นำเข้าชนิดข้อความใหม่
from visualization_msgs.msg import Marker 
from geometry_msgs.msg import Point
import numpy as np
import math
import time
from collections import deque

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
FIXED_FRAME      = 'laser_chair' 
MAX_RANGE        = 1.8       
MIN_DIST         = 0.10      
MATCH_RADIUS     = 0.40      
LOST_TIMEOUT     = 5.0
HISTORY_LEN      = 40
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

        self.declare_parameter('filter_min_deg', -0.0) 
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
        # Publisher ใหม่: ส่งออกจำนวนเก้าอี้
        self.count_pub = self.create_publisher(Int32, '/chair_count', 10)

        # ── Subscribers ──
        self.sub = self.create_subscription(LaserScan, self.get_parameter('scan_topic').value, self._scan_cb, 10)
        # Subscriber ใหม่: รับค่าพารามิเตอร์แบบ Float32MultiArray
        self.param_sub = self.create_subscription(Float32MultiArray, '/chair_params', self._param_cb, 10)

        self._tracks = []

        print(f"\n" + "="*50)
        print(f" CHAIR TRACKER STARTED | Frame: {FIXED_FRAME}")
        print(f" Publishes: /chair_count (Int32)")
        print(f" Subscribes: /chair_params (Float32MultiArray)")
        print(f"="*50 + "\n")

    def _param_cb(self, msg: Float32MultiArray):
        """ ฟังก์ชันรับค่าจาก Topic มาอัปเดต Parameter """
        if len(msg.data) >= 4:
            # อัปเดต Parameter ทันทีเมื่อได้รับข้อความ
            self.set_parameters([
                Parameter('min_width', Parameter.Type.DOUBLE, float(msg.data[0])),
                Parameter('max_width', Parameter.Type.DOUBLE, float(msg.data[1])),
                Parameter('two_chair_threshold', Parameter.Type.DOUBLE, float(msg.data[2])),
                Parameter('cluster_gap', Parameter.Type.DOUBLE, float(msg.data[3]))
            ])
            print(f"\n[UPDATED VIA TOPIC] min_w={msg.data[0]:.2f}, max_w={msg.data[1]:.2f}, thresh={msg.data[2]:.2f}, gap={msg.data[3]:.2f}\n")
        else:
            print("\n[ERROR] /chair_params needs 4 values: [min_w, max_w, threshold, gap]\n")

    def _scan_cb(self, msg: LaserScan):
        min_rad = math.radians(self.get_parameter('filter_min_deg').value)
        max_rad = math.radians(self.get_parameter('filter_max_deg').value)

        filtered_ranges, pts = [], []
        
        for i, r in enumerate(msg.ranges):
            raw_angle = msg.angle_min + i * msg.angle_increment
            angle_rad = math.atan2(math.sin(-(raw_angle + math.pi)), math.cos(-(raw_angle + math.pi)))

            if math.isfinite(r) and msg.range_min < r < MAX_RANGE and min_rad <= angle_rad <= max_rad:
                filtered_ranges.append(r)
                pts.append(np.array([r * math.cos(raw_angle), r * math.sin(raw_angle)]))
            else:
                filtered_ranges.append(float('inf'))

        msg.ranges = filtered_ranges
        msg.header.frame_id = FIXED_FRAME
        self.filter_pub.publish(msg)

        self._publish_filter_marker(min_rad, max_rad)

        if pts:
            clusters = self._cluster(pts)
            detections = self._classify(clusters)
            self._update_tracks(detections)
        else:
            for t in self._tracks: t.visible = False
            
        self._publish_chair_marker()
        self._publish_chair_count() # ส่งจำนวนเก้าอี้ออกไป
        self._print_status()

    def _publish_chair_count(self):
        """ นับและส่งจำนวนเก้าอี้ออกไปยัง Topic /chair_count """
        visible_tracks = [t for t in self._tracks if t.visible]
        total_chairs = sum(t.chairs_count for t in visible_tracks)
        
        count_msg = Int32()
        count_msg.data = total_chairs
        self.count_pub.publish(count_msg)

    def _publish_filter_marker(self, min_rad, max_rad):
        marker = Marker()
        marker.header.frame_id = FIXED_FRAME
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "bounds"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        
        p_center = Point(x=0.0, y=0.0, z=0.0)
        p_min = Point(x=MAX_RANGE * math.cos(-(min_rad + math.pi)), y=MAX_RANGE * math.sin(-(min_rad + math.pi)), z=0.0)
        p_max = Point(x=MAX_RANGE * math.cos(-(max_rad + math.pi)), y=MAX_RANGE * math.sin(-(max_rad + math.pi)), z=0.0)
        
        marker.points = [p_center, p_min, p_center, p_max]
        marker.scale.x = 0.02 
        marker.color.a, marker.color.r, marker.color.g = 1.0, 1.0, 1.0 
        self.marker_pub.publish(marker)

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
        marker.color.a, marker.color.r, marker.color.g, marker.color.b = 1.0, 0.6, 0.6, 0.6

        for t in visible_tracks:
            marker.points.append(Point(x=float(t.p1[0]), y=float(t.p1[1]), z=0.0))
            marker.points.append(Point(x=float(t.p2[0]), y=float(t.p2[1]), z=0.0))
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

    def _print_status(self):
        visible = [t for t in self._tracks if t.visible]
        total_chairs = sum(t.chairs_count for t in visible)
        
        print("\033[H\033[J", end="") 
        print(f"--- FRAME: {FIXED_FRAME} ---")
        print(f"Total Chairs Detected: {total_chairs}")
        print(f"Settings -> Width: {self.get_parameter('min_width').value}m-{self.get_parameter('max_width').value}m | 2-Chair Thresh: {self.get_parameter('two_chair_threshold').value}m | Gap: {self.get_parameter('cluster_gap').value}m")
        print("-" * 55)
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