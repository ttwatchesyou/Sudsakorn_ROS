"""
chair_counter.py  —  LiDAR Chair Detector for ROS2
====================================================
SETUP:
  - RPLidar C1 mounted UPSIDE DOWN on a shelf/table
  - Sensor is at chair-seat level (scanning seat front edges)
  - Chair seats appear as CONCAVE arcs (curving AWAY from sensor)
    because the sensor is inside/under the arc looking outward
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import cv2
import math

# ─────────────────────────────────────────────────
#  TUNING  — adjust these for your environment
# ─────────────────────────────────────────────────

MAX_RANGE        = 1.2    # metres — chairs are close at seat level
FOV_DEG          = 180.0  # front 180° (after flip)

# ── Clustering ────────────────────────────────────
CLUSTER_GAP      = 0.08   # max gap between consecutive points (m)
MIN_PTS          = 5      # min points to form a cluster

# ── Chair seat-edge chord (width of seat front arc) ──
ONE_CHAIR_MIN    = 0.32
ONE_CHAIR_MAX    = 0.58
IDEAL_ONE_CHAIR  = 0.45

TWO_CHAIR_MIN    = 0.65
TWO_CHAIR_MAX    = 1.00
IDEAL_TWO_CHAIR  = 0.82

# ── Concave arc geometry ──────────────────────────
MIN_DEV          = 0.006
MAX_DEV          = 0.060
MIN_CURVE_RATIO  = 0.008
MAX_CURVE_RATIO  = 0.090

MAX_SPIKE        = 0.14
MAX_SPIKE_RATIO  = 3.5
MAX_WIGGLE       = 1.30
MAX_ARC_OFF      = 0.35

# ── Noise / quality filters ───────────────────────
MIN_CLUSTER_PTS  = 5
MAX_CLUSTER_PTS  = 90
MIN_ARC_LENGTH   = 0.12
MIN_PT_DENSITY   = 8.0
MAX_PT_DENSITY   = 220.0
MIN_STRAIGHTNESS = 0.70
MIN_DIST         = 0.18

# ── Visualisation ─────────────────────────────────
IMG_SIZE         = 700
SCALE            = 240
# ─────────────────────────────────────────────────


class ChairDetector(Node):

    def __init__(self):
        super().__init__('chair_detector')
        self.sub = self.create_subscription(
            LaserScan, '/sudsakhon_scan', self._scan_cb, 10)
        self._fov_rad       = math.radians(FOV_DEG / 2.0)
        self._last_scan_pts = []
        self.get_logger().info(
            'ChairDetector started  [upside-down C1 / seat-level mode]')

    # ── ROS callback ─────────────────────────────────────────────────────────

    def _scan_cb(self, msg: LaserScan):
        scan_pts, clusters = self._cluster(msg)
        self._last_scan_pts = scan_pts
        shapes = self._classify(clusters)
        chair_count = sum(s['chairs'] for s in shapes)
        self._draw(shapes, chair_count)

    # ── Step 1: filter + cluster ──────────────────────────────────────────────

    def _cluster(self, msg):
        polar = []
        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r) or r <= msg.range_min or r >= MAX_RANGE:
                continue

            raw_angle = msg.angle_min + i * msg.angle_increment
            angle = -(raw_angle + math.pi)
            angle = math.atan2(math.sin(angle), math.cos(angle))

            if abs(angle) > self._fov_rad:
                continue

            polar.append((angle, r))

        polar.sort(key=lambda x: x[0])
        pts = [np.array([r * math.cos(a), r * math.sin(a)])
               for a, r in polar]
        scan_pts = list(pts)

        clusters, curr = [], ([pts[0]] if pts else [])
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - curr[-1]) < CLUSTER_GAP:
                curr.append(pts[i])
            else:
                if len(curr) >= MIN_PTS:
                    clusters.append(np.array(curr))
                curr = [pts[i]]
        if len(curr) >= MIN_PTS:
            clusters.append(np.array(curr))

        return scan_pts, clusters

    # ── Step 2: geometry (concave arc aware) ──────────────────────────────────

    def _geometry(self, cluster):
        p1, p2 = cluster[0], cluster[-1]
        chord  = np.linalg.norm(p2 - p1)
        if chord < 0.01:
            return 0.0, 0.0, 0.0, 1.0, 0.0, 0.0

        edge = p2 - p1
        signed_devs = []
        abs_devs    = []
        path        = 0.0

        for i, p in enumerate(cluster):
            cross = float(np.cross(edge, p - p1)) / chord
            signed_devs.append(cross)
            abs_devs.append(abs(cross))
            if i > 0:
                path += np.linalg.norm(cluster[i] - cluster[i - 1])

        avg_signed = float(np.mean(signed_devs))
        avg_abs    = float(np.mean(abs_devs))
        max_abs    = float(np.max(abs_devs))
        wiggle     = path / chord
        peak       = int(np.argmax(abs_devs))
        arc_off    = abs(peak - len(abs_devs) // 2) / len(abs_devs)

        return avg_signed, avg_abs, max_abs, wiggle, arc_off, path

    # ── Step 3: classify ──────────────────────────────────────────────────────

    def _classify(self, clusters):
        shapes = []

        for cluster in clusters:
            p1, p2 = cluster[0], cluster[-1]
            length = np.linalg.norm(p1 - p2)
            n_pts  = len(cluster)

            if n_pts < MIN_CLUSTER_PTS:                         continue
            if n_pts > MAX_CLUSTER_PTS:                         continue

            avg_signed, avg_abs, max_abs, wiggle, arc_off, arc_len = \
                self._geometry(cluster)

            if arc_len < MIN_ARC_LENGTH:                        continue

            density = n_pts / max(arc_len, 0.001)
            if density < MIN_PT_DENSITY:                        continue
            if density > MAX_PT_DENSITY:                        continue

            straightness = length / max(arc_len, 0.001)
            if straightness < MIN_STRAIGHTNESS:                 continue

            if avg_signed > -MIN_DEV:                           continue
            if avg_abs > MAX_DEV:                               continue
            if max_abs > MAX_SPIKE:                             continue
            if avg_abs > 0 and max_abs > avg_abs * MAX_SPIKE_RATIO: continue
            if wiggle  > MAX_WIGGLE:                            continue
            if arc_off > MAX_ARC_OFF:                           continue

            avg_dist = float(np.mean(np.linalg.norm(cluster, axis=1)))
            if avg_dist < MIN_DIST:                             continue

            curve_ratio = avg_abs / max(length, 0.001)
            if curve_ratio < MIN_CURVE_RATIO:                   continue
            if curve_ratio > MAX_CURVE_RATIO:                   continue

            chairs, score = 0, 999.0
            if ONE_CHAIR_MIN <= length <= ONE_CHAIR_MAX:
                chairs = 1
                score  = abs(length - IDEAL_ONE_CHAIR) + arc_off * 0.1
            elif TWO_CHAIR_MIN <= length <= TWO_CHAIR_MAX:
                chairs = 2
                score  = abs(length - IDEAL_TWO_CHAIR)
            elif ONE_CHAIR_MAX < length < TWO_CHAIR_MIN:
                chairs = 2
                score  = abs(length - IDEAL_TWO_CHAIR) + 0.05

            if chairs > 0:
                cx = float(np.mean(cluster[:, 0]))
                cy = float(np.mean(cluster[:, 1]))
                shapes.append({
                    'chairs':   chairs,
                    'centroid': np.array([cx, cy]),
                    'p1': p1,   'p2': p2,
                    'length':   length,
                    'cluster':  cluster,
                    'score':    score,
                    'dev':      avg_abs,
                    'signed':   avg_signed,
                    'arc':      arc_off,
                    'dist':     avg_dist,
                })

        shapes.sort(key=lambda x: np.linalg.norm(x['centroid']))
        shapes = shapes[:2]

        total = sum(s['chairs'] for s in shapes)
        if total not in (1, 2):
            return []
        return shapes

    # ── Step 4: draw ──────────────────────────────────────────────────────────

    def _draw(self, shapes, chair_count):
        img    = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        ox, oy = IMG_SIZE // 2, IMG_SIZE // 2

        def to_px(p):
            return (int(ox - p[1] * SCALE), int(oy - p[0] * SCALE))

        # Grid rings
        for r_m, col in [(0.5, (35, 35, 35)),
                         (1.0, (55, 55, 55)),
                         (1.5, (35, 35, 35))]:
            cv2.circle(img, (ox, oy), int(r_m * SCALE), col, 1)
            cv2.putText(img, f'{r_m}m',
                        (ox + int(r_m * SCALE) + 3, oy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (55, 55, 55), 1)

        # FOV cone lines
        for sign in (-1, 1):
            ex = int(ox - math.sin(sign * self._fov_rad) * SCALE * MAX_RANGE)
            ey = int(oy - math.cos(sign * self._fov_rad) * SCALE * MAX_RANGE)
            cv2.line(img, (ox, oy), (ex, ey), (40, 40, 65), 1)

        # All scan points
        for pt in self._last_scan_pts:
            px, py = to_px(pt)
            if 0 <= px < IMG_SIZE and 0 <= py < IMG_SIZE:
                cv2.circle(img, (px, py), 1, (50, 50, 50), -1)

        # Robot / sensor dot
        cv2.circle(img, (ox, oy), 7, (0, 0, 200), -1)
        cv2.circle(img, (ox, oy), 7, (80, 80, 255), 1)

        # ── Draw detected shapes ─────────────────────────────────────────
        for obj in shapes:
            color = (0, 255, 80) if obj['chairs'] == 1 else (0, 210, 255)

            for pt in obj['cluster']:
                cv2.circle(img, to_px(pt), 3, color, -1)

            cv2.line(img, to_px(obj['p1']), to_px(obj['p2']),
                     (255, 255, 255), 1)

            cpx, cpy = to_px(obj['centroid'])
            dx, dy   = cpx - ox, cpy - oy
            dist_px  = max(math.hypot(dx, dy), 1)
            stop_x   = int(ox + dx * (1 - 18 / dist_px))
            stop_y   = int(oy + dy * (1 - 18 / dist_px))
            cv2.arrowedLine(img, (ox, oy), (stop_x, stop_y),
                            color, 2, tipLength=0.15)

            cv2.drawMarker(img, (cpx, cpy), color,
                           cv2.MARKER_CROSS, 14, 2)

            lx, ly = to_px(obj['p1'])
            tag = '1-chair' if obj['chairs'] == 1 else '2-chair'
            cv2.putText(img,
                        f"{tag}  {obj['dist']:.2f}m  "
                        f"w={obj['length']:.2f}m",
                        (lx + 4, ly - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)

        # ── HUD (top-right) ──────────────────────────────────────────────
        hud_x, hud_y, lh = IMG_SIZE - 210, 14, 19

        direction = self._facing_dir()
        dir_col   = {'FORWARD': (100, 255, 100),
                     'LEFT':    (100, 200, 255),
                     'RIGHT':   (255, 200, 100)}.get(direction, (160, 160, 160))

        hud = [
            ('--- LIDAR C1 -------',  (80, 80, 80)),
            ('Mount:  UPSIDE DOWN',   (180, 120, 60)),
            ('Level:  SEAT HEIGHT',   (180, 120, 60)),
            (f'Facing  {direction}',  dir_col),
            (f'FOV     {int(FOV_DEG)} deg',  (150, 150, 150)),
            (f'Range   {MAX_RANGE}m', (150, 150, 150)),
            ('',                      (0, 0, 0)),
            ('--- DETECTION ------',  (80, 80, 80)),
            (f'Clusters  {len(shapes)}',    (200, 200, 200)),
            (f'Chairs    {chair_count}',    (200, 200, 200)),
        ]

        panel_h = lh * len(hud) + 10
        overlay = img.copy()
        cv2.rectangle(overlay,
                      (hud_x - 8, hud_y - 12),
                      (IMG_SIZE - 4, hud_y + panel_h),
                      (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

        for i, (text, col) in enumerate(hud):
            if text:
                cv2.putText(img, text,
                            (hud_x, hud_y + i * lh),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

        # Big chair count (bottom-left) — ตรงๆ จากเฟรมปัจจุบัน
        count_col = (0, 255, 80) if chair_count > 0 else (70, 70, 70)
        cv2.putText(img, f'Chairs: {chair_count}',
                    (14, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, count_col, 2)

        cv2.imshow('Chair Detector  [seat-level / upside-down C1]', img)
        cv2.waitKey(1)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _facing_dir(self):
        if not self._last_scan_pts:
            return 'UNKNOWN'
        pts    = np.array(self._last_scan_pts)
        angles = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
        mean_a = float(np.mean(angles))
        if   mean_a < -20: return 'LEFT'
        elif mean_a >  20: return 'RIGHT'
        else:              return 'FORWARD'


# ─────────────────────────────────────────────────

def main():
    rclpy.init()
    node = ChairDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()