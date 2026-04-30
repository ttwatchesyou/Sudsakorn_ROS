import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ObjectSubscriber(Node):

    def __init__(self):

        super().__init__("object_subscriber")

        self.declare_parameter("qos_depth", 1)
        self.declare_parameter("print_hz", 5.0)
        self._last_print_time = 0.0

        qos_depth = int(self.get_parameter("qos_depth").value)
        if qos_depth < 1:
            qos_depth = 1

        self.subscription = self.create_subscription(
            String,
            "/detected_objects",
            self.listener_callback,
            qos_depth,
        )

    def listener_callback(self, msg):
        print_hz = float(self.get_parameter("print_hz").value)
        if print_hz > 0:
            now = time.monotonic()
            if (now - self._last_print_time) < (1.0 / print_hz):
                return
            self._last_print_time = now

        data = msg.data

        if data == "":
            return

        objects = data.split("|")

        counter = {}
        printed_header = False

        for obj in objects:

            parts = [p.strip() for p in obj.split(",") if p.strip() != ""]
            if len(parts) < 4:
                continue

            if not printed_header:
                print("\n=== Latest Objects ===")
                printed_header = True

            label = parts[0]
            dist = parts[1]
            w_mm = parts[2]
            h_mm = parts[3]

            if label not in counter:
                counter[label] = 1
            else:
                counter[label] += 1

            idx = counter[label]

            try:
                dist_mm = float(dist)
            except Exception:
                dist_mm = 0.0

            try:
                area_mm2 = int(float(w_mm) * float(h_mm))
            except Exception:
                area_mm2 = 0

            print(
                f"{label}{idx}: "
                f"{dist_mm:.0f}mm "
                f"{area_mm2}mm2"
            )


def main(args=None):

    rclpy.init(args=args)

    node = ObjectSubscriber()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()
