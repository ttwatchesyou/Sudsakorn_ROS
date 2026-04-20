class ServoDevice:
    def __init__(self, parent, servo_id, off_angle, on_angle, name):
        self._parent = parent
        self._id = servo_id
        self._off_angle = off_angle
        self._on_angle = on_angle
        self._name = name

    def on(self):
        """สั่งให้ Servo ไปที่ตำแหน่ง ON (Max)"""
        self._parent.set_servo(self._id, self._on_angle)
        self._parent.get_logger().info(f'Servo {self._name} (ID:{self._id}): ON ({self._on_angle} deg)')

    def off(self):
        """สั่งให้ Servo ไปที่ตำแหน่ง OFF (Min)"""
        self._parent.set_servo(self._id, self._off_angle)
        self._parent.get_logger().info(f'Servo {self._name} (ID:{self._id}): OFF ({self._off_angle} deg)')

class ServoSystem:
    def __init__(self, parent):
        # กำหนดค่า Servo: ID, Min(Off), Max(On)
        self.BlockBottle = ServoDevice(parent, 0, 90, 15, "BlockBottle")
        self.BlockBottleDown = ServoDevice(parent, 4, 80, 90, "BlockBottleDown")
        self.PullBox = ServoDevice(parent, 2, 100, 30, "PullBox")