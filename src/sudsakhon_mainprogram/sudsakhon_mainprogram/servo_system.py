class ServoDevice:
    def __init__(self, parent, servo_id, grab_angle, support_angle, release_angle, name):
        self._parent = parent
        self._id = servo_id
        self._grab_angle = grab_angle       # มุมสำหรับ "จับ"
        self._support_angle = support_angle # มุมสำหรับ "ประคอง"
        self._release_angle = release_angle # มุมสำหรับ "คลาย"
        self._name = name

    def grab(self):
        """สั่งให้ Servo ไปที่ตำแหน่ง จับ"""
        self._parent.set_servo(self._id, self._grab_angle)
        self._parent.get_logger().info(f'Servo {self._name} (ID:{self._id}): GRAB/จับ ({self._grab_angle} deg)')

    def support(self):
        """สั่งให้ Servo ไปที่ตำแหน่ง ประคอง"""
        self._parent.set_servo(self._id, self._support_angle)
        self._parent.get_logger().info(f'Servo {self._name} (ID:{self._id}): SUPPORT/ประคอง ({self._support_angle} deg)')

    def release(self):
        """สั่งให้ Servo ไปที่ตำแหน่ง คลาย"""
        self._parent.set_servo(self._id, self._release_angle)
        self._parent.get_logger().info(f'Servo {self._name} (ID:{self._id}): RELEASE/คลาย ({self._release_angle} deg)')

class ServoSystem:
    def __init__(self, parent):
        self.BottleRight = ServoDevice(parent, 0, grab_angle=120, support_angle=105, release_angle=90, name="BottleRight")
        self.BottleLeft  = ServoDevice(parent, 2, grab_angle=60, support_angle=75,  release_angle=90, name="BottleLeft")
        self.BoxPusher   = ServoDevice(parent, 4, grab_angle=90, support_angle=85,  release_angle=80, name="BoxPusher")