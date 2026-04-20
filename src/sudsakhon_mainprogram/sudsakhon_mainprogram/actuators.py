class SlideSystem:
    def __init__(self, parent):
        self._parent = parent

    def enable(self):
        self._parent.current_state[3] = 1
        self._parent.publish_state()
        self._parent.get_logger().info('Slide System: ENABLED')

    def disable(self):
        self._parent.current_state[3] = 0
        self._parent.publish_state()
        self._parent.get_logger().info('Slide System: DISABLED')

    def up(self):
        self._parent.current_state[0] = 2
        self._parent.publish_state()
        self._parent.get_logger().info('Slide Box: UP')

    def down(self):
        self._parent.current_state[0] = 1
        self._parent.publish_state()
        self._parent.get_logger().info('Slide Box: DOWN')

class BottleSystem:
    def __init__(self, parent):
        self._parent = parent

    def enable(self):
        self._parent.current_state[4] = 1
        self._parent.publish_state()

    def disable(self):
        self._parent.current_state[4] = 0
        self._parent.publish_state()

    def in_pos(self):
        self._parent.current_state[2] = 1
        self._parent.publish_state()

    def out_pos(self):
        self._parent.current_state[2] = 2
        self._parent.publish_state()

class BucketSystem:
    def __init__(self, parent):
        self._parent = parent

    def in_pos(self):
        self._parent.current_state[1] = 2
        self._parent.publish_state()

    def out_pos(self):
        self._parent.current_state[1] = 1
        self._parent.publish_state()