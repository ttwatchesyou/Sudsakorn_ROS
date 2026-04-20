class SensorSystem:
    def __init__(self):
        # สถานะเซนเซอร์ (0 หรือ 1)
        self.table_l = 1
        self.table_r = 1
        self.box_down = 1
        self.box_up = 1
        self.bottle_detected = 1
        self.box_detected = 1
        self.sw1 = 1
        self.sw2 = 1
        self.sw3 = 1

    def update(self, data):
        """อัปเดตค่าจากอาเรย์ที่รับมาจาก bridge_node"""
        if len(data) >= 9:
            self.table_l = data[0]
            self.table_r = data[1]
            self.box_down = data[2]
            self.box_up = data[3]
            self.bottle_detected = data[4]
            self.box_detected = data[5]
            self.sw1 = data[6]
            self.sw2 = data[7]
            self.sw3 = data[8]