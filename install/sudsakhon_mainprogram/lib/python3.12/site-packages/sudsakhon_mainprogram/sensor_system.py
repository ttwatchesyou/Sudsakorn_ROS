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


        self.LimitBoxBUp = 1
        self.LimitBoxBDw = 1
        self.LimitBoxBOut = 1
        self.LimitBoxBIn = 1
        self.SW_1 = 1
        self.SW_2 = 1
        self.SensorbottleL_B_UP = 1
        self.SensorbottleL_B_DW = 1
        self.SensorbottleR_B_UP = 1
        self.SensorbottleR_B_DW = 1
        self.SensorbottleL_Check = 1
        self.SensorbottleR_Check = 1
        self.SensorCheckBoxUp = 1

    def update(self, data):
        """อัปเดตค่าจากอาเรย์ที่รับมาจาก bridge_node"""
        #print("--> " + str(len(data) ))
        if len(data) >= 13:
            self.LimitBoxBUp = data[0]
            self.LimitBoxBDw = data[1]
            self.LimitBoxBOut = data[2]
            self.LimitBoxBIn = data[3]
            self.SW_1 = data[4]
            self.SW_2 = data[5]
            self.SensorbottleL_B_UP = data[6]
            self.SensorbottleL_B_DW = data[7]
            self.SensorbottleR_B_UP = data[8]
            self.SensorbottleR_B_DW = data[9]
            self.SensorbottleL_Check = data[10]
            self.SensorbottleR_Check = data[11]
            self.SensorCheckBoxUp = data[12]
