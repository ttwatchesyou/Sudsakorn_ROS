from dataclasses import dataclass, field
from typing import Dict, Optional, List

# =====================================================================
# 1. Class สำหรับเก็บข้อมูลพิกัด "1 จุด" (Waypoint)
# =====================================================================
@dataclass
class Waypoint:
    x: float
    y: float
    theta: float = 0.0  # มุม (องศา หรือ เรเดียน) ค่าเริ่มต้นคือ 0.0
    speed_limit : float = 0.0
    curve_strength : float = 0.0
    curve_kp_ : float = 0.0

    # เพิ่มตัวแปร PID Array พร้อมตั้งค่าเริ่มต้นตามที่กำหนด (ใช้ lambda เพื่อให้แต่ละจุดแยก Array กัน)
    yaw_pid_Set: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    pos_x_pid_set: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    pos_y_pid_set: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
# =====================================================================
# 2. Class สำหรับ "จัดการจุดหลายๆ จุด" (Position Manager)
# =====================================================================
class PositionManager:
    def __init__(self):
        # ใช้ Dictionary เก็บข้อมูลแบบ Key: Value -> "ชื่อจุด": Waypoint
        self.saved_positions: Dict[str, Waypoint] = {}

    def save_position(self, 
                        name: str, 
                        x: float, 
                        y: float, 
                        theta: float = 0.0, 
                        speed_limit: float = 0.0, 
                        curve_strength: float = 0.0, 
                        curve_kp_: float = 0.0 , 
                        yaw_pid_Set: Optional[List[float]] = None, 
                        pos_x_pid_set: Optional[List[float]] = None,
                        pos_y_pid_set: Optional[List[float]] = None):

        """บันทึกพิกัดพร้อมตั้งชื่อ"""
        self.saved_positions[name] = Waypoint(x, y, theta ,speed_limit ,curve_strength ,curve_kp_, yaw_pid_Set, pos_x_pid_set,pos_y_pid_set)
        #print(f"✅ บันทึกจุด: '{name}' สำเร็จ (X={x}, Y={y}, Theta={theta}, Theta={speed_limit}, curve_strength={curve_strength}, curve_kp_={curve_kp_})")

    def get_position(self, name: str) -> Optional[Waypoint]:
        """ดึงข้อมูลพิกัดจากชื่อ"""
        if name in self.saved_positions:
            return self.saved_positions[name]
        else:
            print(f"❌ ไม่พบจุดที่ชื่อว่า: '{name}'")
            return None

    def delete_position(self, name: str):
        """ลบจุดพิกัดที่บันทึกไว้"""
        if name in self.saved_positions:
            del self.saved_positions[name]
            print(f"🗑️ ลบจุด: '{name}' สำเร็จ")

    def show_all_positions(self):
        """แสดงจุดพิกัดทั้งหมดที่บันทึกไว้"""
        print("\n--- 📍 จุดพิกัดทั้งหมดที่บันทึกไว้ ---")
        if not self.saved_positions:
            print("ยังไม่มีการบันทึกจุดพิกัด")
            return
            
        for name, pos in self.saved_positions.items():
            print(f" - [{name}]: X={pos.x}, Y={pos.y}, Theta={pos.theta}, speed_limit={pos.speed_limit}, curve_strength={pos.curve_strength}, curve_kp_={pos.curve_kp_}")
        print("------------------------------------\n")
