import serial
import time
import struct

# 그리퍼 하드웨어 특성: 왼쪽 조(jaw)는 고정, 오른쪽 조만 모터로 움직이는 비대칭(1-DOF)
# 구조. 관절이 이 모터 하나뿐이라 raw 위치값 하나가 "얼마나 벌어졌는지"를 완전히
# 설명하며, 이 비대칭성 자체는 이미 range_min/range_max 캘리브레이션에 통째로
# 반영되어 있어 별도 보정이 필요 없다.
#
# 방향은 실측으로 확인함 (my_robot_task_depth_top 에피소드 0, 손목캠 프레임 대조):
#   - raw 위치가 range_min에 가까울수록 -> 완전히 닫힘(그리퍼가 물체를 쥔 상태)
#   - raw 위치가 range_max에 가까울수록 -> 완전히 열림
GRIPPER_CLOSED_AT_RANGE_MIN = True


def gripper_open_ratio(raw_pos, range_min, range_max):
    """그리퍼 raw 위치를 '얼마나 열렸는지' 비율(0.0=완전히 닫힘, 1.0=완전히 열림)로 변환.
    GRIPPER_CLOSED_AT_RANGE_MIN 방향을 반영한 그리퍼 전용 헬퍼."""
    ratio = (raw_pos - range_min) / (range_max - range_min)
    ratio = max(0.0, min(1.0, ratio))
    return ratio if GRIPPER_CLOSED_AT_RANGE_MIN else 1.0 - ratio


class MiniFeetechDriver:
    # Feetech 모터 주요 레지스터 주소 (STS3215 기준)
    REG_GOAL_POSITION = 42
    REG_PRESENT_POSITION = 56
    REG_TORQUE_ENABLE = 40

    # ===== 캘리브레이션용(추가) =====
    # 아래 3개 주소는 STS3215 컨트롤테이블에 따라 달라질 수 있습니다.
    # 필요시 본인 모터 문서/테이블로 확인해서 수정하세요.
    REG_HOMING_OFFSET = 31
    REG_MIN_POSITION_LIMIT = 9
    REG_MAX_POSITION_LIMIT = 11
    REG_TORQUE_LIMIT = 48  # 0~1000 (0~100%). 낮으면 부하(중력)에 밀려 목표를 못 따라감

    def __init__(self, port='/dev/ttyUSB0', baudrate=1000000):
        # 1. 시리얼 포트 설정 (LeRobot도 1M baudrate 사용)
        self.ser = serial.Serial(port, baudrate, timeout=0.05)

    ''' 
    def _send_packet(self, motor_id, instruction, parameters):
        """Feetech 프로토콜에 맞춘 패킷 생성 및 전송"""
        length = len(parameters) + 2
        packet = [0xFF, 0xFF, motor_id, length, instruction] + parameters
        
        # Checksum 계산 (LeRobot의 _checksum 함수와 동일한 논리)
        checksum = ~(sum(packet[2:]) & 0xFF) & 0xFF
        packet.append(checksum)
        
        self.ser.write(bytearray(packet))
        return self.ser.read(100) # 응답 패킷 읽기 (생략 가능하나 확인용으로 권장)
    '''

    def _make_packet(self, motor_id, instruction, parameters):
        length = len(parameters) + 2
        packet = [0xFF, 0xFF, motor_id, length, instruction] + parameters
        checksum = (~(sum(packet[2:]) & 0xFF)) & 0xFF
        packet.append(checksum)
        return bytearray(packet)
    
    def _write_only(self, motor_id, instruction, parameters):
        """
        쓰기 전용: 응답을 읽지 않음
        → set_position, set_torque 등에 사용
        → 50ms 낭비 없음
        """
        self.ser.write(self._make_packet(motor_id, instruction, parameters))

    def _write_and_read(self, motor_id, instruction, parameters, resp_bytes=8):
        """
        읽기용: 정확히 resp_bytes만 읽음
        → get_position 등에 사용
        → 8바이트 도착하면 즉시 리턴 (timeout 낭비 없음)
        """
        self.ser.reset_input_buffer()  # 이전 잔여 바이트 제거
        self.ser.write(self._make_packet(motor_id, instruction, parameters))
        return self.ser.read(resp_bytes)  # 정확히 8바이트만 기다림

    
    # ---- Generic read/write helpers (추가) ----
    def write_u16(self, motor_id, reg_addr, value):
        lo = value & 0xFF
        hi = (value >> 8) & 0xFF
        self._write_only(motor_id, 0x03, [reg_addr & 0xFF, lo, hi])

    def read_u16(self, motor_id, reg_addr):
        resp = self._write_and_read(motor_id, 0x02, [reg_addr & 0xFF, 2], resp_bytes=8)
        if len(resp) < 8:
            return None
        return ((resp[6] << 8) | resp[5]) & 0x0FFF

    def set_torque(self, motor_id, enable):
        """모터의 토크를 켜거나 끕니다 (1: On, 0: Off)"""
        # [주소, 값] 순서로 파라미터 전달
        self._write_only(motor_id, 0x03, [self.REG_TORQUE_ENABLE, 1 if enable else 0])

    def set_position(self, motor_id, position):
        """목표 위치로 이동 (position: 0 ~ 4095)"""
        # 16비트 데이터를 2개의 8비트로 분리 (Little Endian)
        pos_low = position & 0xFF
        pos_high = (position >> 8) & 0xFF
        self._write_only(motor_id, 0x03, [self.REG_GOAL_POSITION, pos_low, pos_high])

    def get_position(self, motor_id):
        resp = self._write_and_read(motor_id, 0x02, [self.REG_PRESENT_POSITION, 2], resp_bytes=8)
        if len(resp) < 8:
            return None
        if resp[0] != 0xFF or resp[1] != 0xFF:
            return None
        if resp[2] != motor_id:
            return None
        pos = ((resp[6] << 8) | resp[5]) & 0x0FFF
        return pos
    
    def sync_write_position(self, motor_ids, positions):
        """
        여러 모터에 목표 위치를 한꺼번에 전송 (Sync Write)
        - motor_ids: [1, 2, 3, 4, 5, 6]
        - positions: [2048, 1024, ...]
        """
        # 0x83: SYNC WRITE instruction
        # 파라미터 구조: [시작주소, 데이터길이, ID1, Data1_L, Data1_H, ID2, Data2_L, Data2_H, ...]
        start_address = self.REG_GOAL_POSITION # 42
        data_length = 2 # 위치값은 2바이트
        
        parameters = [start_address, data_length]
        for m_id, pos in zip(motor_ids, positions):
            parameters.append(m_id)
            parameters.append(pos & 0xFF)         # Low byte
            parameters.append((pos >> 8) & 0xFF)  # High byte
        
        # ID 0xFE는 전체 모터에게 보내는 Broadcast ID입니다.
        # Sync Write는 응답을 읽지 않아야 하므로 _write_only를 사용합니다.
        self._write_only(0xFE, 0x83, parameters)
    
     # ---- 캘리브레이션 레지스터 write/read (추가) ----
    def set_homing_offset(self, motor_id, offset):
        """
        STS3215의 Offset(홈 오프셋) 레지스터는 일반 2의 보수가 아니라
        부호+절대값(sign-magnitude) 방식입니다: 11번 비트(0x800)=부호, 0~10번 비트=크기(0~2047).
        일반 2의 보수로 쓰면 음수일 때 완전히 다른 값이 들어가 위치가 크게 어긋납니다.
        """
        offset = int(offset)
        magnitude = min(abs(offset), 2047)
        encoded = magnitude | (0x800 if offset < 0 else 0)
        self.write_u16(motor_id, self.REG_HOMING_OFFSET, encoded)

    def read_homing_offset(self, motor_id):
        raw = self.read_u16(motor_id, self.REG_HOMING_OFFSET)
        if raw is None:
            return None
        magnitude = raw & 0x7FF
        return -magnitude if (raw & 0x800) else magnitude

    def set_position_limits(self, motor_id, min_pos, max_pos):
        self.write_u16(motor_id, self.REG_MIN_POSITION_LIMIT, int(min_pos))
        self.write_u16(motor_id, self.REG_MAX_POSITION_LIMIT, int(max_pos))

    def read_position_limits(self, motor_id):
        mn = self.read_u16(motor_id, self.REG_MIN_POSITION_LIMIT)
        mx = self.read_u16(motor_id, self.REG_MAX_POSITION_LIMIT)
        return mn, mx

    def set_torque_limit(self, motor_id, value=1000):
        """최대 출력 토크 설정 (0~1000 = 0~100%)"""
        self.write_u16(motor_id, self.REG_TORQUE_LIMIT, int(value))

    def read_torque_limit(self, motor_id):
        return self.read_u16(motor_id, self.REG_TORQUE_LIMIT)


