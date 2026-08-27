"""
101_data_collect_using_teleop.py 의 깊이카메라 탑뷰 버전.

기존 탑뷰 USB 웹캠(index 6) 자리를 Orbbec Astra S 깊이카메라의 컬러+깊이 스트림으로
교체했다. Astra S는 OpenNI2로만 열리므로 카메라 캡처 부분만
orbbec_color_camera.ThreadedOrbbecRGBDCamera로 바뀌었고, 모터 제어/캘리브레이션/
녹화 로직은 101_data_collect_using_teleop.py와 동일하다.

깊이(mm) 맵도 실제로 모델 입력에 쓰기 위해 "observation.images.top_depth" 라는 새
카메라 키로 함께 저장한다. raw depth(mm, uint16)를 그대로 넣지 않고 지정된 범위로
정규화 + JET 컬러맵을 입혀 기존 top/wrist와 동일한 3채널 "video" 포맷으로 만들었다 —
그래야 LeRobot/ACT가 카메라 키마다 자동으로 붙이는 시각 인코더를 그대로 재사용할 수
있어서(정책 코드 수정 불필요), 그냥 카메라가 3대로 늘어난 것처럼 취급된다.

[중요] Astra S 같은 구조광(structured light) 깊이센서는 검은색 표면에서 IR 패턴이
거의 반사되지 않아 깊이값을 못 읽는다 — 실측해보니 이 태스크의 목표물인 "검은 상자"
윗면 대부분이 깊이 무효(0mm) 픽셀로 나온다 (debug_depth_top_camera_preview.py로
확인 가능). 즉 깊이 채널은 빨간 상자·팔 위치 등에는 도움이 되지만, 검은 상자 자체의
형상 정보는 거의 못 준다. RGB 채널은 검은 상자를 정상적으로 보여주므로 place 태스크
자체가 안 되는 건 아니지만, 이 한계는 인지하고 있을 것.

DEPTH_MIN_MM/DEPTH_MAX_MM은 실제 작업대까지의 거리(현재 세팅 실측: 약 380~714mm)에
맞춘 값이다. 카메라 마운트 높이가 바뀌면 debug_depth_top_camera_preview.py로 다시
확인 후 조정할 것.

기존에 웹캠 탑뷰로 모아둔 데이터셋(~/.cache/huggingface/lerobot/my_robot_task)과
섞이지 않도록 repo_id를 "my_robot_task_depth_top"으로 분리했다. 카메라 종류가
바뀌면 관측 분포가 달라지므로 같은 데이터셋에 이어붙이면 안 된다.

실행 전 준비:
  1) Astra S가 일반 사용자 권한으로 열리는지 debug_depth_top_camera_preview.py로 먼저 확인
     (udev 규칙 적용은 README/61-orbbec-astra.rules 안내 참고, sudo 필요 — 이미 적용됨)
  2) 리더/팔로워 포트가 실제로 맞는지 103_probe_leader_follower_port.py로 확인
     (USB 재연결마다 /dev/ttyACM* 번호가 바뀔 수 있음.
      2026-08-25 기준으로는 ACM0=리더, ACM1=팔로워로 확인됨)
"""

import time
import json
import queue
import threading
import cv2
import torch
from pathlib import Path
from motor_control import MiniFeetechDriver
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from orbbec_color_camera import ThreadedOrbbecRGBDCamera
from object_zoom import ObjectZoomTracker, DEFAULT_HSV_RANGES

DEPTH_MIN_MM = 350
DEPTH_MAX_MM = 800

# top_zoom(집을 물체 동적 확대뷰) 크롭 한 변 길이. 640x480 원본 기준.
OBJECT_ZOOM_CROP_SIZE = 200

# 지금 집는 물체(빨간 상자) 기준 HSV 범위. 물체를 바꾸면 이 값만 교체하면 됨
# (object_zoom.py의 DEFAULT_HSV_RANGES 참고 - 새 물체 색상 실측해서 바꿀 것).
TARGET_HSV_RANGES = DEFAULT_HSV_RANGES

# wrist_roll(5번 관절) 홈 포즈. 팔로워 raw 위치 기준 (2026-08-25 실측: 사용자가 돌려놓은 위치).
# 녹화 시작 직전에 팔로워를 이 위치로 먼저 이동시켜, 매 세션이 항상 같은 손목 각도에서
# 시작하도록 한다.
HOME_WRIST_ROLL_POS = 2025

# 팔로워 관절별 torque_limit. 기존 코드는 재캘리브레이션할 때마다 전 관절에 1000을
# 일괄로 다시 써서, wrist_flex/gripper에 설정돼 있던 500(그리퍼가 상자를 으스러뜨리지
# 않도록 낮춰둔 값으로 추정)을 매번 지우고 있었다. 2026-08-25 실측한 현재 값 그대로
# 고정해서 재캘리브레이션해도 이 설정이 유지되게 함.
FOLLOWER_TORQUE_LIMITS = {
    "shoulder_pan": 1000,
    "shoulder_lift": 1000,
    "elbow_flex": 1000,
    "wrist_flex": 500,
    "wrist_roll": 1000,
    "gripper": 500,
}
LEADER_TORQUE_LIMIT = 1000  # 리더는 전 관절 1000으로 균일 (실측 확인됨, 손으로 움직이는 용도라 원래와 동일하게 유지)


class ThreadedCamera:
    """손목캠(일반 UVC 웹캠)용. 101_data_collect_using_teleop.py와 동일하되,
    좌우 반전(mirror) 보정 옵션을 갖고 있지만, 2026-08-26 확인 결과 이 손목캠은 반전
    보정 없이(원본 그대로) 저장한 게 실제로 맞는 방향이었음 (기존 mirror=True 보정이
    오히려 거꾸로였음). 그래서 기본값을 False로 바꿈 — 기존 44개 에피소드도 전부
    원본 방향으로 재인코딩해서 일관성을 맞춰뒀다."""

    def __init__(self, index, width=320, height=240, mirror=False):
        self.index = index
        self.width = width
        self.height = height
        self.mirror = mirror
        self.cap = None

        self._lock = threading.Lock()
        self._ret = False
        self._frame = None
        self._running = True
        self._opened_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        self.cap = cv2.VideoCapture(self.index)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # 이 카메라(USB 2.0 PC Cam)는 CAP_PROP_BRIGHTNESS 기본값이 드라이버/전원 이벤트에
        # 따라 0(최저)으로 잡힐 때가 있어서(실측: 화면이 확 어두워짐, 학습 데이터 대비 mean
        # 밝기 190 -> 44) 매번 명시적으로 재설정한다. 125가 학습 데이터 밝기와 가장 가까움.
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 125)
        self._opened_event.set()

        while self._running:
            ret, frame = self.cap.read()
            if ret and self.mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._ret = ret
                if ret:
                    self._frame = frame

    def isOpened(self):
        self._opened_event.wait(timeout=5.0)
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        with self._lock:
            if self._frame is None:
                return self._ret, None
            return self._ret, self._frame.copy()

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


class TeleopRecorder:
    def __init__(
        self,
        leader_port,
        follower_port,
        repo_id="my_robot_task_depth_top_zoom",
        wrist_cam_index=4,
        task_description="pick up the red box and drop it in the black bin",
    ):
        # 1. 하드웨어 드라이버 설정
        self.leader = MiniFeetechDriver(port=leader_port)
        self.follower = MiniFeetechDriver(port=follower_port)

        # 2. 칼리브레이션 데이터 로드 (각 조인트의 범위를 알기 위함)
        with open("full_arm_calibration_leader.json", "r") as f:
            self.leader_cfg = json.load(f)
        with open("full_arm_calibration_follower.json", "r") as f:
            self.follower_cfg = json.load(f)

        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        self.f_ids = [self.follower_cfg[name]["id"] for name in self.joint_names]

        # 3. 카메라 설정: 탑뷰는 Astra S(OpenNI2), 손목캠은 기존과 동일하게 index로 지정
        self.wrist_cam_index = wrist_cam_index
        self.object_zoom_tracker = ObjectZoomTracker(crop_size=OBJECT_ZOOM_CROP_SIZE, hsv_ranges=TARGET_HSV_RANGES)

        # 4. LeRobot 데이터셋 설정
        self.repo_id = repo_id
        self.fps = 30
        self.task_description = task_description
        self.dataset = self._setup_dataset()

        # 5. 제어 및 필터 변수
        self.alpha = 0.3  # EMA 필터 계수
        self.prev_goals = {name: None for name in self.joint_names}
        self.is_recording = False

        # 6. 이미지 인코딩/저장(add_frame)을 메인 제어 루프에서 분리하기 위한 백그라운드 워커.
        self.frame_queue = queue.Queue()
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

    def _writer_loop(self):
        while True:
            item = self.frame_queue.get()
            if item is None:
                self.frame_queue.task_done()
                break
            frame_top, frame_top_depth, frame_top_zoom, frame_wrist, norm_states, norm_actions = item
            img_top_torch = self._to_chw_rgb_tensor(frame_top)
            img_top_depth_torch = self._to_chw_rgb_tensor(frame_top_depth)
            img_top_zoom_torch = self._to_chw_rgb_tensor(frame_top_zoom)
            img_wrist_torch = self._to_chw_rgb_tensor(frame_wrist)
            self.dataset.add_frame(
                {
                    "observation.images.top": img_top_torch,
                    "observation.images.top_depth": img_top_depth_torch,
                    "observation.images.top_zoom": img_top_zoom_torch,
                    "observation.images.wrist": img_wrist_torch,
                    "observation.state": torch.tensor(norm_states, dtype=torch.float32),
                    "action": torch.tensor(norm_actions, dtype=torch.float32),
                    "task": self.task_description,
                }
            )
            self.frame_queue.task_done()

    def _setup_dataset(self):
        dataset_path = Path(Path.home(), ".cache/huggingface/lerobot", self.repo_id)

        # names는 ["channels","height","width"]로 미리 맞춰둠 - lerobot 공식 학습 코드의
        # dataset_to_policy_features가 3개짜리 축 이름을 기대해서 ["color"](1개)로 두면
        # IndexError가 나는 걸 이미 여러 번 겪었음 (105번 학습 스크립트 참고).
        _img_feature = lambda: {"dtype": "video", "shape": (3, 224, 224), "names": ["channels", "height", "width"]}
        features = {
            "observation.images.top": _img_feature(),
            "observation.images.top_depth": _img_feature(),
            "observation.images.top_zoom": _img_feature(),
            "observation.images.wrist": _img_feature(),
            "observation.state": {"dtype": "float32", "shape": (6,)},
            "action": {"dtype": "float32", "shape": (6,)},
        }

        if dataset_path.exists():
            print(f"기존 데이터셋에 이어서 녹화합니다: {dataset_path}")
            return LeRobotDataset.resume(repo_id=self.repo_id, root=dataset_path)

        print(f"새 데이터셋을 생성합니다: {dataset_path}")
        return LeRobotDataset.create(repo_id=self.repo_id, fps=self.fps, features=features, root=dataset_path)

    def _apply_calibration_to_servos(self):
        for driver, cfg, torque_limits in (
            (self.leader, self.leader_cfg, None),
            (self.follower, self.follower_cfg, FOLLOWER_TORQUE_LIMITS),
        ):
            for name in self.joint_names:
                c = cfg[name]
                motor_id = c["id"]
                torque_limit = LEADER_TORQUE_LIMIT if torque_limits is None else torque_limits[name]
                driver.set_torque(motor_id, False)
                time.sleep(0.15)
                driver.set_homing_offset(motor_id, c["homing_offset"])
                time.sleep(0.1)
                driver.set_position_limits(motor_id, c["range_min"], c["range_max"])
                time.sleep(0.1)
                driver.set_torque_limit(motor_id, torque_limit)
                time.sleep(0.1)

                driver.ser.reset_input_buffer()
                readback = driver.read_homing_offset(motor_id)
                if readback != c["homing_offset"]:
                    print(f"  [경고] {name} homing_offset 재적용 검증 실패, 재시도 (목표={c['homing_offset']} 실제={readback})")
                    driver.set_homing_offset(motor_id, c["homing_offset"])
                    time.sleep(0.15)
        print(">> 캘리브레이션 값 재적용 완료")

    @staticmethod
    def _to_chw_rgb_tensor(frame_bgr):
        img_resized = cv2.resize(frame_bgr, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img_rgb).permute(2, 0, 1)

    def run(self):
        cap_top = ThreadedOrbbecRGBDCamera(depth_min_mm=DEPTH_MIN_MM, depth_max_mm=DEPTH_MAX_MM)
        cap_wrist = ThreadedCamera(self.wrist_cam_index)
        time.sleep(0.5)

        if not cap_top.isOpened() or not cap_wrist.isOpened():
            print(f"[오류] 카메라를 열 수 없습니다. (top=Astra S, wrist_index={self.wrist_cam_index})")
            print("       debug_depth_top_camera_preview.py 로 먼저 각각 열리는지 확인하세요.")
            cap_top.release()
            cap_wrist.release()
            return

        print("\n[작동 가이드]")
        print("- 'r' 키: 녹화 시작 | 's' 키: 에피소드 저장 | 'd' 키: 현재 에피소드 취소(잘못 잡았을 때) | 'q' 키: 종료")

        self._apply_calibration_to_servos()

        for f_id in self.f_ids:
            self.follower.set_torque(f_id, True)

        wrist_roll_id = self.follower_cfg["wrist_roll"]["id"]
        print(f">> wrist_roll 홈 포즈로 이동 중... (raw {HOME_WRIST_ROLL_POS})")
        self.follower.set_position(wrist_roll_id, HOME_WRIST_ROLL_POS)
        time.sleep(0.8)  # 모터가 실제로 도달할 때까지 대기
        # EMA 필터의 시작점도 홈 포즈로 맞춰둬야, 텔레옵 루프 첫 스텝에서 리더 위치로
        # 갑자기 확 튀지 않고 거기서부터 자연스럽게 이어짐
        self.prev_goals["wrist_roll"] = HOME_WRIST_ROLL_POS
        print(">> 홈 포즈 이동 완료")

        loop_count = 0
        t_prev_report = time.time()
        target_dt = 1.0 / self.fps

        try:
            while True:
                t0 = time.time()
                ret_top, frame_top, frame_top_depth = cap_top.read()
                ret_wrist, frame_wrist = cap_wrist.read()
                if not ret_top or frame_top is None or frame_top_depth is None or not ret_wrist or frame_wrist is None:
                    print("[경고] 카메라 프레임을 읽지 못했습니다. 다음 루프로 건너뜁니다.")
                    continue
                frame_top_zoom, zoom_detected = self.object_zoom_tracker.update(frame_top)
                t_cam = time.time()

                goals_raw = []
                norm_states = []
                norm_actions = []
                all_read_success = True

                self.leader.ser.reset_input_buffer()

                for name in self.joint_names:
                    l_cfg = self.leader_cfg[name]
                    f_cfg = self.follower_cfg[name]

                    l_pos = self.leader.get_position(l_cfg["id"])
                    f_pos = self.follower.get_position(f_cfg["id"])

                    if l_pos is None or f_pos is None:
                        all_read_success = False
                        break

                    l_ratio = (l_pos - l_cfg["range_min"]) / (l_cfg["range_max"] - l_cfg["range_min"])
                    l_ratio = max(0.0, min(1.0, l_ratio))

                    raw_goal = int(l_ratio * (f_cfg["range_max"] - f_cfg["range_min"]) + f_cfg["range_min"])

                    if self.prev_goals[name] is None:
                        filtered_goal = raw_goal
                    else:
                        filtered_goal = int(self.alpha * raw_goal + (1 - self.alpha) * self.prev_goals[name])

                    self.prev_goals[name] = filtered_goal
                    goals_raw.append(filtered_goal)

                    f_ratio = (f_pos - f_cfg["range_min"]) / (f_cfg["range_max"] - f_cfg["range_min"])
                    norm_states.append(max(0.0, min(1.0, f_ratio)))

                    target_ratio = (filtered_goal - f_cfg["range_min"]) / (f_cfg["range_max"] - f_cfg["range_min"])
                    norm_actions.append(max(0.0, min(1.0, target_ratio)))

                t_motor = time.time()

                t_encode = t_motor
                if all_read_success and len(goals_raw) == 6:
                    self.follower.sync_write_position(self.f_ids, goals_raw)

                    if self.is_recording:
                        self.frame_queue.put(
                            (frame_top, frame_top_depth, frame_top_zoom, frame_wrist, norm_states, norm_actions)
                        )
                        t_encode = time.time()

                loop_count += 1
                if loop_count % 15 == 0:
                    now = time.time()
                    hz = 15.0 / (now - t_prev_report) if now > t_prev_report else 0.0
                    t_prev_report = now
                    print(
                        f"[perf] {hz:.1f}Hz | cam={(t_cam - t0)*1000:.0f}ms "
                        f"motor={(t_motor - t_cam)*1000:.0f}ms "
                        f"add_frame={(t_encode - t_motor)*1000:.0f}ms "
                        f"recording={self.is_recording}",
                        flush=True,
                    )

                panels = [
                    ("TOP RGB", frame_top),
                    ("TOP DEPTH", frame_top_depth),
                    ("TOP ZOOM" + ("" if zoom_detected else " (lost)"), frame_top_zoom),
                    ("WRIST", frame_wrist),
                ]
                h = min(p.shape[0] for _, p in panels)
                resized = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for _, p in panels]
                combined = cv2.hconcat(resized)

                if self.is_recording:
                    cv2.putText(combined, "● RECORDING", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                x_off = 0
                for (label, _), r in zip(panels, resized):
                    color = (0, 255, 0) if "lost" not in label else (0, 165, 255)
                    cv2.putText(combined, label, (x_off + 20, combined.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    x_off += r.shape[1]

                cv2.imshow("Teleop & Record (RGB | Depth | Zoom | wrist)", combined)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("r") and not self.is_recording:
                    self.is_recording = True
                    # 이전 에피소드에서 마지막으로 검출된 물체 위치를 이어받지 않도록 리셋
                    self.object_zoom_tracker.reset()
                    print(">> 녹화 시작")
                elif key == ord("s") and self.is_recording:
                    self.is_recording = False
                    print(">> 프레임 저장 마무리 중...")
                    self.frame_queue.join()
                    self.dataset.save_episode()
                    print(">> 에피소드 저장 완료")
                elif key == ord("d") and self.is_recording:
                    self.is_recording = False
                    print(">> 에피소드 취소 중...")
                    self.frame_queue.join()
                    self.dataset.clear_episode_buffer()
                    print(">> 에피소드 취소 완료 (저장되지 않음)")
                elif key == ord("q"):
                    break

                elapsed = time.time() - t0
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)

        finally:
            cap_top.release()
            cap_wrist.release()
            cv2.destroyAllWindows()
            for f_id in self.f_ids:
                self.follower.set_torque(f_id, False)
            self.frame_queue.join()
            self.frame_queue.put(None)
            self.writer_thread.join()
            if hasattr(self, "dataset"):
                print("데이터셋 저장 마무리 중(finalize)...")
                self.dataset.finalize()
                del self.dataset


if __name__ == "__main__":
    # 포트는 USB 재연결마다 바뀔 수 있으니 103_probe_leader_follower_port.py로 실행 전 매번 확인 필요.
    # 2026-08-26: 물리 테스트(103_probe_leader_follower_port.py)로 ACM1=리더 확인됨
    recorder = TeleopRecorder(
        leader_port="/dev/ttyACM1",
        follower_port="/dev/ttyACM0",
        wrist_cam_index=4,
    )
    recorder.run()
