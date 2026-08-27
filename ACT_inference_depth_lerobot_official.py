"""
107_ACT_inference_lerobot_official.py 의 깊이카메라 탑뷰 버전.
105_ACT_training_lerobot_official_depth_top.sh 로 학습한 체크포인트
(lerobot_official_act_depth_top)로 추론한다.

107.py와의 차이는 오직 카메라 쪽뿐이다:
  - 탑뷰가 일반 웹캠(cv2.VideoCapture) 대신 Orbbec Astra S RGB+Depth
    (orbbec_color_camera.ThreadedOrbbecRGBDCamera)로 바뀌었고, 카메라 관측이
    "top"/"wrist" 2개에서 "top"/"top_depth"/"wrist" 3개로 늘었다.
  - 깊이 정규화 범위(DEPTH_MIN_MM/DEPTH_MAX_MM)는 data_collect_using_teleop_depth_top.py로
    학습 데이터를 모을 때 쓴 값(350~800mm)과 반드시 동일해야 한다. 다르면 모델이 학습 때
    본 적 없는 깊이 이미지 분포를 보게 돼 예측이 어긋난다.
  - 모델 로딩/select_action()/temporal ensembling/모터 제어 로직은 107.py와 동일.

[중요] 이 스크립트는 팔로워 암만 있으면 된다 (리더 암 연결 불필요, 학습된 정책이 카메라
입력만으로 자율 동작함). 다만 포트는 USB 재연결마다 바뀔 수 있어서(실제로 이 세션
안에서도 몇 번 바뀐 걸 확인함) 아래 FOLLOWER_PORT가 최신 상태와 다를 수 있다. 그래서
실행 시작 시 팔로워 그리퍼의 torque_limit이 500(팔로워에 설정해둔 값)인지 자동으로
확인해서, 포트가 잘못됐으면 바로 멈추고 물어본다.
"""
import os
import threading
import time
import traceback
import cv2
import numpy as np
import torch
from motor_control import MiniFeetechDriver
import json

from lerobot.policies.act import ACTPolicy
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
from lerobot.policies import make_pre_post_processors

from orbbec_color_camera import ThreadedOrbbecRGBDCamera
from object_zoom import ObjectZoomTracker, DEFAULT_HSV_RANGES

CAMERA_KEYS = ["top", "top_depth", "top_zoom", "wrist"]  # 학습 때와 동일한 순서/개수여야 함
TASK_TEXT = "pick up the red box and drop it in the black bin"  # 데이터 수집 때 넣었던 task 문자열과 동일하게

# my_robot_task_depth_top_zoom 데이터셋(top_zoom 채널 추가된 버전)으로 새로 학습한 체크포인트.
# checkpoints/last는 가장 최근 저장분을 가리키는 심볼릭 링크.
CHECKPOINT_DIR = "./lerobot_official_act_depth_top_zoom/checkpoints/last/pretrained_model"

# data_collect_using_teleop_depth_top.py 의 DEPTH_MIN_MM/DEPTH_MAX_MM과 반드시 일치시킬 것.
DEPTH_MIN_MM = 350
DEPTH_MAX_MM = 800

# top_zoom(집을 물체 동적 확대뷰) 크롭 크기 + 색상 범위. 데이터 수집 스크립트와 반드시 일치시킬 것.
OBJECT_ZOOM_CROP_SIZE = 200
TARGET_HSV_RANGES = DEFAULT_HSV_RANGES

# 팔로워 포트. USB 재연결마다 바뀔 수 있다. 아래 값이 틀리면 실행 시 자동 시그니처 체크
# (FOLLOWER_GRIPPER_TORQUE_SIGNATURE)에서 걸러진다.
FOLLOWER_PORT = "/dev/ttyACM0"

# 팔로워 그리퍼의 torque_limit=500 (data_collect_using_teleop_depth_top.py에서 설정해둔 값).
# 리더는 전 관절 1000이라 이 값으로 포트가 실제 팔로워인지 구분할 수 있다.
FOLLOWER_GRIPPER_TORQUE_SIGNATURE = 500

# 팔로워 관절별 torque_limit. data_collect_using_teleop_depth_top.py와 동일한 값을 써서,
# 재캘리브레이션 시 wrist_flex/gripper의 낮은 토크 설정(그리퍼가 상자를 으스러뜨리지 않도록)이
# 지워지지 않게 한다.
FOLLOWER_TORQUE_LIMITS = {
    "shoulder_pan": 1000,
    "shoulder_lift": 1000,
    "elbow_flex": 1000,
    "wrist_flex": 500,
    "wrist_roll": 1000,
    "gripper": 500,
}

# wrist_roll(5번 관절) 홈 포즈. data_collect_using_teleop_depth_top.py와 동일한 값.
# 학습 데이터가 항상 이 각도에서 시작했으므로, 추론도 같은 초기 자세에서 시작해야
# 모델이 본 적 있는 상태 분포에 가깝다.
HOME_WRIST_ROLL_POS = 2025

# [디버그 전용] 설정 시 모델이 보는 원본 top/top_depth/wrist 프레임을 1초 간격으로 저장.
DEBUG_FRAME_DIR = os.environ.get("ACT_DEBUG_FRAME_DIR")
if DEBUG_FRAME_DIR:
    os.makedirs(DEBUG_FRAME_DIR, exist_ok=True)
_last_debug_save = [0.0]
_debug_frame_idx = [0]


class ThreadedCamera:
    """손목캠(일반 UVC 웹캠)용. cv2.VideoCapture.read()를 백그라운드 스레드에서 계속
    돌리고, 메인 스레드는 항상 최신 프레임을 즉시 가져다 쓴다. USB 허브 병목으로
    read()가 몇 초씩 멈추면서(select() timeout) 추론 루프 전체가 멈춰 로봇이 안
    움직이던 문제를 해결하기 위함. 좌우 반전(mirror) 보정 옵션을 갖고 있지만,
    2026-08-26 확인 결과 반전 없이(원본 그대로) 쓰는 게 실제로 맞는 방향이라
    (학습 데이터도 그렇게 재인코딩됨) 기본값을 False로 둔다."""

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
        # CAP_PROP_BRIGHTNESS 기본값이 드라이버/전원 이벤트에 따라 0(최저)으로 잡힐 때가
        # 있어서 매번 명시적으로 재설정한다. 학습 데이터 수집 때와 밝기를 맞춰야 함
        # (data_collect_using_teleop_depth_top.py와 동일한 값).
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


def load_policy(checkpoint_dir, device):
    policy = ACTPolicy.from_pretrained(checkpoint_dir)
    policy.to(device)
    policy.eval()

    # 107.py와 동일한 이유로 공식 temporal ensembling으로 교체 (매 스텝 새로 예측 +
    # 과거 예측과 가중평균 -> 한 번 나쁜 청크를 예측해도 50step 내내 눈 감고 그대로
    # 실행하는 문제를 완화).
    policy.config.temporal_ensemble_coeff = 0.01  # ACT 논문 기본값
    policy.temporal_ensembler = ACTTemporalEnsembler(
        policy.config.temporal_ensemble_coeff, policy.config.chunk_size
    )

    preprocess, postprocess = make_pre_post_processors(policy.config, pretrained_path=checkpoint_dir)
    return policy, preprocess, postprocess


def to_input_tensor(frame_bgr, device):
    img_resized = cv2.resize(frame_bgr, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_torch = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return img_torch.unsqueeze(0).to(device)  # (1, 3, 224, 224)


def apply_calibration_to_servos(driver, cfg, joint_names, torque_limits):
    # USB 재연결/전원 이벤트마다 서보의 homing_offset이 초기화되는 경우가 있어서,
    # 매 실행 시작 시 저장된 캘리브레이션 JSON 값을 서보 레지스터에 다시 써준다.
    for name in joint_names:
        c = cfg[name]
        motor_id = c["id"]
        driver.set_torque(motor_id, False)
        time.sleep(0.15)
        driver.set_homing_offset(motor_id, c["homing_offset"])
        time.sleep(0.1)
        driver.set_position_limits(motor_id, c["range_min"], c["range_max"])
        time.sleep(0.1)
        driver.set_torque_limit(motor_id, torque_limits[name])
        time.sleep(0.1)
    print(">> 캘리브레이션 값 재적용 완료")


def verify_follower_port(driver, gripper_id):
    """포트가 USB 재연결로 바뀌어 엉뚱한 장치에 연결됐을 가능성을 자동 감지.
    팔로워 그리퍼는 torque_limit=500으로 설정돼 있어서, 이 값으로 실제 팔로워가
    맞는지 확인한다."""
    tl = driver.read_torque_limit(gripper_id)
    if tl != FOLLOWER_GRIPPER_TORQUE_SIGNATURE:
        print(f"[경고] 그리퍼 torque_limit={tl} (팔로워 시그니처={FOLLOWER_GRIPPER_TORQUE_SIGNATURE}와 다름)")
        print("       포트가 팔로워가 아닌 다른 장치일 수 있습니다 (USB 재연결로 ACM 번호가 바뀌었을 가능성).")
        print("       FOLLOWER_PORT 값을 확인하고 다시 실행하는 걸 권장합니다.")
        answer = input("       그래도 이 포트를 팔로워로 간주하고 계속할까요? [y/N] ").strip().lower()
        if answer != "y":
            print("중단합니다.")
            return False
    return True


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, preprocess, postprocess = load_policy(CHECKPOINT_DIR, device)
    policy.reset()  # 내부 액션 큐 초기화
    print(f"정식 lerobot ACT(depth-top) 체크포인트 로드 완료: {CHECKPOINT_DIR} (device={device})")

    driver = MiniFeetechDriver(port=FOLLOWER_PORT)

    with open("full_arm_calibration_follower.json", "r") as f:
        follower_cfg = json.load(f)
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    f_ids = [follower_cfg[n]["id"] for n in joint_names]

    if not verify_follower_port(driver, follower_cfg["gripper"]["id"]):
        return

    apply_calibration_to_servos(driver, follower_cfg, joint_names, FOLLOWER_TORQUE_LIMITS)

    object_zoom_tracker = ObjectZoomTracker(crop_size=OBJECT_ZOOM_CROP_SIZE, hsv_ranges=TARGET_HSV_RANGES)

    cap_top = ThreadedOrbbecRGBDCamera(depth_min_mm=DEPTH_MIN_MM, depth_max_mm=DEPTH_MAX_MM)
    cap_wrist = ThreadedCamera(4)  # udevadm으로 확인: USB_2.0_PC_Cam이 손목캠
    time.sleep(0.5)

    if not cap_top.isOpened() or not cap_wrist.isOpened():
        print("[오류] 카메라를 열 수 없습니다. debug_depth_top_camera_preview.py로 먼저 확인하세요.")
        cap_top.release()
        cap_wrist.release()
        return

    for f_id in f_ids:
        driver.set_torque(f_id, True)

    # [주석처리] wrist_roll 홈 포즈 이동. 필요하면 아래 4줄 주석 해제.
    # wrist_roll_id = follower_cfg["wrist_roll"]["id"]
    # print(f">> wrist_roll 홈 포즈로 이동 중... (raw {HOME_WRIST_ROLL_POS})")
    # driver.set_position(wrist_roll_id, HOME_WRIST_ROLL_POS)
    # time.sleep(0.8)
    # print(">> 홈 포즈 이동 완료")

    print("로봇 추론 시작 (정식 lerobot ACT, depth-top)... 'q'를 누르면 종료합니다.")

    alpha = 0.7  # EMA 필터 계수. 낮을수록 목표값에 천천히 다가가서(=느리게) 움직인다.
    # [테스트] 0.3->0.7로 올려서 반응성 높임 - pick 정밀도 부족이 데이터 문제가 아니라
    # 이 스무딩이 미세 보정을 뭉개서 생기는 문제인지 확인하기 위한 실험. 값을 되돌리려면
    # 0.3으로.
    prev_goals = None
    target_dt = 1.0 / 30  # 데이터 수집 때(fps=30)와 같은 주기로 제한

    try:
        with torch.no_grad():
            while True:
                t_loop_start = time.time()
                ret_top, frame_top, frame_top_depth = cap_top.read()
                ret_wrist, frame_wrist = cap_wrist.read()

                if not ret_top or frame_top is None or frame_top_depth is None or not ret_wrist:
                    print("[경고] 카메라 프레임을 읽지 못했습니다. 재시도합니다.")
                    continue

                frame_top_zoom, zoom_detected = object_zoom_tracker.update(frame_top)

                combined_view = cv2.hconcat(
                    [
                        cv2.resize(frame_top, (320, 240)),
                        cv2.resize(frame_top_depth, (320, 240)),
                        cv2.resize(frame_top_zoom, (320, 240)),
                        cv2.resize(frame_wrist, (320, 240)),
                    ]
                )
                cv2.imshow("win (top | top_depth | top_zoom | wrist) - lerobot official ACT depth-top", combined_view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if DEBUG_FRAME_DIR:
                    now = time.time()
                    if now - _last_debug_save[0] >= 1.0:
                        _last_debug_save[0] = now
                        idx = _debug_frame_idx[0]
                        _debug_frame_idx[0] += 1
                        cv2.imwrite(os.path.join(DEBUG_FRAME_DIR, f"{idx:03d}_top.jpg"), frame_top)
                        cv2.imwrite(os.path.join(DEBUG_FRAME_DIR, f"{idx:03d}_top_depth.jpg"), frame_top_depth)
                        cv2.imwrite(os.path.join(DEBUG_FRAME_DIR, f"{idx:03d}_top_zoom.jpg"), frame_top_zoom)
                        cv2.imwrite(os.path.join(DEBUG_FRAME_DIR, f"{idx:03d}_wrist.jpg"), frame_wrist)

                img_top = to_input_tensor(frame_top, device)
                img_top_depth = to_input_tensor(frame_top_depth, device)
                img_top_zoom = to_input_tensor(frame_top_zoom, device)
                img_wrist = to_input_tensor(frame_wrist, device)

                current_states = []
                for name in joint_names:
                    cfg = follower_cfg[name]
                    pos = driver.get_position(cfg["id"])
                    if pos is None:
                        current_states = None
                        break
                    ratio = (pos - cfg["range_min"]) / (cfg["range_max"] - cfg["range_min"])
                    current_states.append(max(0.0, min(1.0, ratio)))

                if current_states is None:
                    print("[경고] 관절 상태 읽기 실패. 이번 스텝 스킵.")
                    continue

                state_torch = torch.tensor(current_states).float().unsqueeze(0).to(device)

                # 1. 모델 예측: select_action()이 내부 청크 큐를 관리 (50 step마다 새로 예측)
                obs = {
                    "observation.images.top": img_top,
                    "observation.images.top_depth": img_top_depth,
                    "observation.images.top_zoom": img_top_zoom,
                    "observation.images.wrist": img_wrist,
                    "observation.state": state_torch,
                    "task": TASK_TEXT,
                }
                obs = preprocess(obs)
                action = policy.select_action(obs)
                action = postprocess(action)
                target_action = action.squeeze(0).cpu().numpy()  # (6,)

                # 2. 로봇에게 명령 전달 (정규화 해제)
                raw_goals = []
                for i, name in enumerate(joint_names):
                    cfg = follower_cfg[name]
                    # 모델 출력이 정규화 범위(0~1)를 살짝 벗어날 수 있어 방어적으로 클램프
                    val = max(0.0, min(1.0, float(target_action[i])))
                    actual_pos = int(val * (cfg["range_max"] - cfg["range_min"]) + cfg["range_min"])
                    raw_goals.append(actual_pos)

                if prev_goals is None:
                    filtered_goals = raw_goals
                else:
                    filtered_goals = [
                        int(alpha * g + (1 - alpha) * pg) for g, pg in zip(raw_goals, prev_goals)
                    ]
                prev_goals = filtered_goals

                print(f"Pred Goals: {filtered_goals}")
                driver.sync_write_position(f_ids, filtered_goals)

                elapsed = time.time() - t_loop_start
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)

    except KeyboardInterrupt:
        print("\n사용자 중단 (Ctrl+C)")
    except Exception:
        print("[오류] 추론 루프 중 예외 발생:")
        traceback.print_exc()
    finally:
        cap_top.release()
        cap_wrist.release()
        cv2.destroyAllWindows()
        if "driver" in locals() and driver is not None:
            try:
                for f_id in f_ids:
                    driver.set_torque(f_id, False)
                print("✅ 모든 모터 토크 해제 완료")
            except Exception as e:
                print(f"⚠️ 토크 해제 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
