# robot_arm — SO-101 ACT (depth-top-zoom) 추론 + 재학습 패키지

`107_ACT_inference_depth_lerobot_official.py`를 그대로 실행하는 데 필요한 모든 것과,
같은 데이터셋으로 재학습(또는 새 데이터 수집)까지 할 수 있는 것들을 모아둔 폴더입니다.

## 폴더 구성

```
robot_arm/
├── 107_ACT_inference_depth_lerobot_official.py   # 추론 실행 스크립트 (메인)
├── 105_ACT_training_lerobot_official_depth_top_zoom.sh  # 재학습 스크립트
├── 103_data_collect_using_teleop_depth_top.py    # 새 데이터 수집 스크립트 (리더+팔로워 둘 다 필요)
├── debug_depth_top_camera_preview.py             # 카메라(top/top_depth/wrist) 연결 확인용
├── motor_control.py / orbbec_color_camera.py / object_zoom.py  # 공용 모듈
├── full_arm_calibration_follower.json / _leader.json  # 서보 캘리브레이션 값
├── 61-orbbec-astra.rules                         # Orbbec 카메라 udev 권한 규칙
├── openni2_redist/                                # OpenNI2 드라이버(Orbbec SDK) 런타임
├── lerobot_official_act_depth_top_zoom/
│   └── checkpoints/last/pretrained_model/         # 학습된 ACT 체크포인트 (591MB급)
├── hf_dataset_cache/my_robot_task_depth_top_zoom/ # 학습에 쓴 원본 lerobot 데이터셋
└── requirements.txt
```

## 1. 환경 설치

원본은 Python 3.12 conda 환경(`lerobot_312`)에서 실행했습니다. 동일하게 맞추는 걸 권장합니다.

```bash
conda create -n lerobot_312 python=3.12 -y
conda activate lerobot_312

# lerobot은 특정 커밋에 고정 (requirements.txt 주석 참고)
pip install "git+https://github.com/huggingface/lerobot.git@1e3a158e1395db7e5ac7639f993902ed85748a57"

pip install -r requirements.txt
```

## 2. 하드웨어 준비

- **SO-101 팔로워 로봇팔** (Feetech 서보, USB-시리얼 연결). 추론만 할 거면 팔로워만 있으면 됩니다
  (리더 암은 데이터 수집/텔레옵 할 때만 필요).
- **Orbbec Astra S** RGB+Depth 카메라 (탑뷰) + **일반 UVC 웹캠** (손목캠).
- Orbbec 카메라 USB 권한 등록:
  ```bash
  sudo cp 61-orbbec-astra.rules /etc/udev/rules.d/
  sudo udevadm control --reload-rules && sudo udevadm trigger
  ```
- 카메라가 제대로 열리는지 먼저 확인:
  ```bash
  python3 debug_depth_top_camera_preview.py
  ```
- **캘리브레이션은 로봇팔 개체마다 다릅니다.** `full_arm_calibration_follower.json`은
  원본 팔로워 암 기준 값이라 그대로 쓰면 안 맞을 수 있습니다 — 받는 쪽 로봇팔로 새로
  캘리브레이션해서 이 JSON을 교체하는 걸 권장합니다.
- `107_ACT_inference_depth_lerobot_official.py` 상단의 `FOLLOWER_PORT = "/dev/ttyACM0"`도
  받는 쪽 환경에서 실제 포트로 바꿔야 합니다 (USB 재연결마다 바뀔 수 있음).
- 손목캠 인덱스(`cap_wrist = ThreadedCamera(4)`, 107번 스크립트 224번째 줄 부근)도
  받는 쪽 장치 번호에 맞게 바꿔야 합니다.
- `TASK_TEXT = "pick up the red box and drop it in the black bin"` — 학습 데이터가
  "빨간 상자를 검은 통에 담기" 시나리오라서, 실제로도 같은 물체/색을 준비해야 정확히 동작합니다.

## 3. 추론 실행 (바로 되는 것)

```bash
cd robot_arm
python3 107_ACT_inference_depth_lerobot_official.py
```

체크포인트 경로(`CHECKPOINT_DIR`)는 이미 `lerobot_official_act_depth_top_zoom/checkpoints/last/pretrained_model`을
가리키도록 되어 있고, 이 폴더 구조 그대로 들어있으니 별도 설정 없이 바로 실행됩니다.

## 4. 재학습 (같은 데이터셋으로 처음부터 다시 학습)

데이터셋을 HuggingFace lerobot 캐시 경로로 복사(또는 심볼릭 링크)해야 합니다:

```bash
mkdir -p ~/.cache/huggingface/lerobot
cp -r hf_dataset_cache/my_robot_task_depth_top_zoom ~/.cache/huggingface/lerobot/
```

그 다음 학습 스크립트 실행 (스크립트 안의 `PYTHON=` 경로를 본인 conda env의
`lerobot-train` 실행파일 경로로 수정하세요):

```bash
bash 105_ACT_training_lerobot_official_depth_top_zoom.sh
```

기본 GPU 8GB(RTX 5050 Laptop 기준 batch_size=32)에서 검증됐습니다. VRAM이 더 적으면
`--batch_size`를 낮추세요.

## 5. 새 데이터 직접 수집하고 싶을 때

리더+팔로워 암이 모두 연결된 상태에서:

```bash
python3 103_data_collect_using_teleop_depth_top.py
```

리더 암으로 시연하면 top/top_depth/top_zoom/wrist 4개 카메라 관측과 관절 상태가
같이 기록되어 위 4번 학습 스크립트가 기대하는 `my_robot_task_depth_top_zoom` 데이터셋
포맷으로 저장됩니다.
