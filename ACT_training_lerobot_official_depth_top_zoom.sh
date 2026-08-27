#!/bin/bash

# repo_id=my_robot_task_depth_top_zoom (Astra S 깊이카메라 탑뷰, 4카메라:
# top/top_depth/top_zoom/wrist). top_zoom은 object_zoom.py가 HSV 색상 검출로
# 집을 물체(현재: 빨간 상자) 주변을 동적으로 확대한 채널 - pick 정밀도 보강용.

# 사전 준비 (최초 실행 전 1회 - 아직 자동화 안 됨, 학습 전 아래 파이썬 한 줄 실행):
#   python3 -c "
#   import json
#   p = '$HOME/.cache/huggingface/lerobot/my_robot_task_depth_top_zoom/meta/info.json'
#   info = json.load(open(p))
#   for k in ['observation.images.top','observation.images.top_depth','observation.images.top_zoom','observation.images.wrist']:
#       info['features'][k]['names'] = ['channels','height','width']
#   json.dump(info, open(p,'w'), indent=4)
#   "
#   (lerobot 공식 코드의 dataset_to_policy_features가 3개짜리 축 이름을 기대해서 생기는
#   IndexError를 피하기 위함 - data_collect_using_teleop_depth_top.py는 이미
#   ["channels","height","width"]로 저장하도록 고쳐뒀으니, 그 스크립트로 처음부터
#   모은 데이터셋이면 이 단계 생략 가능. 혹시 IndexError 나면 위 스니펫 실행할 것.)
#
# 사용법:
#   bash ACT_training_lerobot_official_depth_top_zoom.sh
set -e

cd "$(dirname "$0")"

PYTHON=/home/song/miniconda3/envs/lerobot_312/bin/lerobot-train
OUT_DIR="./lerobot_official_act_depth_top_zoom"

if [ -d "$OUT_DIR" ]; then
    echo "[경고] $OUT_DIR 가 이미 있습니다. lerobot-train은 기존 output_dir을 덮어쓰지 않고 에러를 냅니다."
    echo "       이어서 학습하려면 --resume=true를, 새로 시작하려면 폴더를 지우거나 이름을 바꿔주세요."
fi

"$PYTHON" \
    --dataset.repo_id=my_robot_task_depth_top_zoom \
    --dataset.eval_split=0.15 \
    --policy.type=act \
    --policy.device=cuda \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.push_to_hub=false \
    --output_dir="$OUT_DIR" \
    --job_name=so101_act_official_depth_top_zoom \
    --steps=17500 \
    --batch_size=32 \
    --eval_steps=1000 \
    --save_freq=2000 \
    --log_freq=200 \
    --save_checkpoint=true \
    --wandb.enable=false
