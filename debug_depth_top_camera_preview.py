"""
탑뷰로 새로 붙인 Orbbec Astra S(깊이카메라)의 컬러 스트림 + 깊이(depth) 스트림 + 기존
손목캠을 나란히 미리보기. 실제 녹화(data_collect_using_teleop_depth_top.py) 전에
카메라가 정상적으로 열리고 화면이 잘 나오는지, 그리고 깊이 정규화 범위
(depth_min_mm/depth_max_mm)가 실제 작업대 거리에 맞는지 먼저 이걸로 확인할 것.

콘솔에 1초마다 현재 프레임의 유효(0이 아닌) 깊이 min/max(mm)를 출력해준다.
작업대/상자가 화면에 있는 상태에서 그 값을 보고, ThreadedOrbbecRGBDCamera와
data_collect_using_teleop_depth_top.py의 depth_min_mm/depth_max_mm을
실제 거리에 맞게 조정할 것 (너무 넓게 잡으면 깊이 변화가 화면상 미세한 명암차로만
보여서 정책이 학습하기 어려워짐).

준비:
  1) pip install primesense  (lerobot_312 conda env, 이미 설치됨)
  2) udev 규칙 등록 (이미 적용 완료됨. USB 재연결 시 다시 확인 필요할 수 있음):
     sudo cp 61-orbbec-astra.rules /etc/udev/rules.d/
     sudo udevadm control --reload-rules && sudo udevadm trigger

실행: python3 debug_depth_top_camera_preview.py [--wrist <손목캠 index>]
'q' 키로 종료.
"""

import argparse
import time

import cv2
import numpy as np

from orbbec_color_camera import ThreadedOrbbecRGBDCamera
from object_zoom import ObjectZoomTracker, DEFAULT_HSV_RANGES


def open_wrist_camera(index, width=320, height=240):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # CAP_PROP_BRIGHTNESS 기본값이 드라이버/전원 이벤트에 따라 0(최저)으로 잡힐 때가 있어서
    # 매번 명시적으로 재설정한다 (data_collect_using_teleop_depth_top.py와 동일한 값).
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 125)
    return cap


def main():
    parser = argparse.ArgumentParser(description="Orbbec Astra S(탑뷰 RGB+Depth) + 손목캠 미리보기")
    parser.add_argument("--wrist", type=int, default=4, help="손목 카메라 index (기본 4)")
    parser.add_argument("--depth-min", type=int, default=350, help="깊이 정규화 최소값(mm)")
    parser.add_argument("--depth-max", type=int, default=800, help="깊이 정규화 최대값(mm)")
    args = parser.parse_args()

    print("=" * 60)
    print("[Orbbec Astra S 컬러+깊이 스트림 여는 중...]")
    object_zoom_tracker = ObjectZoomTracker(hsv_ranges=DEFAULT_HSV_RANGES)
    cap_top = ThreadedOrbbecRGBDCamera(depth_min_mm=args.depth_min, depth_max_mm=args.depth_max)
    if not cap_top.isOpened():
        print("[오류] Astra S를 열 수 없습니다.")
        print("  - udev 규칙을 아직 적용 안 했거나, 케이블을 재연결하지 않았을 수 있습니다.")
        print("  - `groups` 명령으로 현재 사용자가 video 그룹에 속하는지 확인하세요.")
        cap_top.release()
        return

    print(f"[손목캠 여는 중... index={args.wrist}]")
    cap_wrist = open_wrist_camera(args.wrist)
    if not cap_wrist.isOpened():
        print(f"[오류] 손목캠(index={args.wrist})을 열 수 없습니다.")
        cap_top.release()
        return

    print("모든 카메라 열림. 'q' 키를 누르면 종료됩니다.")
    print(f"현재 깊이 정규화 범위: {args.depth_min}mm ~ {args.depth_max}mm")
    print("=" * 60)

    fps_count = 0
    fps_start = time.time()
    current_fps = 0.0
    t_last_depth_report = time.time()

    try:
        while True:
            ret_top, frame_top, frame_depth = cap_top.read()
            ret_wrist, frame_wrist = cap_wrist.read()

            if not ret_top or frame_top is None or not ret_wrist or not frame_wrist.size:
                time.sleep(0.01)
                continue

            # 2026-08-26 확인: 손목캠은 반전 없이(원본 그대로) 쓰는 게 맞는 방향이라
            # cv2.flip 보정을 뺐다 (기존에 반전 보정을 넣었던 게 오히려 거꾸로였음).

            frame_zoom, zoom_detected = object_zoom_tracker.update(frame_top)

            now = time.time()
            if now - t_last_depth_report >= 1.0:
                t_last_depth_report = now
                raw_mm = cap_top.read_raw_depth_mm()
                if raw_mm is not None:
                    valid = raw_mm[raw_mm > 0]
                    if valid.size > 0:
                        print(f"[depth] 유효 범위: min={int(valid.min())}mm max={int(valid.max())}mm "
                              f"median={int(np.median(valid))}mm (유효 픽셀 비율 {valid.size / raw_mm.size:.0%})")
                    else:
                        print("[depth] 유효 픽셀 없음 (센서와 대상이 너무 가깝거나 멀 수 있음)")

            disp_top = frame_top.copy()
            disp_depth = frame_depth.copy()
            disp_zoom = frame_zoom.copy()
            disp_wrist = frame_wrist.copy()
            h = min(disp_top.shape[0], disp_depth.shape[0], disp_zoom.shape[0], disp_wrist.shape[0])
            disp_top = cv2.resize(disp_top, (int(disp_top.shape[1] * h / disp_top.shape[0]), h))
            disp_depth = cv2.resize(disp_depth, (int(disp_depth.shape[1] * h / disp_depth.shape[0]), h))
            disp_zoom = cv2.resize(disp_zoom, (int(disp_zoom.shape[1] * h / disp_zoom.shape[0]), h))
            disp_wrist = cv2.resize(disp_wrist, (int(disp_wrist.shape[1] * h / disp_wrist.shape[0]), h))
            combined = cv2.hconcat([disp_top, disp_depth, disp_zoom, disp_wrist])

            fps_count += 1
            if time.time() - fps_start >= 1.0:
                current_fps = fps_count / (time.time() - fps_start)
                fps_count = 0
                fps_start = time.time()

            cv2.putText(combined, f"TOP RGB  FPS:{current_fps:.1f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, "TOP DEPTH", (disp_top.shape[1] + 20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            zoom_label = "TOP ZOOM" + ("" if zoom_detected else " (lost)")
            zoom_color = (0, 255, 0) if zoom_detected else (0, 165, 255)
            cv2.putText(combined, zoom_label, (disp_top.shape[1] + disp_depth.shape[1] + 20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, zoom_color, 2)
            cv2.putText(combined, "WRIST", (disp_top.shape[1] + disp_depth.shape[1] + disp_zoom.shape[1] + 20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Depth-top preview (RGB | Depth | Zoom | wrist) - press q to quit", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap_top.release()
        cap_wrist.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
