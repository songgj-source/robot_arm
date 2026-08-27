"""
집을 물체(지금은 빨간 상자)를 색상 검출로 찾아서 그 주변을 동적으로 확대(crop)하는 헬퍼.

배경(왜 top 이미지를 통째로 고정 크롭하지 않았는지):
검은 상자(place 대상)는 에피소드마다 화면 전체에 걸쳐 넓게 놓였어서(실측 확인됨),
탑뷰를 고정된 좁은 영역으로 크롭하면 검은 상자가 크롭 밖으로 잘려나가는 에피소드가
생겨서 이미 잘 되고 있는 place 성능을 해칠 위험이 있었다. 대신 기존 "top"(풀프레임)은
그대로 두고, 집을 물체 주변만 동적으로 확대한 뷰를 새 카메라 채널("top_zoom")로
"추가"한다 — pick에 필요한 정밀도는 이 채널이 보완하고, place에 필요한 넓은 시야는
기존 top이 그대로 담당하는 구조.

[일반화] 딥러닝 기반 학습형 탐지기가 아니라 HSV 색상 임계값으로 찾는 고정 규칙이라,
집는 물체가 바뀌면(예: 빨간 상자 -> 휴지뭉치) 이 색상 범위 자체를 그 물체에 맞게
바꿔줘야 한다. ObjectZoomTracker 생성 시 hsv_ranges를 인자로 넘기면 되고, 데이터
수집/추론 스크립트 쪽에는 TARGET_HSV_RANGES 상수 하나만 고치면 되도록 만들어뒀다
(로직 자체를 다시 짤 필요는 없음). 물체가 흰 작업대와 색이 확실히 대비되는 한
계속 이 방식을 쓸 수 있다.

검출 실패(그리퍼에 가려짐, 이미 집어서 물체가 테이블 위에 없음 등) 시에는 마지막으로
검출됐던 위치를 계속 사용한다 - 완전히 안 보였던 적은 없다고 가정(그리퍼가 물체를
쥐고 있으면 여전히 일부가 보임, 손목캠과 달리 탑뷰는 그리퍼가 물체를 완전히
가리는 경우가 드묾).
"""

import cv2
import numpy as np

# 빨간 상자 기준 HSV 범위 (이번 세션 분석에서 반복 검증됨). 빨강은 hue가 0 근처에서
# 순환하므로 두 구간으로 나눠서 잡는다. 물체가 바뀌면 이 값만 교체하면 됨 - 새 범위는
# cv2.cvtColor로 물체 사진 몇 장을 HSV로 변환해 실측하거나, 아래 물체가 화면에 보이는
# 상태에서 debug_depth_top_camera_preview.py에 간단한 픽셀 샘플링을 추가해 확인할 것.
DEFAULT_HSV_RANGES = [
    ((0, 100, 60), (10, 255, 255)),
    ((170, 100, 60), (180, 255, 255)),
]

MIN_OBJECT_PIXELS = 20  # 이보다 적게 검출되면 "못 찾음"으로 간주


_MORPH_KERNEL = np.ones((5, 5), np.uint8)


def detect_object_center(frame_bgr, hsv_ranges=DEFAULT_HSV_RANGES):
    """물체의 픽셀 중심(cx, cy)과 검출 성공 여부를 반환."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = None
    for low, high in hsv_ranges:
        m = cv2.inRange(hsv, low, high)
        mask = m if mask is None else (mask | m)
    # 노이즈로 생기는 낱개 픽셀(소금-후추 노이즈)이 중심 좌표를 프레임마다 흔드는 걸
    # 줄이기 위해 열림 연산(erode->dilate)으로 작은 잡음을 먼저 지운다.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
    ys, xs = np.where(mask > 0)
    if len(xs) < MIN_OBJECT_PIXELS:
        return None, None, False
    return int(xs.mean()), int(ys.mean()), True


class ObjectZoomTracker:
    """연속 프레임에 걸쳐 물체 중심을 추적하며 동적 크롭을 만들어내는 상태 유지 클래스.
    검출 실패 시 마지막으로 성공했던 위치를 그대로 유지(freeze)한다.

    hsv_ranges: [(low_hsv, high_hsv), ...] 형태. 물체가 바뀌면 이 값을 새로 넘기면 됨.

    smoothing_alpha: 검출된 중심 좌표에 적용하는 EMA 계수 (0~1). raw 검출값을 매 프레임
    그대로 쓰면 조명 노이즈/픽셀 몇 개 차이로 크롭 창이 흔들려서(화면이 계속 미세하게
    떨림) 학습에 불필요한 노이즈가 섞인다. 낮을수록 부드럽지만(=화면 안 흔들림) 실제
    물체가 빠르게 움직일 때(집어서 옮길 때 등) 따라가는 반응이 느려진다.

    dead_zone_px: 실제로 쓰는 크롭 창 위치를 기준으로, 새로 계산된 목표 위치와의 차이가
    이 값보다 작으면 아예 움직이지 않는다(완전히 고정). EMA만으로는 물체가 정지해
    있어도 이론상 0에 수렴할 뿐 완전히 0이 되진 않아 미세한 잔떨림이 남는데, 데드존을
    두면 그 잔떨림을 화면상에서 완전히 없앨 수 있다."""

    def __init__(
        self,
        crop_size=200,
        output_size=224,
        hsv_ranges=DEFAULT_HSV_RANGES,
        smoothing_alpha=0.12,
        dead_zone_px=4,
    ):
        self.crop_size = crop_size
        self.output_size = output_size
        self.hsv_ranges = hsv_ranges
        self.smoothing_alpha = smoothing_alpha
        self.dead_zone_px = dead_zone_px
        self._last_center = None  # 마지막으로 검출 성공한 raw (cx, cy)
        self._smoothed_center = None  # EMA로 부드럽게 만든 (cx, cy)
        self._crop_origin = None  # 실제로 쓰고 있는 크롭 좌상단 (x1, y1) - 데드존 기준점

    def update(self, frame_bgr):
        """frame_bgr(원본 해상도)을 받아 (확대된 224x224 BGR 이미지, 검출 성공 여부) 반환."""
        h, w = frame_bgr.shape[:2]
        cx, cy, detected = detect_object_center(frame_bgr, self.hsv_ranges)

        if detected:
            self._last_center = (cx, cy)
        elif self._last_center is not None:
            cx, cy = self._last_center
        else:
            # 한 번도 검출된 적 없음 (에피소드 시작 직후 등) -> 화면 중앙 기본값
            cx, cy = w // 2, h // 2

        if self._smoothed_center is None:
            self._smoothed_center = (cx, cy)
        else:
            a = self.smoothing_alpha
            px, py = self._smoothed_center
            self._smoothed_center = (a * cx + (1 - a) * px, a * cy + (1 - a) * py)
        scx, scy = self._smoothed_center

        half = self.crop_size // 2
        target_x1 = max(0, min(w - self.crop_size, int(round(scx)) - half))
        target_y1 = max(0, min(h - self.crop_size, int(round(scy)) - half))

        if self._crop_origin is None:
            self._crop_origin = (target_x1, target_y1)
        else:
            ox, oy = self._crop_origin
            if abs(target_x1 - ox) >= self.dead_zone_px or abs(target_y1 - oy) >= self.dead_zone_px:
                self._crop_origin = (target_x1, target_y1)
        x1, y1 = self._crop_origin
        crop = frame_bgr[y1 : y1 + self.crop_size, x1 : x1 + self.crop_size]
        zoomed = cv2.resize(crop, (self.output_size, self.output_size))
        return zoomed, detected

    def reset(self):
        """에피소드 시작 시 호출 - 이전 에피소드의 마지막 위치를 이어받지 않도록."""
        self._last_center = None
        self._smoothed_center = None
        self._crop_origin = None
