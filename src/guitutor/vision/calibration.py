from __future__ import annotations
import cv2, yaml
import numpy as np
from pathlib import Path

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    임의 순서의 4점 -> [좌상, 우상, 우하, 좌하]로 정렬
    - s = x+y:  최소=좌상, 최대=우하
    - d = x-y:  최소=우상, 최대=좌하
    """
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)

class Calibrator:
    def interactive_calibrate(self, video_path: str, save_path: str) -> None:
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("영상/이미지 로드 실패")

        points: list[tuple[int,int]] = []

        def click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))

        win = "click 4 corners (any order: board rect)"
        cv2.imshow(win, frame)
        cv2.setMouseCallback(win, click)

        while True:
            disp = frame.copy()
            for p in points:
                cv2.circle(disp, p, 5, (0, 255, 255), -1)
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if len(points) == 4:
                break

        cv2.destroyAllWindows()
        if len(points) != 4:
            raise RuntimeError("4점을 지정해야 합니다.")

        # 클릭 순서 자동 보정
        src = _order_corners(np.array(points, dtype=np.float32))

        # 정사영(목표 보드 크기)
        dst_w, dst_h = 1000, 300
        dst = np.array([[0, 0],
                        [dst_w, 0],
                        [dst_w, dst_h],
                        [0, dst_h]], dtype=np.float32)

        H = cv2.getPerspectiveTransform(src, dst)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"H": H.tolist(), "dst_w": dst_w, "dst_h": dst_h}, f)

        # 미리보기
        warped = cv2.warpPerspective(frame, H, (dst_w, dst_h))
        cv2.imshow("warped preview", warped)
        cv2.waitKey(800)
        cv2.destroyAllWindows()
