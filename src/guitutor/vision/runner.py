# src/guitutor/vision/runner.py
from __future__ import annotations
import cv2, time, numpy as np
from collections import deque
from typing import Any, Optional
from dataclasses import dataclass

from .overlay import (
    draw_fretboard_cells,
    draw_indices,
    fingertips_with_verdict,   
)

@dataclass
class UIButton:
    id: str
    label: str
    w: int = 80
    h: int = 32
    x: Optional[int] = None   # 좌표는 선택 (자동 or 수동)
    y: Optional[int] = None

class ClickUI:
    def __init__(self, cfg: dict):
        self.enabled = bool(cfg.get("enabled", True))
        self.auto_anchor = cfg.get("auto_anchor")      # "top_right" | "top_left" | None
        self.base_y = int(cfg.get("base_y", 70))
        self.spacing = int(cfg.get("spacing", 6))
        self.pad_x = 10

        self.buttons: list[UIButton] = []
        for b in cfg.get("buttons", []):
            self.buttons.append(
                UIButton(
                    id=b["id"],
                    label=b.get("label", b["id"]),
                    w=int(b.get("w", 80)),
                    h=int(b.get("h", 32)),
                    x=(None if "x" not in b else int(b["x"])),
                    y=(None if "y" not in b else int(b["y"])),
                )
            )
        # 최근 draw에서 계산된 클릭 범위 캐시: [(id, (x1,y1,x2,y2)), ...]
        self._rects: list[tuple[str, tuple[int,int,int,int]]] = []

    def _layout(self, W: int):
        """버튼 배치 좌표를 계산해서 [(UIButton, bx, by)] 반환"""
        positions = []
        # 수동 좌표가 하나라도 있고 auto_anchor가 없으면 그대로 사용
        if self.auto_anchor is None and any(b.x is not None and b.y is not None for b in self.buttons):
            for b in self.buttons:
                bx = b.x if b.x is not None else self.pad_x
                by = b.y if b.y is not None else self.base_y
                positions.append((b, bx, by))
            return positions

        # 자동 배치 (상단 좌/우)
        anchor = self.auto_anchor or "top_left"
        x = W - self.pad_x if anchor == "top_right" else self.pad_x

        # top_right일 땐 reversed
        seq = reversed(self.buttons) if anchor == "top_right" else self.buttons

        for b in seq:
            if anchor == "top_right":
                bx = x - b.w        # 오른쪽 끝에서 왼쪽으로 
                x = bx - self.spacing
            else:
                bx = x              # 왼쪽 끝에서 오른쪽으로
                x = bx + b.w + self.spacing
            positions.append((b, bx, self.base_y))
        return positions

    def draw(self, frame):
        if not self.enabled or not self.buttons:
            return
        H, W = frame.shape[:2]
        self._rects.clear()

        for b, bx, by in self._layout(W):
            x1, y1, x2, y2 = bx, by, bx + b.w, by + b.h
            self._rects.append((b.id, (x1, y1, x2, y2)))
            # 배경/테두리/라벨
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
            cv2.putText(frame, b.label, (x1 + 8, y1 + int(b.h*0.7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)

    def hit(self, x, y) -> Optional[str]:
        if not self.enabled:
            return None
        # draw에서 계산한 사각형 기준으로 판정
        for bid, (x1, y1, x2, y2) in self._rects:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return bid
        return None


class VideoRunner:
    """
    - tracker: 손끝 랜드마크 트래커 (get_fingertips(frame) -> {finger_id: (x,y)})
    - fretboard: 프렛보드 매퍼 (lazy_load_calibration(), map_to_cells(frame, tips))
    - evaluator: 운지 평가기 (compare(smoothed) -> Verdict)
    - overlay: 프레임에 점/상태를 그리는 도우미 (status())
    - cfg: config dict (overlay/hotkeys/video/app 등)
    """
    def __init__(self, tracker, fretboard, evaluator, overlay, *, cfg=None, smoothing_window: int = 7) -> None:
        self.tracker = tracker
        self.fretboard = fretboard
        self.evaluator = evaluator
        self.overlay = overlay
        self.cfg = (cfg or {})
        self.smooth = deque(maxlen=max(3, smoothing_window))
        self.ui = ClickUI(self.cfg.get("ui", {}))


        # --- 코드 리스트 & 현재 선택 인덱스 ---
        chord_names = list(self.evaluator.rules.chords.keys())
        cur = self.evaluator.chord
        self._chords = chord_names
        self._idx = chord_names.index(cur) if cur in chord_names else 0
        # 안전하게 evaluator에도 반영
        self.evaluator.set_chord(self._chords[self._idx])

        # 핫키 설정 (default.yaml에서 오버라이드 가능)
        hk = self.cfg.get("hotkeys", {})
        self._HK_TOGGLE = hk.get("toggle_cells", "C")
        self._HK_NEXT   = hk.get("next_chord",  "N")
        self._HK_PREV   = hk.get("prev_chord",  "P")

    # ---- 코드 전환 도우미 ----
    def _next_chord(self) -> None:
        self._idx = (self._idx + 1) % len(self._chords)
        self.evaluator.set_chord(self._chords[self._idx])

    def _prev_chord(self) -> None:
        self._idx = (self._idx - 1) % len(self._chords)
        self.evaluator.set_chord(self._chords[self._idx])

    def _draw_chord_badge(self, frame, name: str, ok: bool) -> None:
        """현재 코드 이름을 화면 좌상단에 뱃지로 표시"""
        label = f"[{name}]"
        color = (80,220,80) if ok else (60,60,220)
        # 반투명 박스
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x0, y0, pad = 10, 40, 6
        box = (x0, y0 - th - pad, x0 + tw + 2*pad, y0 + pad)
        overlay = frame.copy()
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color, -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.putText(frame, label, (x0 + pad, y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    def run(self, source: Any) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError("Cannot open source")
        
        win = "guitutor"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        def _on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                action = self.ui.hit(x, y)
                if action:
                    self._handle_action(action)

        cv2.setMouseCallback(win, _on_mouse)

        
        # ---- 요청 해상도/FPS 적용 (장치가 거부할 수도) ----
        vcfg = self.cfg.get("video", {})
        req_w, req_h, req_fps = vcfg.get("req_width"), vcfg.get("req_height"), vcfg.get("req_fps")
        if req_w:  cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(req_w))
        if req_h:  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(req_h))
        if req_fps: cap.set(cv2.CAP_PROP_FPS, int(req_fps))

        # 실제 적용된 값 읽기
        act_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        act_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        act_fps= cap.get(cv2.CAP_PROP_FPS) or 0.0
        print(f"[Capture] actual: {act_w}x{act_h} @ ~{act_fps:.1f}fps (device reported)")

        # FPS 계산용
        import time
        last = time.time()
        fps_ema = None

        # 보드 캘리브레이션 로드
        if hasattr(self.fretboard, "lazy_load_calibration"):
            self.fretboard.lazy_load_calibration()

        ov = self.cfg.get("overlay", {})

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # 1) 손끝 추적
            tips = self.tracker.get_fingertips(frame)  # {1:(x,y), 2:(x,y), ...}

            # 2) 지판 셀 매핑 → 평활(다수결)
            hits = self.fretboard.map_to_cells(frame, tips)   # {finger: (string,fret)|None}
            self.smooth.append(hits)
            smoothed = self._majority_vote(self.smooth)

            # 3) 평가
            verdict = self.evaluator.compare(smoothed)

            # 4) 손끝을 판정 색으로 그리기 + 상태문구 + 현재 코드 뱃지
            fingertips_with_verdict(frame, tips, verdict)
            if hasattr(self.overlay, "status"):
                self.overlay.status(frame, verdict.summary,
                                    color=(50,220,50) if verdict.ok else (60,60,220))
            self._draw_chord_badge(frame, verdict.chord, verdict.ok)

            # 5) 반투명 셀/인덱스 오버레이
            if ov.get("show_cells", False):
                draw_fretboard_cells(
                    frame,
                    fretboard=self.fretboard,
                    cfg={
                        "cell_alpha": ov.get("cell_alpha", 0.25),
                        "cell_color": ov.get("cell_color", [50, 220, 50]),  # BGR
                        "cell_border_px": ov.get("cell_border_px", 1),
                    },
                )
            if ov.get("show_indices", False):
                draw_indices(frame, self.fretboard)

            self.ui.draw(frame)
            # 6) 화면 표시 + 키 처리
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF

            # 셀 토글
            if key in (ord(self._HK_TOGGLE), ord(self._HK_TOGGLE.lower())):
                ov["show_cells"] = not ov.get("show_cells", False)

            # 코드 전환 (N/P)
            if key in (ord(self._HK_NEXT), ord(self._HK_NEXT.lower())):
                self._next_chord()
            if key in (ord(self._HK_PREV), ord(self._HK_PREV.lower())):
                self._prev_chord()

            # ---- FPS 표시 ----
            now = time.time()
            inst = 1.0 / max(1e-3, (now - last))
            last = now
            fps_ema = inst if fps_ema is None else (0.9*fps_ema + 0.1*inst)
            cv2.putText(frame, f"{act_w}x{act_h}  FPS:{fps_ema:4.1f}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 50), 2)
            

            # ESC 종료
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def _majority_vote(buf):
        """최근 N프레임의 매핑 결과에 대해 손가락별 다수결"""
        agg = {}
        for k in (1, 2, 3, 4):
            vals = [d.get(k) for d in buf if d and d.get(k) is not None]
            if vals:
                agg[k] = max(set(vals), key=vals.count)
        return agg
    
    def _handle_action(self, action_id: str):
        # UI 버튼 ID ↔ 동작 매핑
        if action_id == "toggle_cells":
            self._toggle_cells()
        elif action_id == "toggle_grid":
            ov = self.cfg.setdefault("overlay", {})
            ov["show_grid"] = not ov.get("show_grid", False)
            print(f"[UI] grid → {ov['show_grid']}")
        elif action_id == "next_chord":
            self._next_chord()
            print(f"[UI] next chord → {self.evaluator.chord}")
        elif action_id == "prev_chord":
            self._prev_chord()
            print(f"[UI] prev chord → {self.evaluator.chord}")

    def _toggle_cells(self):
        ov = self.cfg.setdefault("overlay", {})
        ov["show_cells"] = not ov.get("show_cells", False)
        print(f"[UI] cells → {ov['show_cells']}")

