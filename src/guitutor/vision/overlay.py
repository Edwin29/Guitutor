from __future__ import annotations
import cv2, numpy as np
from typing import Iterable, Tuple

class Overlay:
    def __init__(self, draw: bool = True) -> None:
        self.draw = draw

    def grid(self, frame, grid_pts: Iterable[Tuple[int,int]]) -> None:
        if not self.draw: return
        for (x,y) in grid_pts:
            cv2.circle(frame, (int(x),int(y)), 2, (0,255,0), -1)

    def fingertips(self, frame, tips: dict[int, tuple[int,int]]) -> None:
        if not self.draw: return
        for f,(x,y) in tips.items():
            cv2.circle(frame, (int(x),int(y)), 6, (0,0,255), 2)
            cv2.putText(frame, str(f), (int(x)+6,int(y)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    def status(self, frame, text: str, color=(50,220,50)) -> None:
        if not self.draw: return
        cv2.putText(frame, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def _fill_poly_alpha(img, pts, color=(50,220,50), alpha=0.25, border_px=1):
    pts = np.asarray(pts, dtype=np.float32).astype(np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    if border_px > 0:
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=border_px)


def draw_fretboard_cells(frame, fretboard, cfg):
    """
    기대 인터페이스 (FretboardMapper):
      - H_inv: rect→img (3 * 3)
      - u_frets: [u0=0, ..., uN=1]  # 프렛 '경계' 리스트
      - v_strings: [v0=0, ..., vM=1] # 줄 '경계' 리스트
      - _dst_w, _dst_h: rect 픽셀 폭/높이
    """
    # 1) H_inv 확인
    Hinv = getattr(fretboard, "H_inv", None)
    if Hinv is None:
        return
    Hinv = Hinv.astype(np.float32)

    # 2) 경계(edge) 그대로 사용
    u_edges = np.asarray(getattr(fretboard, "u_frets", []), dtype=np.float32)
    v_edges = np.asarray(getattr(fretboard, "v_strings", []), dtype=np.float32)
    if len(u_edges) < 2 or len(v_edges) < 2:
        return

    # 3) rect 픽셀 크기
    W = int(getattr(fretboard, "_dst_w", 1000))
    H = int(getattr(fretboard, "_dst_h", 300))

    color = tuple(int(c) for c in cfg.get("cell_color", [50,220,50]))
    alpha = float(cfg.get("cell_alpha", 0.25))
    border_px = int(cfg.get("cell_border_px", 1))

    # 전체 프렛보드 외곽 디버그 라인
    # rect_all = np.array([[0,0],[W,0],[W,H],[0,H]], np.float32)[None,...]
    # cv2.polylines(frame, [cv2.perspectiveTransform(rect_all, Hinv)[0].astype(np.int32)],
    #               True, (0,255,255), 2)

    # 4) 각 셀을 rect(픽셀)에서 img로 투영해 채운다
    for j in range(len(u_edges)-1):
        x0 = u_edges[j]     * W
        x1 = u_edges[j + 1] * W
        for s in range(len(v_edges)-1):
            y0 = v_edges[s]     * H
            y1 = v_edges[s + 1] * H

            rect_uv_px = np.array([[x0, y0],
                                   [x1, y0],
                                   [x1, y1],
                                   [x0, y1]], dtype=np.float32).reshape(1, -1, 2)
            rect_xy = cv2.perspectiveTransform(rect_uv_px, Hinv).reshape(-1, 2)
            _fill_poly_alpha(frame, rect_xy, color=color, alpha=alpha, border_px=border_px)


def draw_indices(frame, fretboard):
    """프렛/줄 라벨 (경계 기준으로 중앙에 배치)"""
    Hinv = getattr(fretboard, "H_inv", None)
    if Hinv is None:
        return
    Hinv = Hinv.astype(np.float32)

    u_edges = np.asarray(getattr(fretboard, "u_frets", []), dtype=np.float32)
    v_edges = np.asarray(getattr(fretboard, "v_strings", []), dtype=np.float32)
    if len(u_edges) < 2 or len(v_edges) < 2:
        return

    W = int(getattr(fretboard, "_dst_w", 1000))
    H = int(getattr(fretboard, "_dst_h", 300))

    # 프렛 번호: 각 칸 중앙
    for j in range(1, len(u_edges)-1):
        u = 0.5 * (u_edges[j-1] + u_edges[j])
        v = 0.5 * (v_edges[0] + v_edges[1])  # 첫 줄 칸 중앙 쪽
        pt = np.array([[[u*W, v*H]]], dtype=np.float32)
        xy = cv2.perspectiveTransform(pt, Hinv)[0,0].astype(np.int32)
        cv2.putText(frame, f"F{j}", tuple(xy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

    # 줄 번호: 각 줄 첫 프렛 칸 중앙
    for s in range(1, len(v_edges)):
        u = 0.5 * (u_edges[0] + u_edges[1])
        v = 0.5 * (v_edges[s-1] + v_edges[s])
        pt = np.array([[[u*W, v*H]]], dtype=np.float32)
        xy = cv2.perspectiveTransform(pt, Hinv)[0,0].astype(np.int32)
        cv2.putText(frame, f"S{s}", tuple(xy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

def _cell_uv_rect(fretboard, s: int, f: int):
    u0, u1 = fretboard.u_frets[f-1], fretboard.u_frets[f]
    vS = fretboard.v_strings
    # v_edges 계산(줄-사이 경계)
    v_edges = [0]* (len(vS)+1)
    for i in range(1, len(vS)): v_edges[i] = 0.5*(vS[i-1] + vS[i])
    dv = (vS[1]-vS[0]) if len(vS)>=2 else 1.0
    v_edges[0] = vS[0] - 0.5*dv; v_edges[-1] = vS[-1] + 0.5*dv
    v0, v1 = v_edges[s-1], v_edges[s]
    rect_uv = np.array([[u0, v0],[u1, v0],[u1, v1],[u0, v1]], dtype=np.float32).reshape(-1,1,2)
    rect_xy = cv2.perspectiveTransform(rect_uv, fretboard.H_inv.astype(np.float32)).reshape(-1,2)
    return rect_xy

def highlight_cells(frame, fretboard, greens: list[tuple[int,int]], reds: list[tuple[int,int]]):
    for rect in greens:
        xy = _cell_uv_rect(fretboard, rect[0], rect[1])
        _fill_poly_alpha(frame, xy, color=(0,200,0), alpha=0.30, border_px=2)
    for rect in reds:
        xy = _cell_uv_rect(fretboard, rect[0], rect[1])
        _fill_poly_alpha(frame, xy, color=(0,0,220), alpha=0.30, border_px=2)

def fingertips_with_verdict(frame, tips, verdict) -> None:
    """
    손끝을 판정 결과에 맞춰 색으로 그려준다.
    verdict는 다음 중 하나를 갖고 있다고 가정:
      - verdict.per_finger: {finger_id: "hit"|"wrong"|"missing"|"idle"}
      - 혹은 verdict.details 와 같은 딕셔너리
    없으면 안전하게 전부 'idle'로 처리.
    """
    if tips is None:
        return

    # verdict에서 per-finger 상태 꺼내기(여러 이름을 허용)
    per = getattr(verdict, "per_finger", None) \
          or getattr(verdict, "details", None) \
          or {}

    colors = {
        "hit":    (0, 220, 0),    # green
        "wrong":  (0, 0, 255),    # red
        "missing":(0, 255, 255),  # yellow
        "idle":   (160, 160, 160) # gray
    }

    for f, (x, y) in tips.items():
        status = per.get(f, "idle")
        color = colors.get(status, (160, 160, 160))
        cv2.circle(frame, (int(x), int(y)), 8, color, 2)
        cv2.putText(frame, str(f), (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

