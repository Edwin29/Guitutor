from __future__ import annotations
from pathlib import Path
import yaml, numpy as np
from typing import Dict, Tuple, Optional


class FretboardMapper:
    """
    - calib.yaml 포맷: {"H": [[...]*3]*3, "dst_w": 1000, "dst_h": 300}
    - 좌표계: 보정 평면의 u(가로), v(세로)를 0..1로 정규화
      * u 방향: nut→브리지 (프렛 증가 방향)
      * v 방향: 아래줄(1번줄) → 위줄(6번줄)
    """

    def __init__(self, strings: int = 6, frets: int = 5, calib_path: str = "config/calib.yaml") -> None:
        self.strings = int(strings)
        self.frets = int(frets)
        self.calib_path = calib_path

        self._H: Optional[np.ndarray] = None     # img -> rect homography
        self._dst_w: int = 1000
        self._dst_h: int = 300

        # overlay가 참조하는 경계(0..1, 길이=frets+1 / strings+1)
        self._u_edges: list[float] = []
        self._v_edges: list[float] = []

    # -------- overlay에서 쓰는 공개 프로퍼티 --------
    @property
    def H_inv(self) -> Optional[np.ndarray]:
        """rect -> img 변환이 필요할 때 사용 (overlay에서 사각형 투영용)"""
        if self._H is None:
            return None
        return np.linalg.inv(self._H).astype("float32")

    @property
    def u_frets(self) -> list[float]:
        """프렛 경계 u값(0..1), 길이=frets+1"""
        return self._u_edges

    @property
    def v_strings(self) -> list[float]:
        """줄 경계 v값(0..1), 길이=strings+1 (v↑가 줄 번호 증가 방향)"""
        return self._v_edges

    # ---------------- 로딩 ----------------
    def lazy_load_calibration(self) -> None:
        """calib 파일이 있으면 H/사이즈/격자 경계 로드"""
        p = Path(self.calib_path)
        if not p.exists():
            print("⚠️ calib file not found; cells will not render.")
            return

        raw = yaml.safe_load(open(p, "r", encoding="utf-8"))
        self._H = np.array(raw["H"], dtype=np.float32)
        self._dst_w = int(raw.get("dst_w", 1000))
        self._dst_h = int(raw.get("dst_h", 300))

        # 균등 분할(먼저 이렇게 쓰고, 필요시 12-TET 비율로 교체 가능)
        self._u_edges = [i / self.frets for i in range(self.frets + 1)]
        self._v_edges = [i / self.strings for i in range(self.strings + 1)]

    # --------------- 매핑 -----------------
    def map_to_cells(self, frame, fingertips: Dict[int, Tuple[int, int]]) -> Dict[int, Optional[Tuple[int, int]]]:
        """
        손끝 (x,y)[img] -> (string, fret) 1-base.
        보정이 없으면 모두 None.
        """
        if self._H is None:
            return {f: None for f in fingertips.keys()}

        out: Dict[int, Optional[Tuple[int, int]]] = {}
        for fid, (x, y) in fingertips.items():
            u, v = self._img_to_uv_norm(float(x), float(y))  # 0..1
            if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
                out[fid] = None
                continue

            fret_idx = self._locate_bin(u, self._u_edges)          # 0..frets-1
            string_bin = self._locate_bin(v, self._v_edges)        # 0..strings-1
            string_idx = string_bin + 1                            # 1..strings

            out[fid] = (string_idx, fret_idx + 1)                  # (줄, 프렛) 모두 1-base로 반환
        return out

    # -------------- 내부 도우미 --------------
    def _img_to_uv_norm(self, x: float, y: float) -> Tuple[float, float]:
        """img(x,y) -> rect(u,v) 정규화(0..1)"""
        pt = np.array([[x, y, 1.0]], dtype="float32").T  # (3,1)
        uv1 = self._H @ pt
        u_px = float(uv1[0, 0] / uv1[2, 0])
        v_px = float(uv1[1, 0] / uv1[2, 0])
        return u_px / self._dst_w, v_px / self._dst_h

    @staticmethod
    def _locate_bin(val: float, edges: list[float]) -> int:
        """
        edges: [e0=0, e1, ..., en=1] 에 대해
        val ∈ [ei, e(i+1)) 인 i를 반환. 경계=1.0이면 마지막 칸에 포함.
        """
        n = len(edges) - 1
        if val >= edges[-1]:
            return n - 1
        for i in range(n):
            if edges[i] <= val < edges[i + 1]:
                return i
        return 0
