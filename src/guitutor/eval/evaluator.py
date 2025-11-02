# src/guitutor/eval/evaluator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from .fingering import FingeringRules, FCell


@dataclass
class Verdict:
    summary: str
    ok : bool
    chord: str
    target: Dict[int, Optional[FCell]]    # 목표 배치
    hits_ok: List[FCell]                  # 맞춘 셀
    hits_bad: List[FCell]                 # 틀린 셀(손가락이 다른 셀 위)
    misses: List[FCell]                   # 놓쳐야 할 건 아니지만, 필요한 손가락이 없음
    per_finger: Dict[int, str]            # "hit" | "wrong" | "missing" | "idle"

    # --- 10/28 추가: 오버레이 색칠을 위한 per-finger 상태 맵 ---


class FingeringEvaluator:
    def __init__(self, rules: FingeringRules, hit_margin: float = 0.06):
        self.rules = rules
        self.current = next(iter(rules.chords.keys()))  # 첫 코드
        self.hit_margin = hit_margin

    # --- 추가: runner 호환을 위한 별칭 프로퍼티 (evaluator.chord) ---
    @property
    def chord(self) -> str:
        """runner가 기대하는 이름: 현재 코드명"""
        return self.current

    def set_chord(self, name: str) -> None:
        if name in self.rules.chords:
            self.current = name
        else:
            raise ValueError(f"Unknown chord: {name}")

    def next_chord(self, step: int = 1) -> str:
        keys = list(self.rules.chords.keys())
        i = (keys.index(self.current) + step) % len(keys)
        self.current = keys[i]
        return self.current

    # ------------------------------
    # per-finger 상태 계산 추가
    # ------------------------------
    def compare(self, hits: Dict[int, Optional[FCell]]) -> Verdict:
        """
        hits: {finger_id -> (string, fret) | None}
        """
        target = self.rules.chords[self.current]
        hits_ok: List[FCell] = []
        hits_bad: List[FCell] = []
        misses:  List[FCell] = []
        per: Dict[int, str] = {}

        # 손가락별 비교
        for finger, goal in target.items():
            got = hits.get(finger)

            if goal is None:
                # 이 손가락은 자유(free). 놓여 있든 아니든 페널티 X → 'idle'
                per[finger] = "idle"
                continue

            if got is None:
                # 필요 손가락이 안 올라갔음
                per[finger] = "missing"
                misses.append(goal)
            else:
                if self._cell_eq(got, goal):
                    per[finger] = "hit"
                    # got 이 FCell/dataclass/tuple 임을 모두 허용
                    hits_ok.append(self._to_tuple(got))
                else:
                    per[finger] = "wrong"
                    hits_bad.append(self._to_tuple(got))

        ok = (len(misses) == 0 and len(hits_bad) == 0)
        summary = f"{self.current}: " + ("GOOD" if ok else "WRONG FINGER")

        return Verdict(
            summary=summary,
            ok=ok,
            chord=self.current,
            target=target,
            hits_ok=hits_ok,
            hits_bad=hits_bad,
            misses=misses,
            per_finger=per,
        )

    # --- 내부 유틸: FCell/tuple 안전 비교 & 정규화 ---
    @staticmethod
    def _to_tuple(cell: FCell) -> Tuple[int, int]:
        """
        FCell 이 dataclass/NamedTuple/tuple 어떤 형태든 (s,f) 튜플로 변환
        """
        if cell is None:
            return None  # type: ignore
        if isinstance(cell, tuple):
            return (int(cell[0]), int(cell[1]))
        # dataclass 같은 객체 지원: .string/.fret 또는 .s/.f 추정
        s = getattr(cell, "string", getattr(cell, "s", None))
        f = getattr(cell, "fret",   getattr(cell, "f", None))
        if s is None or f is None:
            # 마지막 안전장치: 인덱서블?
            try:
                return (int(cell[0]), int(cell[1]))  # type: ignore[index]
            except Exception:
                raise TypeError(f"Unsupported FCell type: {type(cell)}")
        return (int(s), int(f))

    def _cell_eq(self, a: Optional[FCell], b: Optional[FCell]) -> bool:
        if a is None or b is None:
            return a is None and b is None
        ta = self._to_tuple(a)
        tb = self._to_tuple(b)
        return ta == tb
