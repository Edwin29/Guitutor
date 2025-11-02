# src/guitutor/eval/fingering.py
from __future__ import annotations
import yaml
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List

FCell = Tuple[int,int]  # (string, fret)

@dataclass
class FingeringRules:
    chords: Dict[str, Dict[int, Optional[FCell]]]   # name -> finger -> (s,f) | None
    mute: Dict[str, List[int]]                      # name -> muted string numbers (옵션)

    @classmethod
    def from_yaml(cls, path: str) -> "FingeringRules":
        raw = yaml.safe_load(open(path, "r", encoding="utf-8"))

        # -----------------------------
        # 포맷 A (최신 포맷)
        # -----------------------------
        if "items" in raw:
            chords: Dict[str, Dict[int, Optional[FCell]]] = {}
            mute: Dict[str, List[int]] = {}
            for it in raw["items"]:
                name = it["name"]
                fm: Dict[int, Optional[FCell]] = {}
                for rec in it.get("expected", []):
                    f = int(rec["finger"])
                    s = int(rec["string"])
                    r = int(rec["fret"])
                    fm[f] = (s, r)
                # 사용 안 하는 손가락은 명시되어 있지 않을 수 있으므로 None으로 채우기
                for f in (1,2,3,4):
                    fm.setdefault(f, None)
                chords[name] = fm
                mute[name] = [int(x) for x in it.get("mute", [])]
            return cls(chords=chords, mute=mute)

        # -----------------------------
        # 포맷 B (이전 포맷)
        # -----------------------------
        if "chords" in raw:
            chords = {}
            mute = {}
            for name, mp in raw["chords"].items():
                fm: Dict[int, Optional[FCell]] = {}
                for fk, cell in mp.items():
                    i = int(fk)
                    fm[i] = tuple(cell) if cell is not None else None
                for f in (1,2,3,4):
                    fm.setdefault(f, None)
                chords[name] = fm
                mute[name] = []
            return cls(chords=chords, mute=mute)

        raise ValueError("fingerings.yaml 포맷을 인식할 수 없습니다.")
