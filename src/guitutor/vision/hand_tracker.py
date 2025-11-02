from __future__ import annotations
from typing import Dict, Tuple

class HandTracker:
    def __init__(self, kind: str = "mediapipe", min_conf: float = 0.6) -> None:
        self.kind = kind
        self.min_conf = min_conf
        self._mp = None
        if kind == "mediapipe":
            try:
                import mediapipe as mp  # type: ignore
                self._mp = mp
                self._hands = mp.solutions.hands.Hands(
                    max_num_hands=1, min_detection_confidence=min_conf, min_tracking_confidence=min_conf
                )
            except Exception as e:
                print('mediapipe를 사용할 수 없습니다. \'pip install "guitutor[vision]"\'을 확인하세요.', e)

                self.kind = "dummy"

    def get_fingertips(self, frame) -> Dict[int, Tuple[int,int]]:
        if self.kind != "mediapipe":
            return {}
        import cv2
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)
        tips = {}
        if res.multi_hand_landmarks:
            # 손가락 끝: index=8, middle=12, ring=16, pinky=20
            idxs = {1:8, 2:12, 3:16, 4:20}
            for hand in res.multi_hand_landmarks:
                for f, li in idxs.items():
                    lm = hand.landmark[li]
                    tips[f] = (int(lm.x * w), int(lm.y * h))
        return tips