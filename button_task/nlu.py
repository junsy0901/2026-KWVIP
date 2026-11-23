"""한국어 명령을 누르기, 대기, 거절, 확인 질문 중 하나로 나눈다."""

from dataclasses import dataclass
import re


PRESS_VERBS = ("눌러", "선택", "찍어", "작동")
WAIT_MARKERS = ("누르지", "대기", "움직이지", "정지해", "정지", "멈추", "하지마")
FLOOR3_MARKERS = ("3층", "삼층", "3번버튼", "삼번버튼")
OTHER_FLOOR_MARKERS = (
    "1층",
    "2층",
    "4층",
    "5층",
    "6층",
    "7층",
    "8층",
    "9층",
    "일층",
    "이층",
    "사층",
    "오층",
    "육층",
    "칠층",
    "팔층",
    "구층",
)


@dataclass(frozen=True)
class Intent:
    action: str
    reason: str
    nlu_ms: float = 0.0


def normalize(text: str) -> str:
    compact = re.sub(r"\s+", "", text.strip().lower())
    compact = compact.replace("삼층", "3층").replace("칠층", "7층")
    return compact


def classify_command(text: str) -> Intent:
    """고정 문장 하나의 궤적 재생이 아니라, 표현을 같은 동작 또는 거절로 모은다."""
    compact = normalize(text)
    if any(marker in compact for marker in WAIT_MARKERS):
        return Intent("wait", "대기 또는 금지 명령이므로 팔을 움직이지 않음")

    other_floors = [marker for marker in OTHER_FLOOR_MARKERS if marker in compact]
    has_floor3 = any(marker in compact for marker in FLOOR3_MARKERS)
    if other_floors and not has_floor3:
        floors = ", ".join(other_floors)
        return Intent("reject", f"장면에 없는 버튼({floors})이라 수행하지 않음")
    if other_floors and has_floor3:
        return Intent("ask", "3층과 다른 층이 함께 있어 어떤 버튼인지 확인이 필요함")

    has_verb = any(verb in compact for verb in PRESS_VERBS)
    if has_floor3 and has_verb:
        return Intent("press_3", "3층 버튼을 누르는 명령")
    if has_floor3:
        return Intent("ask", "3층은 지정됐지만 누를지 대기할지 동사가 없음")
    if has_verb:
        return Intent("ask", "버튼을 누르라는 동사는 있으나 층수가 없음")
    return Intent("ask", "버튼 작업으로 해석되지 않음")
