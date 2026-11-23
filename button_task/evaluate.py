"""기본 10회, 변화 10회, 명령 5회를 상태 기반과 카메라 기반으로 측정한다."""

from pathlib import Path
import json

import numpy as np

from button_task.nlu import classify_command
from button_task.task import ButtonTask


PRESS_TEXT = "3층 버튼을 눌러줘"
COMMAND_TRIALS = [
    ("3층 버튼을 눌러줘", "press_3"),
    ("엘리베이터 3층을 선택해 줘", "press_3"),
    ("버튼은 누르지 말고 대기해", "wait"),
    ("팔은 움직이지 말고 정지해", "wait"),
    ("현재 장면에 없는 7층 버튼을 눌러줘", "reject"),
]


def basic_condition(index: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1000 + index)
    offset = np.zeros(3)
    noise = rng.uniform(-0.12, 0.12, size=6)
    return offset, noise


def variation_condition(index: int) -> tuple[np.ndarray, np.ndarray]:
    """기본 조건의 작은 초기 자세 범위를 벗어난 패널 위치."""
    rng = np.random.default_rng(2000 + index)
    while True:
        offset = np.array(
            [
                rng.uniform(-0.06, 0.06),
                rng.uniform(-0.03, 0.03),
                rng.uniform(-0.05, 0.05),
            ]
        )
        if abs(offset[0]) >= 0.02 or abs(offset[2]) >= 0.02:
            break
    noise = rng.uniform(-0.12, 0.12, size=6)
    return offset, noise


def _rate(rows: list[dict], key: str = "success") -> float:
    if not rows:
        return 0.0
    return 100.0 * sum(1 for row in rows if row[key]) / len(rows)


def _median(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row.get("success")]
    if not values:
        return None
    return float(np.median(values))


def _failure_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row["success"]:
            continue
        if row.get("appropriateness") and row["intent"] != "press_3":
            continue
        label = (row["failure"] or "unknown").split(":", 1)[0]
        counts[label] = counts.get(label, 0) + 1
    return counts


def evaluate(task: ButtonTask | None = None) -> dict:
    task = task or ButtonTask()
    physical = []
    for split, maker in (("basic", basic_condition), ("variation", variation_condition)):
        for index in range(10):
            offset, noise = maker(index)
            for mode in ("state", "vision"):
                result, _ = task.run(
                    PRESS_TEXT,
                    mode=mode,
                    panel_offset=offset,
                    arm_noise=noise,
                    seed=index,
                )
                row = result.to_dict()
                row["split"] = split
                physical.append(row)
                print(
                    f"{split:9} {mode:6} #{index:02d} "
                    f"success={row['success']} button={row['button_m']*1000:.1f}mm "
                    f"{row['failure'] or row['perception']}"
                )

    commands = []
    for text, expected in COMMAND_TRIALS:
        result, _ = task.run(text, mode="state", seed=0)
        row = result.to_dict()
        row["expected_intent"] = expected
        row["intent_correct"] = row["intent"] == expected
        if expected == "press_3":
            row["appropriateness"] = row["intent_correct"] and row["success"]
        else:
            row["appropriateness"] = row["intent_correct"] and row["appropriateness"]
        commands.append(row)
        print(
            f"command {row['intent']:8} expected={expected:8} "
            f"ok={row['appropriateness']} button={row['button_m']*1000:.1f}mm"
        )

    def subset(split: str, mode: str) -> list[dict]:
        return [row for row in physical if row["split"] == split and row["mode"] == mode]

    state_basic = subset("basic", "state")
    state_var = subset("variation", "state")
    vision_basic = subset("basic", "vision")
    vision_var = subset("variation", "vision")
    summary = {
        "success_threshold_m": 0.010,
        "model_change": task.model_change_note,
        "seeds": {
            "basic": "numpy Generator 1000+index, 초기 관절 ±0.12 rad, 패널 고정",
            "variation": "numpy Generator 2000+index, 패널 x,z가 2 cm 이상 이동",
        },
        "state_basic_sr": _rate(state_basic),
        "state_variation_sr": _rate(state_var),
        "state_gap_pp": _rate(state_basic) - _rate(state_var),
        "vision_basic_sr": _rate(vision_basic),
        "vision_variation_sr": _rate(vision_var),
        "vision_gap_pp": _rate(vision_basic) - _rate(vision_var),
        "command_appropriateness": _rate(commands, "appropriateness"),
        "state_basic_time_s_median": _median(state_basic, "duration_s"),
        "state_variation_time_s_median": _median(state_var, "duration_s"),
        "vision_basic_time_s_median": _median(vision_basic, "duration_s"),
        "plan_ms_state_median": float(np.median([row["plan_ms"] for row in state_basic + state_var])),
        "plan_ms_vision_median": float(np.median([row["plan_ms"] for row in vision_basic + vision_var])),
        "failure_counts_state": _failure_counts(state_basic + state_var),
        "failure_counts_vision": _failure_counts(vision_basic + vision_var),
        "physical": physical,
        "commands": commands,
    }
    return summary


def check_language() -> None:
    expected = {
        "3층 버튼을 눌러줘": "press_3",
        "엘리베이터 3층을 선택해 줘": "press_3",
        "삼층 버튼 좀 눌러줄래": "press_3",
        "버튼은 누르지 말고 대기해": "wait",
        "팔은 움직이지 말고 정지해": "wait",
        "현재 장면에 없는 7층 버튼을 눌러줘": "reject",
        "4층을 선택해 줘": "reject",
        "버튼 눌러줘": "ask",
        "오늘 날씨 어때": "ask",
    }
    for text, action in expected.items():
        got = classify_command(text).action
        if got != action:
            raise AssertionError(f"{text!r} -> {got}, expected {action}")


def save_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    check_language()
    report = evaluate()
    out = Path("results") / "button_eval.json"
    save_summary(report, out)
    print(json.dumps({k: v for k, v in report.items() if k not in {"physical", "commands"}}, ensure_ascii=False, indent=2))
    print(f"saved {out}")
