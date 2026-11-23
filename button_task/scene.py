"""휠체어–PiPER 장면을 불러오고 시험 조건을 적용한다."""

from pathlib import Path

import mujoco
import numpy as np

XML_PATH = Path(__file__).resolve().parents[1] / "wheelchair_piper_lite.xml"

# 배포 모델은 팔 시각 지오메트리의 충돌이 꺼져 있다.
# 끝점 사이트와 같은 자리에 반지름 12 mm 구를 넣어 버튼 접촉이 물리로 일어나게 한다.
PUSHER_GEOM = (
    '<geom name="ee_pusher" type="sphere" pos="0 0 .15" size=".012" '
    'rgba=".2 .85 .45 1" contype="1" conaffinity="1" mass=".02" '
    'friction="1.2 .005 .0001"/>'
)
DEMO_CAMERA = (
    '<camera name="eval_cam" mode="fixed" pos="1.05 -0.25 1.25" '
    'xyaxes="-0.610 0.793 0 -0.231 -0.178 0.957" fovy="48"/>'
)

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
# 버튼 사이트는 전면보다 +Y로 25 mm 앞에 있다. 가압 목표는 그 점에서 -Y로 35 mm.
APPROACH_OFFSET = np.array([0.0, 0.05, 0.0])
PRESS_OFFSET_FROM_SITE = np.array([0.0, -0.035, 0.0])
# 카메라 광선은 버튼 표면에 맞는다. 표면 기준으로 앞 60 mm, 안쪽 20 mm.
APPROACH_OFFSET_FROM_HIT = np.array([0.0, 0.06, 0.0])
PRESS_OFFSET_FROM_HIT = np.array([0.0, -0.020, 0.0])
MODEL_CHANGE_NOTE = (
    "wheelchair_piper_lite.xml의 팔 링크는 충돌이 꺼져 있어 끝점이 버튼 좌표에 "
    "도착해도 버튼이 눌리지 않는다. 불러올 때 ee_site와 같은 위치(link6, z=0.15 m)에 "
    "반지름 12 mm 충돌 구 ee_pusher를 추가했다. 시각 확인용 eval_cam도 추가했다. "
    "버튼 성공은 이 구의 위치가 아니라 button_3_slide 변위로만 판정한다."
)


def load_model(xml_path: Path | None = None) -> mujoco.MjModel:
    path = Path(xml_path) if xml_path else XML_PATH
    text = path.read_text(encoding="utf-8")
    if 'name="ee_pusher"' not in text:
        text = text.replace(
            '<site name="ee_site"',
            PUSHER_GEOM + '\n                      <site name="ee_site"',
        )
    if 'name="eval_cam"' not in text:
        text = text.replace(
            '<geom name="floor"',
            DEMO_CAMERA + "\n    <geom name=\"floor\"",
        )
    return mujoco.MjModel.from_xml_string(text)


def clip_arm(model: mujoco.MjModel, q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=float).copy()
    for index, name in enumerate(ARM_JOINTS):
        low, high = model.joint(name).range
        out[index] = float(np.clip(out[index], low + 1e-4, high - 1e-4))
    return out


def arm_qpos(data: mujoco.MjData) -> np.ndarray:
    return np.array([data.joint(name).qpos[0] for name in ARM_JOINTS], dtype=float)


def button_displacement(data: mujoco.MjData) -> float:
    return float(data.joint("button_3_slide").qpos[0])


def set_panel_offset(model: mujoco.MjModel, offset: np.ndarray) -> None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "elevator_panel")
    base = np.array([0.43, -0.87, 0.99], dtype=float)
    model.body_pos[body_id] = base + np.asarray(offset, dtype=float)


def command_arm(data: mujoco.MjData, q: np.ndarray) -> None:
    for name, value in zip(ARM_JOINTS, q):
        data.actuator(name).ctrl[0] = float(value)
    data.actuator("gripper").ctrl[0] = 0.0


def reset_trial(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    panel_offset: np.ndarray,
    arm_noise: np.ndarray,
) -> np.ndarray:
    set_panel_offset(model, panel_offset)
    mujoco.mj_resetData(model, data)
    q0 = clip_arm(model, np.asarray(arm_noise, dtype=float))
    for name, value in zip(ARM_JOINTS, q0):
        data.joint(name).qpos[0] = value
    command_arm(data, q0)
    mujoco.mj_forward(model, data)
    settle_steps = int(0.4 / model.opt.timestep)
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
    return arm_qpos(data)
