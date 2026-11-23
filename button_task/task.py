"""명령 하나를 물리 시뮬레이션에서 수행한다."""

from dataclasses import dataclass, asdict
import time

import mujoco
import numpy as np

from button_task import SUCCESS_DISPLACEMENT_M
from button_task.ik import SiteIK
from button_task.nlu import Intent, classify_command
from button_task.perceive import IMAGE_HEIGHT, IMAGE_WIDTH, perceive_button
from button_task.scene import (
    APPROACH_OFFSET,
    APPROACH_OFFSET_FROM_HIT,
    ARM_JOINTS,
    MODEL_CHANGE_NOTE,
    PRESS_OFFSET_FROM_HIT,
    PRESS_OFFSET_FROM_SITE,
    arm_qpos,
    button_displacement,
    command_arm,
    load_model,
    reset_trial,
)


@dataclass
class TrialResult:
    text: str
    mode: str
    seed: int
    panel_offset: list[float]
    intent: str
    reason: str
    success: bool
    appropriateness: bool
    failure: str
    button_m: float
    duration_s: float
    plan_ms: float
    nlu_ms: float
    max_joint_delta_rad: float
    ik_error_m: float
    perception: str

    def to_dict(self) -> dict:
        return asdict(self)


class ButtonTask:
    def __init__(self, xml_path=None, nlu: str = "rules"):
        self.model = load_model(xml_path)
        self.data = mujoco.MjData(self.model)
        self.ik = SiteIK(self.model)
        self.renderer = mujoco.Renderer(self.model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
        self.model_change_note = MODEL_CHANGE_NOTE
        if nlu == "rules":
            self._classify = classify_command
            self.nlu_name = "rules"
        elif nlu == "llm":
            from button_task.nlu_llm import ExaoneClassifier

            self._classify = ExaoneClassifier()
            self.nlu_name = "exaone-4.0-1.2b"
        else:
            raise ValueError("nlu는 rules 또는 llm")

    def run(
        self,
        text: str,
        mode: str = "vision",
        panel_offset: np.ndarray | None = None,
        arm_noise: np.ndarray | None = None,
        seed: int = 0,
        record: bool = False,
    ) -> tuple[TrialResult, list[np.ndarray]]:
        if mode not in {"state", "vision"}:
            raise ValueError("mode는 state 또는 vision")
        offset = np.zeros(3) if panel_offset is None else np.asarray(panel_offset, dtype=float)
        noise = np.zeros(6) if arm_noise is None else np.asarray(arm_noise, dtype=float)
        q_start = reset_trial(self.model, self.data, offset, noise)
        frames: list[np.ndarray] = []
        if record:
            frames.append(self._grab())

        started = time.perf_counter()
        sim_started = float(self.data.time)
        intent = self._classify(text)
        if intent.action != "press_3":
            self._hold(seconds=0.5, frames=frames if record else None)
            delta = float(np.max(np.abs(arm_qpos(self.data) - q_start)))
            pressed = button_displacement(self.data) >= SUCCESS_DISPLACEMENT_M
            moved = delta > 0.03 or pressed
            appropriate = intent.action in {"wait", "reject", "ask"} and not moved
            return (
                TrialResult(
                    text=text,
                    mode=mode,
                    seed=seed,
                    panel_offset=offset.tolist(),
                    intent=intent.action,
                    reason=intent.reason,
                    success=False,
                    appropriateness=appropriate,
                    failure="" if appropriate else "command: 금지 명령인데 팔 또는 버튼이 움직임",
                    button_m=button_displacement(self.data),
                    duration_s=float(self.data.time - sim_started),
                    plan_ms=(time.perf_counter() - started) * 1000.0,
                    nlu_ms=intent.nlu_ms,
                    max_joint_delta_rad=delta,
                    ik_error_m=0.0,
                    perception="",
                ),
                frames,
            )

        plan_started = time.perf_counter()
        target, perception, press_offset, approach_offset = self._target(mode)
        if target is None:
            return (
                self._failed_press(
                    text, mode, seed, offset, intent, perception,
                    q_start, sim_started, plan_started, 0.0,
                    frames if record else None,
                    intent.nlu_ms,
                ),
                frames,
            )

        q_now = arm_qpos(self.data)
        q_approach, approach_error = self.ik.solve(q_now, target + approach_offset)
        q_press, press_error = self.ik.solve(q_approach, target + press_offset)
        plan_ms = (time.perf_counter() - plan_started) * 1000.0
        if press_error > 0.008:
            result = self._failed_press(
                text, mode, seed, offset, intent,
                f"ik: 가압 자세 오차 {press_error * 1000:.1f} mm",
                q_start, sim_started, plan_started, press_error,
                frames if record else None,
                intent.nlu_ms,
            )
            result.perception = perception
            result.plan_ms = plan_ms
            return result, frames

        self._drive(q_approach, seconds=2.0, frames=frames if record else None, until_near=target + approach_offset)
        self._drive(
            q_press,
            seconds=2.2,
            frames=frames if record else None,
            until_pressed=True,
        )
        if button_displacement(self.data) >= SUCCESS_DISPLACEMENT_M:
            self._hold(seconds=0.45, frames=frames if record else None)
        peak = button_displacement(self.data)
        success = peak >= SUCCESS_DISPLACEMENT_M
        if success:
            self._drive(q_approach, seconds=0.8, frames=frames if record else None)
            failure = ""
        else:
            ee = self.data.site("ee_site").xpos
            distance = float(np.linalg.norm(ee - (target + press_offset)))
            if distance < 0.04:
                failure = "contact: 버튼 근처까지 갔으나 10 mm 미만으로 눌림"
            else:
                failure = "ik: 제한 시간 안에 가압 자세에 도달하지 못함"
        delta = float(np.max(np.abs(arm_qpos(self.data) - q_start)))
        result = TrialResult(
            text=text,
            mode=mode,
            seed=seed,
            panel_offset=[float(v) for v in offset],
            intent=intent.action,
            reason=intent.reason,
            success=success,
            appropriateness=success,
            failure=failure,
            button_m=peak,
            duration_s=float(self.data.time - sim_started),
            plan_ms=plan_ms,
            nlu_ms=intent.nlu_ms,
            max_joint_delta_rad=delta,
            ik_error_m=float(max(approach_error, press_error)),
            perception=perception,
        )
        return result, frames

    def _target(self, mode: str):
        site = self.data.site("button_target").xpos.copy()
        if mode == "state":
            return site, "state: button_target 사이트 좌표", PRESS_OFFSET_FROM_SITE, APPROACH_OFFSET
        hit, message = perceive_button(self.model, self.data, self.renderer)
        if hit is None:
            return None, message, PRESS_OFFSET_FROM_HIT, APPROACH_OFFSET_FROM_HIT
        return hit, message, PRESS_OFFSET_FROM_HIT, APPROACH_OFFSET_FROM_HIT

    def _failed_press(
        self, text, mode, seed, offset, intent: Intent, failure, q_start,
        sim_started, plan_started, ik_error, frames, nlu_ms,
    ) -> TrialResult:
        self._hold(seconds=0.3, frames=frames if frames is not None else None)
        return TrialResult(
            text=text,
            mode=mode,
            seed=seed,
            panel_offset=[float(v) for v in offset],
            intent=intent.action,
            reason=intent.reason,
            success=False,
            appropriateness=False,
            failure=failure,
            button_m=button_displacement(self.data),
            duration_s=float(self.data.time - sim_started),
            plan_ms=(time.perf_counter() - plan_started) * 1000.0,
            nlu_ms=nlu_ms,
            max_joint_delta_rad=float(np.max(np.abs(arm_qpos(self.data) - q_start))),
            ik_error_m=float(ik_error),
            perception=failure,
        )

    def _drive(self, q_goal, seconds, frames, until_near=None, until_pressed=False) -> None:
        command_arm(self.data, q_goal)
        steps = int(seconds / self.model.opt.timestep)
        sample = max(1, int(round(1.0 / (30 * self.model.opt.timestep))))
        for step in range(steps):
            mujoco.mj_step(self.model, self.data)
            if frames is not None and step % sample == 0:
                frames.append(self._grab())
            if until_pressed and button_displacement(self.data) >= SUCCESS_DISPLACEMENT_M:
                break
            if until_near is not None:
                error = np.linalg.norm(self.data.site("ee_site").xpos - until_near)
                if error < 0.015 and step > 50:
                    break

    def _hold(self, seconds: float, frames) -> None:
        steps = int(seconds / self.model.opt.timestep)
        sample = max(1, int(round(1.0 / (30 * self.model.opt.timestep))))
        for step in range(steps):
            mujoco.mj_step(self.model, self.data)
            if frames is not None and step % sample == 0:
                frames.append(self._grab())

    def _grab(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="eval_cam")
        return self.renderer.render().copy()
