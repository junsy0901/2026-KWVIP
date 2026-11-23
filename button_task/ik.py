"""끝점 위치 오차를 줄이는 감쇠 최소제곱 역기구학."""

import mujoco
import numpy as np

from button_task.scene import ARM_JOINTS, clip_arm


class SiteIK:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.kin = mujoco.MjData(model)
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    def solve(
        self,
        q_seed: np.ndarray,
        target: np.ndarray,
        iterations: int = 80,
        damping: float = 0.08,
        max_step: float = 0.25,
        tolerance: float = 1e-3,
    ) -> tuple[np.ndarray, float]:
        q = clip_arm(self.model, q_seed)
        target = np.asarray(target, dtype=float)
        for _ in range(iterations):
            self._write(q)
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.kin, jacp, jacr, self.site_id)
            jacobian = jacp[:, :6]
            error = target - self.kin.site("ee_site").xpos
            error_norm = float(np.linalg.norm(error))
            if error_norm < tolerance:
                return q, error_norm
            system = jacobian @ jacobian.T + (damping ** 2) * np.eye(3)
            delta = jacobian.T @ np.linalg.solve(system, error)
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > max_step:
                delta *= max_step / delta_norm
            q = clip_arm(self.model, q + delta)
        self._write(q)
        error_norm = float(np.linalg.norm(target - self.kin.site("ee_site").xpos))
        return q, error_norm

    def _write(self, q: np.ndarray) -> None:
        for name, value in zip(ARM_JOINTS, q):
            self.kin.joint(name).qpos[0] = float(value)
        mujoco.mj_forward(self.model, self.kin)
