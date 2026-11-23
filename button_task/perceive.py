"""머리 카메라 영상에서 파란 버튼을 찾고 광선으로 3차원 점을 만든다."""

import mujoco
import numpy as np

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
MIN_BLOB_PIXELS = 20


def _largest_blob(mask: np.ndarray) -> list[tuple[int, int]]:
    points = list(zip(*np.nonzero(mask)))
    if not points:
        return []
    remaining = set(points)
    best: list[tuple[int, int]] = []
    neighbors = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            y, x = stack.pop()
            for dy, dx in neighbors:
                nxt = (y + dy, x + dx)
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
                    component.append(nxt)
        if len(component) > len(best):
            best = component
    return best


def blue_button_centroid(image: np.ndarray) -> tuple[float, float] | None:
    """조명으로 어두워진 버튼 파랑(대략 R 30, G 70, B 110)을 고른다."""
    red = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    blue = image[:, :, 2].astype(np.int16)
    mask = (blue > 80) & (blue > red + 30) & (blue > green + 5)
    blob = _largest_blob(mask)
    if len(blob) < MIN_BLOB_PIXELS:
        return None
    ys = np.array([point[0] for point in blob], dtype=float)
    xs = np.array([point[1] for point in blob], dtype=float)
    return float(xs.mean()), float(ys.mean())


def ray_from_pixel(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: str,
    u: float,
    v: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    fovy = np.deg2rad(float(model.cam_fovy[cam_id]))
    fy = height / (2.0 * np.tan(fovy / 2.0))
    fx = fy
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    direction_cam = np.array([(u - cx) / fx, -(v - cy) / fy, -1.0], dtype=float)
    direction_cam /= np.linalg.norm(direction_cam)
    rotation = np.array(data.cam_xmat[cam_id]).reshape(3, 3)
    direction = rotation @ direction_cam
    origin = np.array(data.cam_xpos[cam_id], dtype=float)
    return origin, direction


def perceive_button(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    camera: str = "head_rgb",
) -> tuple[np.ndarray | None, str]:
    renderer.update_scene(data, camera=camera)
    image = renderer.render()
    centroid = blue_button_centroid(image)
    if centroid is None:
        return None, "perception: 머리 카메라에서 파란 버튼을 찾지 못함"
    origin, direction = ray_from_pixel(
        model,
        data,
        camera,
        centroid[0],
        centroid[1],
        image.shape[1],
        image.shape[0],
    )
    geomid = np.array([-1], dtype=np.int32)
    geomgroup = np.ones(6, dtype=np.uint8)
    distance = mujoco.mj_ray(
        model,
        data,
        origin,
        direction,
        geomgroup,
        1,
        -1,
        geomid,
    )
    if distance < 0 or geomid[0] < 0:
        return None, "perception: 버튼 픽셀 방향으로 충돌 지점이 없음"
    hit = origin + direction * float(distance)
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geomid[0])) or ""
    if geom_name not in {"button_3_geom", "panel"}:
        return None, f"perception: 광선이 버튼이 아닌 {geom_name}에 맞음"
    return hit, f"perception: {geom_name} 표면 {hit.round(3).tolist()}"
