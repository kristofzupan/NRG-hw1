import math
import numpy as np
import pygame

# 4x4 view matrix: world space to camera space.
def look_at(eye, target):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    world_up = np.array([0.0, 1.0, 0.0]) # world up is +Y

    # Camera axes in world space (right-handed: X right, Y up, Z out of screen)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right) 
    if right_norm < 1e-8:
        right = np.array([1.0, 0.0, 0.0])  # eye above/below target
    else:
        right = right / right_norm

    up = np.cross(right, forward)

    # View matrix
    M = np.eye(4, dtype=np.float64)
    M[0, :3] = right
    M[0, 3] = -np.dot(right, eye)
    M[1, :3] = up
    M[1, 3] = -np.dot(up, eye)
    M[2, :3] = -forward
    M[2, 3] = np.dot(forward, eye)

    return M.astype(np.float32)

# 4x4 perspective matrix: camera space to clip space (NDC).
def perspective(fovy_deg, aspect, near, far):
    half_fov_rad = math.radians(fovy_deg) / 2.0 # half of the vertical field of view in radians
    scale_y = 1.0 / math.tan(half_fov_rad)  # Scale so normalized device coordinates y in [-1,1] at z=-1
    scale_x = scale_y / aspect

    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = scale_x
    M[1, 1] = scale_y
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2.0 * far * near) / (near - far)
    M[3, 2] = -1.0
    return M


class OrbitCamera:
    def __init__(self, target, radius, yaw=30.0, pitch=15.0, fovy=60.0, near=0.01, far=1000.0, width=900, height=700):
        self.target = np.array(target, dtype=np.float64)
        self.radius = float(radius)
        self.yaw = float(yaw)      # degrees, around world Y
        self.pitch = float(pitch) # degrees, clamped later to [-89, 89]
        self.fovy = float(fovy)
        self.near = float(near)
        self.far = float(far)
        self.width = width
        self.height = height
        self._dragging_left = False
        self._dragging_right = False
        self._last_mouse_xy = (0, 0)

    def get_view_proj(self):
        # Compute eye position from target, radius, yaw, pitch
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        # Spherical to Cartesian: offset from target
        x = self.radius * math.cos(pitch_rad) * math.sin(yaw_rad)
        y = self.radius * math.sin(pitch_rad)
        z = self.radius * math.cos(pitch_rad) * math.cos(yaw_rad)
        eye = self.target + np.array([x, y, z])

        view = look_at(eye, self.target)
        aspect = self.width / self.height
        proj = perspective(self.fovy, aspect, self.near, self.far)
        return view, proj

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # left
                self._dragging_left = True
                self._last_mouse_xy = event.pos
            elif event.button == 3:  # right
                self._dragging_right = True
                self._last_mouse_xy = event.pos
            elif event.button == 4:  # scroll up - zoom in
                self.radius = max(0.01, self.radius * 0.92)
            elif event.button == 5:  # scroll down - zoom out
                self.radius *= 1.08

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self._dragging_left = False
            elif event.button == 3:
                self._dragging_right = False

        elif event.type == pygame.MOUSEMOTION:
            dx = event.pos[0] - self._last_mouse_xy[0]
            dy = event.pos[1] - self._last_mouse_xy[1]
            self._last_mouse_xy = event.pos

            if self._dragging_left and (dx != 0 or dy != 0):
                # Orbit: change yaw and pitch
                self.yaw -= dx * 0.4
                self.pitch -= dy * 0.4
                self.pitch = max(-89.0, min(89.0, self.pitch))

            elif self._dragging_right and (dx != 0 or dy != 0):
                # Pan: move target along camera's right and up
                yaw_rad = math.radians(self.yaw)
                pitch_rad = math.radians(self.pitch)
                camera_right = np.array([math.cos(yaw_rad), 0.0, -math.sin(yaw_rad)])
                camera_up = np.array([
                    -math.sin(pitch_rad) * math.sin(yaw_rad),
                    math.cos(pitch_rad),
                    -math.sin(pitch_rad) * math.cos(yaw_rad),
                ])
                pan_scale = self.radius * 0.001
                self.target -= camera_right * dx * pan_scale
                self.target += camera_up * dy * pan_scale
