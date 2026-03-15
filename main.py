import sys
import time
from camera import OrbitCamera
import numpy as np
import pygame

SPLAT_SCALE = 5.0
NUM_RENDER_MODES = 3 

def load_splats(path):
    with open(path, 'rb') as f:
        raw = f.read()

    bytes_per_splat = 32 # 32 bytes per splat
    n = len(raw) // bytes_per_splat
    data = np.frombuffer(raw[:n * bytes_per_splat], dtype=np.dtype([
        ('position', np.float32, (3)),
        ('scale', np.float32, (3)),
        ('color', np.uint8, (4)),
        ('rotation', np.uint8, (4)),
    ]))

    # Copy positions and convert coordinate system: splat files uses COLMAP convention (Y-down, Z-forward)
    # OpenGL convention uses Y-up, Z-backward so we negate Y and Z to get the right-handed system!
    positions = data['position'].astype(np.float32).copy()
    positions[:, 1] *= -1
    positions[:, 2] *= -1

    # Convert colors from uint8 to float32
    colors = data['color'].astype(np.float32) / 255.0

    print(f"Naloženo {n} splatov iz '{path}'")
    return positions, colors


def render_mode_1(framebuffer, px, py, valid, colors, width, height):
    px_v = np.round(px[valid]).astype(np.int32)
    py_v = np.round(py[valid]).astype(np.int32)
    in_bounds = (px_v >= 0) & (px_v < width) & (py_v >= 0) & (py_v < height)
    framebuffer[py_v[in_bounds], px_v[in_bounds]] = colors[valid][in_bounds, :3]


def render_mode_2(framebuffer, px, py, valid, depth, colors, width, height):
    half_size = SPLAT_SCALE / np.where(valid, depth, 1.0)
    order = np.argsort(depth)[::-1]  # far first, close last
    for i in order:
        if not valid[i]:
            continue
        center_x, center_y = px[i], py[i]
        x_low = max(0, int(np.floor(center_x - half_size[i])))
        x_high = min(width, int(np.ceil(center_x + half_size[i])))
        y_low = max(0, int(np.floor(center_y - half_size[i])))
        y_high = min(height, int(np.ceil(center_y + half_size[i])))
        if x_low >= x_high or y_low >= y_high:
            continue
        framebuffer[y_low:y_high, x_low:x_high] = colors[i, :3]
        

def render_mode_3(framebuffer, px, py, valid, depth, colors, width, height):
    """Order-correct blending: sort by depth (back-to-front), blend with straight alpha.
    RGB'd = (1 - As) * RGBd + As * RGBs. Framebuffer is initialized to (1, 1, 1)."""
    half_size = SPLAT_SCALE / np.where(valid, depth, 1.0)
    order = np.argsort(depth)[::-1]  # far first, close last (back-to-front)
    for i in order:
        if not valid[i]:
            continue
        as_alpha = colors[i, 3]
        if as_alpha <= 0.0: 
            continue
        # Source color with straight alpha: use RGBs, As for blending
        rgb_s = colors[i, :3]
        center_x, center_y = px[i], py[i]
        x_low = max(0, int(np.floor(center_x - half_size[i])))
        x_high = min(width, int(np.ceil(center_x + half_size[i])))
        y_low = max(0, int(np.floor(center_y - half_size[i])))
        y_high = min(height, int(np.ceil(center_y + half_size[i])))
        if x_low >= x_high or y_low >= y_high:
            continue
        # Blend: RGB'd = (1 - As) * RGBd + As * RGBs
        rgb_d = framebuffer[y_low:y_high, x_low:x_high]
        framebuffer[y_low:y_high, x_low:x_high] = (1.0 - as_alpha) * rgb_d + as_alpha * rgb_s


def render_points(framebuffer, positions, colors, view, proj, width, height, mode):
    N = len(positions)

    # Build homogeneous coordinates (N, 4) 
    # This means we add a 1 to the end of each position vector to make it a 4D vector so we can multiply it by the view and projection matrices.
    ones = np.ones((N, 1), dtype=np.float32)
    homogeneous_coordinates = np.concatenate([positions, ones], axis=1)


    model_view_projection_matrix = proj.astype(np.float64) @ view.astype(np.float64) # @ is matrix multiplication
    clip_space_coordinates = (model_view_projection_matrix @ homogeneous_coordinates.astype(np.float64).T).T  # We get the clip space coordinates by multiplying the model view projection matrix by the homogeneous coordinates.

    clip_space_w_coordinates = clip_space_coordinates[:, 3] # We get the w coordinate by taking the 4th column of the clip space coordinates.

    # Culling - keep only points with positive w (in front of camera)
    valid = clip_space_w_coordinates > 0

    # Perspective divide - convert clip space coordinates to NDC (normalized device coordinates) coordinates
    # because the clip space coordinates are in the range [-w, w] and we want to convert them to the range [-1, 1] so we can use them to calculate the pixel coordinates.
    ndc_x = np.where(valid, clip_space_coordinates[:, 0] / clip_space_w_coordinates, 0.0)
    ndc_y = np.where(valid, clip_space_coordinates[:, 1] / clip_space_w_coordinates, 0.0)

    # keep only points that are within the visible  square
    valid &= (np.abs(ndc_x) <= 1.0) & (np.abs(ndc_y) <= 1.0)

    # Pixel coordinates
    px = ( ndc_x * 0.5 + 0.5) * width
    py = (-ndc_y * 0.5 + 0.5) * height

    if mode == 1:
        render_mode_1(framebuffer, px, py, valid, colors, width, height)
    elif mode == 2:
        view_space = (view.astype(np.float64) @ homogeneous_coordinates.T).T
        view_z = view_space[:, 2]
        depth = -view_z  # positive depth in front of camera (camera looks along -Z)
        valid &= depth > 1e-6  # exclude behind camera for mode 2
        render_mode_2(framebuffer, px, py, valid, depth, colors, width, height)
    elif mode == 3:
        view_space = (view.astype(np.float64) @ homogeneous_coordinates.T).T
        view_z = view_space[:, 2]
        depth = -view_z
        valid &= depth > 1e-6
        render_mode_3(framebuffer, px, py, valid, depth, colors, width, height)


def camera_init(positions, width, height):
    centre = positions.mean(axis=0)
    span = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)) # distance between furthest apart points
    radius = span * 1.2 # we add a bit of padding to the radius so it doesn't touch the edges of the screen
    return OrbitCamera(
        target=centre,
        radius=radius,
        yaw=30.0,
        pitch=15.0,
        fovy=60.0,
        near=span * 0.001, 
        far=span * 10.0, 
        width=width,
        height=height,
    )

def main():
    if len(sys.argv) < 2:
        print("Manjka pot do datoteke (python main.py <path_to_splat_file>)")
        sys.exit(1)

    splat_path = sys.argv[1]
    positions, colors = load_splats(splat_path)

    WIDTH, HEIGHT = 1920, 1080 # Viewport size

    camera = camera_init(positions, WIDTH, HEIGHT)
    init_radius = camera.radius
    init_target = camera.target.copy()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    framebuffer = np.ones((HEIGHT, WIDTH, 3), dtype=np.float32) 
    surface_framebuffer = pygame.Surface((WIDTH, HEIGHT)) # surface_framebuffer is a pygame surface that we can blit to the screen
    frame_ms = 0.0
    render_mode = 1

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    camera.radius = max(0.01, camera.radius * 0.92)
                elif event.key == pygame.K_s:
                    camera.radius *= 1.08

                elif event.key == pygame.K_r:
                    camera.radius = init_radius
                    camera.target = init_target.copy()
                    camera.yaw    = 30.0
                    camera.pitch  = 15.0

                elif event.key == pygame.K_p:
                    fname = "screenshot.png"
                    pygame.image.save(screen, fname)

                elif event.key == pygame.K_1:
                    render_mode = 1
                elif event.key == pygame.K_2:
                    render_mode = 2
                elif event.key == pygame.K_3:
                    render_mode = 3

            else:
                camera.handle_event(event)

        if not running:
            break

        start_time = time.perf_counter()

        framebuffer[:] = 1.0  # clear to white before rendering
        view, proj = camera.get_view_proj()
        render_points(framebuffer, positions, colors, view, proj, WIDTH, HEIGHT, render_mode)

        frame_ms = (time.perf_counter() - start_time) * 1000.0

        # Convert float framebuffer to uint8 (needed for pygame surfarray)
        fb_uint8 = (np.clip(framebuffer, 0.0, 1.0) * 255).astype(np.uint8) 
        # pygame surfarray expects shape (W, H, 3) so we transpose the array from (H, W, 3)
        pygame.surfarray.blit_array(surface_framebuffer, fb_uint8.transpose(1, 0, 2))

        screen.blit(surface_framebuffer, (0, 0)) # blit the framebuffer to the screen at the origin

        display_text = [
            f"Splats: {len(positions)}    FPS: {1000.0 / max(frame_ms, 0.001):.1f}   Mode: {render_mode}",
            f"FOV: {camera.fovy:.0f} deg    Radius: {camera.radius:.3f}",
            "1 / 2 / 3  render mode   scroll / W / S  zoom   R  reset   P  screenshot",
        ]
        for row, text in enumerate(display_text):
            surf = pygame.font.SysFont("monospace", 14).render(text, True, (255, 255, 0), (0, 0, 0))
            screen.blit(surf, (8, 8 + row * 18))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
