import sys
import time
from camera import OrbitCamera
import numpy as np
import pygame

SPLAT_SCALE = 5.0
NUM_RENDER_MODES = 5

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

    # Copy positions from the file as-is; we treat the file's coordinate system as our world space.
    positions = data['position'].astype(np.float32).copy()

    # Convert colors from uint8 to float32
    colors = data['color'].astype(np.float32) / 255.0

    scales = data['scale'].astype(np.float32).copy()

    rotation_uint8 = data['rotation']
    rotations = (rotation_uint8.astype(np.float32) - 128.0) / 128.0 # (c−128)/128)
    # Normalize to unit quaternion  - described in the paper section 4.
    qnorm = np.linalg.norm(rotations, axis=1, keepdims=True)
    qnorm = np.where(qnorm < 1e-8, 1.0, qnorm)
    rotations = rotations / qnorm

    print(f"Naloženo {n} splatov iz '{path}'")
    return positions, colors, scales, rotations


def render_mode_1(framebuffer, px, py, valid, colors, width, height):
    px_v = np.round(px[valid]).astype(np.int32)
    py_v = np.round(py[valid]).astype(np.int32)
    in_bounds = (px_v >= 0) & (px_v < width) & (py_v >= 0) & (py_v < height)
    framebuffer[py_v[in_bounds], px_v[in_bounds]] = colors[valid][in_bounds, :3]


def render_mode_2(framebuffer, px, py, valid, depth, colors, width, height, splat_scale):
    half_size = splat_scale / np.where(valid, depth, 1.0)
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


def render_mode_3(framebuffer, px, py, valid, depth, colors, width, height, splat_scale):
    half_size = splat_scale / np.where(valid, depth, 1.0)
    order = np.argsort(depth)[::-1]  # far first, close last (back-to-front)
    for i in order:
        if not valid[i]:
            continue
        alpha_splat = colors[i, 3]
        if alpha_splat <= 0.0: 
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
        framebuffer[y_low:y_high, x_low:x_high] = (1.0 - alpha_splat) * rgb_d + alpha_splat * rgb_s


# Gaussian falloff - alpha at each pixel = splat_alpha * g(x), with g(x) = exp(-1/2 * (x-c)^T Sigma^{-1} (x-c)).
def render_mode_4(framebuffer, px, py, valid, depth, colors, width, height, splat_scale):
    half_size = splat_scale / np.where(valid, depth, 1.0)
    order = np.argsort(depth)[::-1]  # back-to-front
    for i in order:
        if not valid[i]:
            continue

        alpha_splat = colors[i, 3]
        if alpha_splat <= 0.0:
            continue

        rgb_s = colors[i, :3]
        center_x, center_y = float(px[i]), float(py[i])
        sigma = half_size[i]  # s/z in pixels (uniform scaling)

        # Calculate the bounds of the gaussian falloff so we don't calculate the gaussian falloff for pixels that are outside the bounds of the splat and waste time.
        x_low = max(0, int(np.floor(center_x - sigma * 3)))   
        x_high = min(width, int(np.ceil(center_x + sigma * 3)))
        y_low = max(0, int(np.floor(center_y - sigma * 3)))
        y_high = min(height, int(np.ceil(center_y + sigma * 3)))
        if x_low >= x_high or y_low >= y_high:
            continue # If the bounds are outside the width or height of the framebuffer, skip the pixel.

        yy, xx = np.ogrid[y_low:y_high, x_low:x_high] # Create a grid of y and x coordinates
        dx = xx.astype(np.float64) - center_x # x - center_x
        dy = yy.astype(np.float64) - center_y # y - center_y

        d_sq = dx**2 + dy**2 # Calculate the squared distance between the pixel and the center of the splat.
        sigma_sq = sigma**2 # Calculate the squared sigma.

        g = np.exp(-0.5 * d_sq / (sigma_sq + 1e-20))
        effective_alpha = (alpha_splat * g).astype(np.float32)[..., np.newaxis] # We multiply the alpha of the splat by the gaussian falloff to get the effective alpha.
        rgb_d = framebuffer[y_low:y_high, x_low:x_high] # Previous color of the pixel.
        framebuffer[y_low:y_high, x_low:x_high] = ((1.0 - effective_alpha) * rgb_d + effective_alpha * rgb_s) # Blend the previous color of the pixel with the new color of the pixel according to the effective alpha.

def quaternion_to_rotation_matrix(q):
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
    q_r, q_i, q_j, q_k = q[:, 0], q[:, 1], q[:, 2], q[:, 3]  
    s = 1.0 / (q_r * q_r + q_i * q_i + q_j * q_j + q_k * q_k + 1e-20)  # s = ‖q‖^{-2}
    two_s = 2.0 * s
    R = np.empty((len(q_r), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - two_s * (q_j * q_j + q_k * q_k)
    R[:, 0, 1] = two_s * (q_i * q_j - q_r * q_k)
    R[:, 0, 2] = two_s * (q_i * q_k + q_r * q_j)
    R[:, 1, 0] = two_s * (q_i * q_j + q_r * q_k)
    R[:, 1, 1] = 1 - two_s * (q_i * q_i + q_k * q_k)
    R[:, 1, 2] = two_s * (q_j * q_k - q_r * q_i)
    R[:, 2, 0] = two_s * (q_i * q_k - q_r * q_j)
    R[:, 2, 1] = two_s * (q_j * q_k + q_r * q_i)
    R[:, 2, 2] = 1 - two_s * (q_i * q_i + q_j * q_j)
    return R.astype(np.float32)

def build_covariance_matrix(rotations, scales):
    # Σ = R S Sᵀ Rᵀ 
    R = quaternion_to_rotation_matrix(rotations.astype(np.float64))
    S = np.zeros((len(scales), 3, 3), dtype=np.float64)
    S[:, 0, 0] = scales[:, 0]
    S[:, 1, 1] = scales[:, 1]
    S[:, 2, 2] = scales[:, 2]
    # We transpose (0, 2, 1) because we have a batch of 3×3 matrices and we only want to transpose each matrix, not the batch.
    covariance_matrix = (R @ S @ S.transpose(0, 2, 1) @ R.transpose(0, 2, 1)).astype(np.float32)
    return covariance_matrix

# Σ' = J W Σ Wᵀ Jᵀ
def project_cov3d_world_to_screen(cov3_world, view, view_pts3, fx, fy):
    W3 = view[:3, :3].astype(np.float32) # W (we get it from the view matrix - the upper-left 3×3 of the view matrix)

    # W Σ Wᵀ - for all N Gaussians. np.newaxis adds a leading dimension
    cov3_view = (W3[np.newaxis] @ cov3_world @ W3.T[np.newaxis]).astype(np.float32)

    # Build Jacobian (25 - formula in paper) 
    # u0, u1, u2 are the x, y, z coordinates of the Gaussian in camera (view) space.
    u0 = view_pts3[:, 0].astype(np.float32)  # X = u0
    u1 = view_pts3[:, 1].astype(np.float32)  # Y = u1
    u2 = view_pts3[:, 2].astype(np.float32)  # Z = u2
    u2_safe = np.where(np.abs(u2) < 1e-6, np.float32(-1e-6), u2) # Avoid division by zero
    inv_u2 = np.float32(1.0) / u2_safe       # 1 / u2

    N = len(u2)
    J = np.zeros((N, 2, 3), dtype=np.float32)
    #   J = [ 1/u2     0    -u0/u2^2 ]
    #       [   0    1/u2   -u1/u2^2 ]
    #       [ u0/l   u1/l    u2/l   ] ,  with l = ||(u0,u1,u2)^T||
    # We only need the first two rows (mapping to 2D), then scale by fx, fy to get pixel coords.
    # QUOTE:
    """
    Σ' = J W Σ Wᵀ Jᵀ
    where 𝐽 is the Jacobian of the affine approximation of the projective
    transformation. Zwicker et al. [2001a] also show that if we skip the
    third row and column of Σ′, we obtain a 2×2 variance matrix with
    the same structure and properties as if we would start from planar
    points with normals, as in previous work [Kopanas et al. 2021].
    """
    J[:, 0, 0] = -float(fx) * inv_u2
    J[:, 0, 2] = float(fx) * u0 * inv_u2**2  
    J[:, 1, 1] = float(fy) * inv_u2
    J[:, 1, 2] = -float(fy) * u1 * inv_u2**2

    # Σ' = J Σ_view Jᵀ = J W Σ Wᵀ Jᵀ
    cov2 = (J @ cov3_view @ J.transpose(0, 2, 1)).astype(np.float32)
    return cov2


# Non-uniform Gaussian falloff 
def render_mode_5(framebuffer, px, py, valid, depth, view_pts3, colors, rotations, scales, view, proj, width, height):
    fx = float(proj[0, 0]) * width * 0.5
    fy = float(proj[1, 1]) * height * 0.5

    cov3_world = build_covariance_matrix(rotations, scales)
    cov2 = project_cov3d_world_to_screen(cov3_world, view, view_pts3, fx, fy)
    
    cov2[:, 0, 0] += 0.3
    cov2[:, 1, 1] += 0.3  # Regularize to avoid degenerate splats

    order = np.argsort(depth)[::-1]  # back-to-front
    for i in order:
        if not valid[i] or colors[i, 3] <= 0.0:
            continue
        rgb_s = colors[i, :3]
        center_x, center_y = float(px[i]), float(py[i])

        a, b, c = cov2[i, 0, 0], cov2[i, 0, 1], cov2[i, 1, 1]  # Σ' = [[a,b],[b,c]] for this splat
        det = max(a * c - b * b, 1e-10)
        inv_det = 1.0 / det
        radius = min(int(np.ceil(2.0 * np.sqrt(max(a + c, 1e-4)))), 96)

        # Bounds of the Gaussian so we don't compute it for pixels outside the splat.
        x_low = max(0, int(center_x) - radius)
        x_high = min(width, int(center_x) + radius + 1)
        y_low = max(0, int(center_y) - radius)
        y_high = min(height, int(center_y) + radius + 1)
        if x_low >= x_high or y_low >= y_high:
            continue  # Bounds outside the framebuffer, skip.

        yy, xx = np.ogrid[y_low:y_high, x_low:x_high]  # Grid of pixel coordinates
        dx = xx.astype(np.float64) - center_x  # x - center_x
        dy = yy.astype(np.float64) - center_y  # y - center_y

        # Exponent for elliptical Gaussian - ZPvBG01 (9 - enačba)
        d_sq = (c * dx * dx - 2.0 * b * dx * dy + a * dy * dy) * inv_det
        g = np.exp(-0.5 * d_sq)
        alpha_splat = colors[i, 3]
        effective_alpha = (alpha_splat * g).astype(np.float32)[..., np.newaxis]
        rgb_d = framebuffer[y_low:y_high, x_low:x_high]  # Previous color of the pixel.
        framebuffer[y_low:y_high, x_low:x_high] = (1.0 - effective_alpha) * rgb_d + effective_alpha * rgb_s  # Blend with effective alpha.


def render_points(framebuffer, positions, colors, view, proj, width, height, mode, splat_scale, scales=None, rotations=None):
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
        render_mode_2(framebuffer, px, py, valid, depth, colors, width, height, splat_scale)
    elif mode == 3:
        view_space = (view.astype(np.float64) @ homogeneous_coordinates.T).T
        view_z = view_space[:, 2]
        depth = -view_z
        valid &= depth > 1e-6
        render_mode_3(framebuffer, px, py, valid, depth, colors, width, height, splat_scale)
    elif mode == 4:
        view_space = (view.astype(np.float64) @ homogeneous_coordinates.T).T
        view_z = view_space[:, 2]
        depth = -view_z
        valid &= depth > 1e-6
        render_mode_4(framebuffer, px, py, valid, depth, colors, width, height, splat_scale)
    elif mode == 5:
        view_space = (view.astype(np.float64) @ homogeneous_coordinates.T).T
        view_z = view_space[:, 2]
        depth = -view_z
        valid &= depth > 1e-6
        view_pts3 = view_space[:, :3] # view space x y z coordinates (N,3)
        render_mode_5(framebuffer, px, py, valid, depth, view_pts3, colors, rotations, scales, view, proj, width, height)


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
    positions, colors, scales, rotations = load_splats(splat_path)

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
    splat_scale = SPLAT_SCALE

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
                    splat_scale = SPLAT_SCALE

                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    splat_scale = min(100.0, splat_scale * 1.2)
                elif event.key == pygame.K_MINUS:
                    splat_scale = max(0.1, splat_scale / 1.2)

                elif event.key == pygame.K_p:
                    fname = "screenshot.png"
                    pygame.image.save(screen, fname)

                elif event.key == pygame.K_1:
                    render_mode = 1
                elif event.key == pygame.K_2:
                    render_mode = 2
                elif event.key == pygame.K_3:
                    render_mode = 3
                elif event.key == pygame.K_4:
                    render_mode = 4
                elif event.key == pygame.K_5:
                    render_mode = 5

            else:
                camera.handle_event(event)

        if not running:
            break

        start_time = time.perf_counter()

        framebuffer[:] = 1.0  # clear to white before rendering
        view, proj = camera.get_view_proj()
        render_points(framebuffer, positions, colors, view, proj, WIDTH, HEIGHT, render_mode, splat_scale, scales=scales, rotations=rotations)

        frame_ms = (time.perf_counter() - start_time) * 1000.0

        # Convert float framebuffer to uint8 (needed for pygame surfarray)
        fb_uint8 = (np.clip(framebuffer, 0.0, 1.0) * 255).astype(np.uint8) 
        # pygame surfarray expects shape (W, H, 3) so we transpose the array from (H, W, 3)
        pygame.surfarray.blit_array(surface_framebuffer, fb_uint8.transpose(1, 0, 2))

        screen.blit(surface_framebuffer, (0, 0)) # blit the framebuffer to the screen at the origin

        display_text = [
            f"Splats: {len(positions)}    FPS: {1000.0 / max(frame_ms, 0.001):.1f}   Mode: {render_mode}   Splat scale: {splat_scale:.2f}",
            f"FOV: {camera.fovy:.0f} deg    Radius: {camera.radius:.3f}",
            "1-5  mode   + / -  splat scale   W / S  zoom   R  reset   P  screenshot",
        ]
        for row, text in enumerate(display_text):
            surf = pygame.font.SysFont("monospace", 14).render(text, True, (255, 255, 0), (0, 0, 0))
            screen.blit(surf, (8, 8 + row * 18))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
