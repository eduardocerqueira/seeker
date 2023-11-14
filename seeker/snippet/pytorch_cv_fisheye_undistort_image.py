#date: 2023-11-14T17:06:46Z
#url: https://api.github.com/gists/57c06b33035a98684e47add07933e486
#owner: https://api.github.com/users/Duke194

def init_fisheye_undistort_rectify_map_pytorch(K, D, R, P, size, m1type='CV_32FC1'):
    height, width = size
    device = K.device

    # Make sure K, D, R, P are all on the same device and in float format
    K = K.float().to(device)
    D = D.float().to(device)
    R = R.float().to(device) if R is not None else torch.eye(3).to(device)
    P = P.float().to(device) if P is not None else K.clone()

    # Calculate the inverse rectification transformation
    iR = torch.linalg.inv(P @ R).to(device)

    # Create meshgrid of pixel coordinates
    jj, ii = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
    ii = ii.to(device)
    jj = jj.to(device)

    # Initialize map1 and map2
    if m1type == 'CV_32FC1':
        map1 = torch.zeros((height, width), dtype=torch.float32, device=device)
        map2 = torch.zeros_like(map1)
    else:
        raise ValueError("Unsupported m1type. Only 'CV_32FC1' is supported here.")

    # Compute undistortion and rectification transformation for all pixels in parallel
    _x = ii * iR[0, 1] + iR[0, 2]
    _y = ii * iR[1, 1] + iR[1, 2]
    _w = ii * iR[2, 1] + iR[2, 2]

    # Apply transformation to x coordinates
    _x += jj * iR[0, 0]
    _y += jj * iR[1, 0]
    _w += jj * iR[2, 0]

    valid = _w > 0
    _x[~valid] = 0
    _y[~valid] = 0
    _w[~valid] = 1

    x = _x / _w
    y = _y / _w

    # Apply fisheye distortion
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan(r)
    theta_d = theta * (1 + D[0]*theta**2 + D[1]*theta**4 + D[2]*theta**6 + D[3]*theta**8)
    scale = torch.where(r == 0, torch.ones_like(r), theta_d / r)

    u = K[0, 0] * x * scale + K[0, 2]
    v = K[1, 1] * y * scale + K[1, 2]

    # Normalize the coordinates for grid_sample
    map1 = 2.0 * u / (width - 1) - 1.0
    map2 = 2.0 * v / (height - 1) - 1.0

    # Stack to get the final map in (H, W, 2) format
    map1 = torch.stack((map1, map2), dim=-1)

    return map1


def undistort_points(distorted, K, D, R=None, max_iter=10, epsilon=1e-6):
    assert len(D) == 4, "Distortion coefficients should be of length 4"
    assert K.size() == (3, 3), "Camera intrinsic matrix K should be 3x3"

    # Prepare K, D, and R
    K = K.to(distorted.dtype)
    D = D.to(distorted.dtype)
    if R is None:
        R = torch.eye(3, dtype=distorted.dtype, device=D.device)
    else:
        assert R.size() == (3, 3), "Rotation matrix R should be 3x3"

    # Convert points to normalized coordinates
    distorted = distorted.double()
    normalized = (distorted - K[:2, 2]) / K[:2, :2].diag()

    # Iteratively solve for theta
    r = torch.norm(normalized, dim=-1)
    r[r == 0] = epsilon  # Avoid division by zero
    theta_d = torch.atan(r)
    theta = theta_d.clone()

    for _ in range(max_iter):
        theta2 = theta**2
        theta4 = theta2**2
        theta6 = theta4 * theta2
        theta8 = theta4**4
        theta_terms = 1 + D[0]*theta2 + D[1]*theta4 + D[2]*theta6 + D[3]*theta8
        theta_fix = (theta * theta_terms - theta_d) / (theta_terms + theta2 * (2*D[0] + 4*D[1]*theta2 + 6*D[2]*theta4 + 8*D[3]*theta6))
        theta -= theta_fix
        if torch.max(torch.abs(theta_fix)) < epsilon:
            break

    # Calculate undistorted points
    scale = torch.tan(theta) / r
    undistorted = normalized * scale.unsqueeze(-1)

    # Reproject using rotation matrix R
    undistorted = torch.cat((undistorted, torch.ones_like(undistorted[:, :1])), dim=-1)
    undistorted = (R @ undistorted.T).T[:, :2]

    return undistorted


def estimate_new_camera_matrix_for_undistort_rectify(K, D, image_size, R=torch.eye(3, dtype=torch.double), balance=0.5, new_size=None, fov_scale=1.0):
    # Step 1: Undistort points
    h, w = image_size
    balance = min(max(balance, 0.0), 1.0)

    points = torch.tensor([[w/2, 0], [w, h/2], [w/2, h], [0, h/2]], dtype=torch.float32, device=D.device)
    points = undistort_points(points, K, D, R.to(D.device))  # You need to implement this function based on fisheye model

    # Calculate center and aspect ratio
    center_mass = points.mean(dim=0)
    aspect_ratio = K[0, 0] / K[1, 1]

    # Convert to identity ratio
    cn = center_mass * torch.tensor([1, aspect_ratio], device=center_mass.device)
    points[:, 1] *= aspect_ratio

    # Step 2: Find new focal lengths
    minx, miny = points.min(dim=0)[0]
    maxx, maxy = points.max(dim=0)[0]

    f1 = w * 0.5 / (cn[0] - minx)
    f2 = w * 0.5 / (maxx - cn[0])
    f3 = h * 0.5 * aspect_ratio / (cn[1] - miny)
    f4 = h * 0.5 * aspect_ratio / (maxy - cn[1])

    fmin = min(f1, f2, f3, f4)
    fmax = max(f1, f2, f3, f4)

    f = balance * fmin + (1.0 - balance) * fmax
    f *= 1.0 / fov_scale if fov_scale > 0 else 1.0

    # Step 3: Calculate new camera matrix
    new_f = torch.tensor([f, f / aspect_ratio], device=f.device)
    new_c = -cn * f + torch.tensor([w, h], device=f.device) * 0.5

    new_K = torch.tensor([[new_f[0], 0, new_c[0]],
                          [0, new_f[1], new_c[1]],
                          [0, 0, 1]], dtype=torch.float32, device=new_f.device)

    # Adjust for new size if provided
    if new_size is not None:
        new_w, new_h = new_size
        rx = new_w / w
        ry = new_h / h

        new_K[0, 0] *= rx
        new_K[1, 1] *= ry
        new_K[0, 2] *= rx
        new_K[1, 2] *= ry

    return new_K


def fisheye_undistort_image_pytorch(distorted, K, D, Knew=None, new_size=None):
    if len(distorted.shape) == 3:
        # Reshape distorted image for grid_sample
        distorted = distorted.unsqueeze(0)  # Add batch dimension

    # Assuming distorted is of shape (C, H, W)
    B, C, H, W = distorted.shape

    # Determine the size for the output image
    height, width = new_size if new_size is not None else (H, W)

    # Ensure Knew is initialized properly if None
    Knew = Knew if Knew is not None else K

    # Generate rectification maps
    map1_normalized = init_fisheye_undistort_rectify_map_pytorch(K, D, torch.eye(3, dtype=K.dtype, device=D.device), Knew, (height, width))

    # Add batch dimension and repeat for each batch
    map1_normalized = map1_normalized.expand(B, -1, -1, -1)

    # Apply grid sampling
    undistorted = F.grid_sample(distorted, map1_normalized, mode='bilinear', padding_mode='zeros')

    return undistorted


def main():
  image_batch = <your image batch Tensor(N, C, H, W)>
  K = <Insert your K Matrix Tensor(3x3)> 
  D = <Insert your D Vector Tensor(4)>

  k_new = estimate_new_camera_matrix_for_undistort_rectify(torch.tensor(K, device=device),
                                                           torch.tensor(D, device=device),
                                                           (image_batch.shape[-2:]),
                                                           balance=0.5)

  rectified_image_batch = fisheye_undistort_image_pytorch(image_batch.to(dtype=torch.float),
                                                          torch.tensor(K, device=device),
                                                          torch.tensor(D, device=device),
                                                          k_new).to(dtype=torch.uint8)
  
  
  
  