#date: 2024-08-26T17:09:09Z
#url: https://api.github.com/gists/d9ec439baf6aefe4b302d2b3a00378e3
#owner: https://api.github.com/users/wufeim

import _init_paths

import os

import numpy as np
import open3d
from PIL import Image
import torch
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, PerspectiveCameras, look_at_view_transform, camera_position_from_spherical_angles
from pytorch3d.renderer import MeshRenderer, HardPhongShader, PointLights, TexturesVertex, SoftPhongShader, BlendParams
from pytorch3d.structures import Meshes

from src.utils.render_utils import rotation_theta, pre_process_mesh_pascal, load_off

from helpers import get_model_path

models = [
    '3063f19f09b743b8923d7d242824ef42',
    '128d94088b7b45e8b6b64b56a6c96f77',
    '348d29370b704ecf96a61fa6b9c40b21',
    'ff77cb8deb8647fca89fca4707ba1cfc',
    'd84491f443384ee488593cc6f0f0839e',
    '9a1eb30b199d441aa230c389cabc58a7']


sigma = 1e-4
device = torch.device('cuda:0')
color=(0.20392156862745098, 0.596078431372549, 0.8588235294117647)

raster_settings = RasterizationSettings(
    image_size=(256, 256),
    blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
    faces_per_pixel=100,
    perspective_correct=False,
    max_faces_per_bin=10000)  # bin_size=0
lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=None,
        raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, lights=lights, cameras=None))
this_camera = PerspectiveCameras(focal_length=3500, principal_point=((128, 128),), image_size=((256, 256),), in_ndc=False, device=device)
phong_renderer.rasterizer.cameras = this_camera
phong_renderer.shader.cameras = this_camera
renderer = phong_renderer

for m in models:
    from pytorch3d.io import IO
    from pytorch3d.io.experimental_gltf_io import MeshGlbFormat

    io = IO()
    io.register_meshes_format(MeshGlbFormat())
    mesh = io.load_mesh(f'new_meshes/{m}.glb', include_textures=True, device=device)

    b = 1
    azimuth = (torch.ones((b, 1)) * 1.0471975512).float().to('cuda:0')
    elevation = (torch.ones((b, 1)) * 0.52359877559).float().to('cuda:0')
    theta = (torch.ones((b, 1)) * 0.0).float().to('cuda:0')
    distance = (torch.ones((b, 1)) * 18.0).float().to('cuda:0')

    R, T = look_at_view_transform(dist=distance, azim=azimuth, elev=elevation, degrees=False, device=device)
    R = torch.bmm(R, rotation_theta(theta, device_=device))
    img = renderer(mesh, R=R, T=T)
    img = torch.ones_like(img[:, :, :, :3]) * (1 - img[:, :, :, 3:4]) + img[:, :, :, :3] * img[:, :, :, 3:4]
    img = np.clip(np.rint(img[0].detach().cpu().numpy() * 255), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(f'debug_{m}.png')
