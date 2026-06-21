import os
import torch
import argparse
import numpy as np
from PIL import Image
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OrthographicCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    BlendParams,
    TexturesUV
)
import smplx as SMPL

class RenderFrontBack:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        smpl_path = os.path.abspath(os.path.join(__file__, "../../sample-data/SMPL/models/"))
        self.smpl = SMPL.create(smpl_path, model_type='smpl', gender='male').to(self.device)

        self.body_pose = torch.zeros(1, 69, device=self.device)
        self.betas = torch.zeros(1, 10, device=self.device)
        self.body_pose[0, 47] = -1.35
        self.body_pose[0, 50] = 1.30

        output = self.smpl(betas=self.betas, body_pose=self.body_pose, return_verts=True)
        self.verts = output.vertices[0]
        self.faces = self.smpl.faces_tensor.to(self.device)

        uv_obj = os.path.abspath(os.path.join(__file__, "../../sample-data/smpl_uv_20200910/smpl_uv.obj"))
        _, self.faces_verts, aux = load_obj(uv_obj)
        self.verts_uvs = aux.verts_uvs[None, ...].to(self.device)
        self.faces_uvs = self.faces_verts.textures_idx[None, ...].to(self.device)
        self.faces_idx = self.faces_verts.verts_idx.to(self.device)

        self.renderer = self.build_renderer()

    def build_renderer(self, image_size=512):
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1
        )
        blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))
        lights = PointLights(device=self.device)

        def render(mesh, azim):
            R, T = look_at_view_transform(dist=2.0, elev=0.0, azim=azim)
            cameras = OrthographicCameras(device=self.device, R=R, T=T)
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                shader=SoftPhongShader(device=self.device, cameras=cameras, lights=lights, blend_params=blend_params)
            )
            image = renderer(mesh)
            return image[0, ..., :3].detach().cpu().numpy()

        return render

    def render_textures(self, textures_folder, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        files = sorted(os.listdir(textures_folder))
        for fname in files:
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            tex_path = os.path.join(textures_folder, fname)
            print(f"Rendering front/back view for: {fname}")
            tex_np = np.array(Image.open(tex_path).convert("RGB"), dtype=np.float32) / 255.0
            tex = torch.from_numpy(tex_np).unsqueeze(0).to(self.device)
            textures = TexturesUV(maps=tex, faces_uvs=self.faces_uvs, verts_uvs=self.verts_uvs)
            mesh = Meshes(verts=[self.verts], faces=[self.faces_idx], textures=textures)

            img_front = self.renderer(mesh, azim=180.0)
            img_back = self.renderer(mesh, azim=0.0)

            basename = fname.rsplit('.', 1)[0]
            Image.fromarray((img_front * 255).astype(np.uint8)).save(os.path.join(output_dir, f"{basename}_front.png"))
            Image.fromarray((img_back * 255).astype(np.uint8)).save(os.path.join(output_dir, f"{basename}_back.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render front and back view from UV textures")
    parser.add_argument("--textures", "-i", required=True, help="Folder of UV texture images")
    parser.add_argument("--output_dir", "-o", default=None, help="Folder to save rendered images")
    args = parser.parse_args()

    tex_folder = args.textures
    out_folder = args.output_dir or os.path.join(tex_folder, "rendered_front_back")

    renderer = RenderFrontBack()
    renderer.render_textures(tex_folder, out_folder)
