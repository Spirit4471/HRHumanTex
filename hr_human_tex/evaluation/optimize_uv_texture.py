import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OrthographicCameras,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    SoftPhongShader,
    TexturesUV,
    BlendParams,
)
import lpips
import torchvision.models as models
import torchvision.transforms as T
from smplx import SMPL

# ------------------------------------------------
# UV Texture Optimization with GT and SMPL pose + Silhouette Mask
# ------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="UV Texture Optimization with Silhouette Mask")
    parser.add_argument('--uv_template', type=str, required=True, help="Path to UV OBJ template")
    parser.add_argument('--initial_texture', type=str, required=True, help="Initial UV texture PNG")
    parser.add_argument('--smpl_model_dir', type=str, required=True, help="SMPL model directory")
    parser.add_argument('--npz_file', type=str, required=True, help="NPZ file with SMPL parameters and camera")
    parser.add_argument('--gt_image', type=str, required=True, help="Ground-truth image for optimization")
    parser.add_argument('--iters', type=int, default=1000, help="Number of optimization steps")
    parser.add_argument('--lr', type=float, default=1e-2, help="Learning rate")
    parser.add_argument('--out_dir', type=str, default='./opt_results', help="Output directory")
    return parser.parse_args()

# device and losses
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpips_model = lpips.LPIPS(net='alex').to(DEVICE)
vgg = models.vgg16(pretrained=True).features[:16].to(DEVICE).eval()
vgg_transform = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

# renderer factory
def get_renderer(resolution):
    rast = RasterizationSettings(image_size=resolution, blur_radius=0.0, faces_per_pixel=1)
    blend = BlendParams(background_color=(1.0,1.0,1.0))
    def render(mesh, R, T):
        cameras = OrthographicCameras(device=DEVICE, R=R, T=T)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=rast),
            shader=SoftPhongShader(device=DEVICE, cameras=cameras,
                                   lights=PointLights(device=DEVICE),
                                   blend_params=blend)
        )
        images = renderer(mesh)
        return images[..., :3]
    return render

if __name__=='__main__':
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # load UV template
    verts_uv, faces_uv, aux = load_obj(args.uv_template)
    verts_uvs = aux.verts_uvs.unsqueeze(0).to(DEVICE)
    faces_uvs = faces_uv.textures_idx.unsqueeze(0).to(DEVICE)
    faces_idx = faces_uv.verts_idx.to(DEVICE)

    # load initial texture
    init = Image.open(args.initial_texture).convert('RGB').resize((1024,1024))
    tex_np = np.array(init, dtype=np.float32)/255.0
    tex = torch.from_numpy(tex_np).unsqueeze(0).to(DEVICE).float()
    tex.requires_grad_(True)

    # load GT image
    gt = Image.open(args.gt_image).convert('RGB').resize((1024,1024))
    gt_np = np.array(gt, dtype=np.float32)/255.0
    gt_t = torch.from_numpy(gt_np).unsqueeze(0).to(DEVICE).permute(0,3,1,2)

    # load NPZ
    raw_npz = np.load(args.npz_file, allow_pickle=True)
    if 'results' in raw_npz.files:
        data = raw_npz['results'].item()
    else:
        data = {k: raw_npz[k] for k in raw_npz.files}

    # SMPL model
    smpl = SMPL(model_path=args.smpl_model_dir, gender='neutral').to(DEVICE)
    if 'smpl_thetas' in data:
        thetas = torch.tensor(data['smpl_thetas'],dtype=torch.float32).view(1,-1).to(DEVICE)
        gob = thetas[:,:3]; bod = thetas[:,3:]
    else:
        gob = torch.tensor(data.get('global_orient',np.zeros(3)),dtype=torch.float32).view(1,3).to(DEVICE)
        bod = torch.tensor(data.get('body_pose',np.zeros(69)),dtype=torch.float32).view(1,69).to(DEVICE)
    betas = torch.tensor(data.get('smpl_betas',np.zeros(10)),dtype=torch.float32).view(1,10).to(DEVICE)
    out = smpl(betas=betas, body_pose=bod, global_orient=gob, return_verts=True)
    verts_smpl = out.vertices[0]

    # camera
    cam_t = np.asarray(data.get('cam_trans', np.array([0.,0.,2.])), dtype=float).flatten()
    if cam_t.size < 3:
        raise ValueError(f"cam_trans must have at least 3 values, got {cam_t.size}")
    tx, ty, tz = cam_t[:3]
    dist = float(np.linalg.norm(cam_t))
    elev = float(np.degrees(np.arcsin(ty/dist))) if dist>1e-3 else 0.0
    raw_azim = float(np.degrees(np.arctan2(tx,tz)))
    azim = 180.0 - raw_azim
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    R, T = R.to(DEVICE), T.to(DEVICE)

    # silhouette rasterizer for mask
    sil_settings = RasterizationSettings(image_size=1024, blur_radius=0.0, faces_per_pixel=1)
    sil_cameras = OrthographicCameras(device=DEVICE, R=R, T=T)
    sil_rasterizer = MeshRasterizer(cameras=sil_cameras, raster_settings=sil_settings)

    # optimization
    opt = torch.optim.Adam([tex], lr=args.lr)
    render = get_renderer(1024)

    for i in tqdm(range(args.iters)):
        opt.zero_grad()
        textures = TexturesUV(maps=tex, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
        mesh = Meshes(verts=[verts_smpl], faces=[faces_idx], textures=textures)
        # render
        img = render(mesh, R, T)  # [1,H,W,3]
        # mask via silhouette
        fragments = sil_rasterizer(mesh)
        face_idx = fragments.pix_to_face[...,0]  # (1,H,W)
        mask = (face_idx >= 0).float().unsqueeze(1)
        # correct orientation
        img = torch.flip(img, dims=[1,2])
        img_t = img.permute(0,3,1,2)
        # masked
        masked_img = img_t * mask
        masked_gt = gt_t * mask
        # losses
        l2 = ((masked_img - masked_gt)**2).sum() / mask.sum()
        lp = lpips_model(masked_img*2-1, masked_gt*2-1).mean()
        f1 = vgg(vgg_transform(masked_img[0]))
        f2 = vgg(vgg_transform(masked_gt[0]))
        vggl = (f1 - f2).abs().mean()
        loss = l2 + 0.1*lp + 0.01*vggl
        loss.backward(retain_graph=True)
        opt.step()
        if i % 100 == 0:
            tm = tex.detach().cpu().numpy()[0]
            Image.fromarray((np.clip(tm,0,1)*255).astype(np.uint8)).save(os.path.join(args.out_dir, f"texture_{i}.png"))
            rm = img.detach().cpu().numpy()[0]
            Image.fromarray((np.clip(rm,0,1)*255).astype(np.uint8)).save(os.path.join(args.out_dir, f"render_{i}.png"))
            print(f"Iter {i}, loss={loss.item():.4f}")
    # final
    tf = tex.detach().cpu().numpy()[0]
    Image.fromarray((np.clip(tf,0,1)*255).astype(np.uint8)).save(os.path.join(args.out_dir,"texture_final.png"))
