import os
import argparse
import re
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import lpips
from smplx import SMPL
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
import csv
from tqdm import tqdm
import warnings

# force single GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

warnings.filterwarnings("ignore", category=UserWarning)

_lpips_model = None
FAILED_LOG_PATH = "failed_samples.txt"
DEBUG_SAVE_FOLDER = "debug"
os.makedirs(DEBUG_SAVE_FOLDER, exist_ok=True)


def get_lpips_model(device):
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net='alex').to(device)
    return _lpips_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SMPL texture rendering using PyTorch3D")
    parser.add_argument('--gt_folder', type=str, required=True)
    parser.add_argument('--texture_folder', type=str, required=True)
    parser.add_argument('--npz_folder', type=str, required=True)
    parser.add_argument('--dp_folder',      type=str, required=True)
    parser.add_argument('--sil_folder', type=str, required=False,help="Optional silhouette mask folder from SGHM")
    parser.add_argument('--save_folder', type=str, required=True)
    parser.add_argument('--csv_path', type=str, default="results.csv")
    return parser.parse_args()


#def extract_id_number(fname):
    #m = re.search(r'id_(\d{8})', fname)
    #return m.group(1) if m else None

def extract_id_number(fname):
    name = os.path.splitext(fname)[0]
    return name if name.isdigit() else None


def find_file(idn, folder):
    for f in os.listdir(folder):
        if idn in f:
            return os.path.join(folder, f)
    return None

# preload UV template once
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
dataset_obj = load_obj(os.path.join(_PROJECT_ROOT, "..", "sample-data", "smpl_uv_20200910", "smpl_uv.obj"))
verts_uvs_template = dataset_obj[2].verts_uvs.unsqueeze(0)
faces_uvs_template = dataset_obj[1].textures_idx.unsqueeze(0)
faces_verts_template = dataset_obj[1].verts_idx
print(f"[DEBUG] UV template: verts_uvs={verts_uvs_template.shape}, faces_uvs={faces_uvs_template.shape}, faces_verts={faces_verts_template.shape}")


def render_mesh_textured(device, verts, tex_np, verts_uvs, faces_uvs, faces_verts,
                         image_size=512, cam_pos=None, mesh_rot=0):
    """
    Render a textured mesh with corrected camera orientation (frontal view, upright).
    mesh_rot: additional rotation around vertical axis (degrees).
    cam_pos: (dist, elev).
    """
    # move data onto device
    verts = verts.to(device)
    verts_uvs = verts_uvs.to(device)
    faces_uvs = faces_uvs.to(device)
    faces_verts = faces_verts.to(device)

    # prepare texture map
    tex = torch.from_numpy(tex_np / 255.0).unsqueeze(0).to(device)
    textures = TexturesUV(maps=tex, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
    mesh = Meshes(verts=[verts], faces=[faces_verts], textures=textures)

    # set up lighting
    lights = PointLights(device=device,
                         ambient_color=[[1.0,1.0,1.0]],
                         diffuse_color=[[0.0,0.0,0.0]],
                         specular_color=[[0.0,0.0,0.0]])

    # camera defaults
    if cam_pos is None:
        dist, elev = 2.0, 0.0
    else:
        dist, elev = float(cam_pos[0].item()), float(cam_pos[1].item())
    # ensure frontal view: azimuth=180
    azim = 180.0 + mesh_rot
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    R, T = R.to(device), T.to(device)
    cameras = OrthographicCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
    blend_params = BlendParams(background_color=(1.0,1.0,1.0))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights, blend_params=blend_params)
    )

    # render and detach grads before conversion
    images = renderer(mesh)
    img_tensor = images[0, ..., :3].detach().cpu()
    img = (img_tensor.numpy() * 255).astype(np.uint8)
    return img


def evaluate_one_sample(args_tuple):
    idn, gt_p, tex_p, npz_p, dp_p, sil_p, save_folder = args_tuple
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=='cuda':
        torch.cuda.set_device(0)

    # load NPZ
    data_npz = np.load(npz_p, allow_pickle=True)
    if 'results' in data_npz.files:
        data = data_npz['results'].item()
    else:
        data = {k: data_npz[k] for k in data_npz.files}

    # SMPL parameters
    thetas_np = data.get('smpl_thetas')
    if thetas_np is not None:
        thetas_np = np.array(thetas_np).reshape(-1)
        thetas = torch.tensor(thetas_np, dtype=torch.float32).to(device)
        gob = thetas[:3].unsqueeze(0)
        bod = thetas[3:].unsqueeze(0)
    else:
        gob = torch.zeros(1,3, device=device)
        bod = torch.zeros(1,69, device=device)
    betas_np = np.array(data.get('smpl_betas', np.zeros((10,))), dtype=np.float32).reshape(1,10)
    betas = torch.tensor(betas_np, dtype=torch.float32).to(device)

    # SMPL forward
    smpl = SMPL(model_path=os.path.join(_PROJECT_ROOT, "..", "sample-data", "SMPL", "models", "smpl"), gender='neutral').to(device)
    try:
        out = smpl(betas=betas, body_pose=bod, global_orient=gob, return_verts=True)
    except Exception as ex:
        print(f"[ERROR] SMPL forward failed for {idn}: {ex}")
        return (idn, -1, -1)
    verts = out.vertices[0]

    # load texture
    tex_img = Image.open(tex_p).convert('RGB')
    if tex_img.size != (1024, 1024):
        tex_img = tex_img.resize((1024, 1024), resample=Image.BILINEAR)
    tex_np = np.array(tex_img, dtype=np.float32)

    # render
    try:
        img_r = render_mesh_textured(
            device, verts, tex_np,
            verts_uvs_template, faces_uvs_template, faces_verts_template,
            image_size=1024, cam_pos=torch.tensor([2.0,0.0,0.0], device=device)
        )
    except Exception as ex:
        print(f"[ERROR] Render failed for {idn}: {ex}")
        return (idn, -1, -1)

    # upright orientation flip before metrics
    img_r = np.flipud(img_r)
    # horizontal mirror to match GT orientation
    img_r = np.fliplr(img_r)

    # load GT
    gt_img = np.array(Image.open(gt_p).convert('RGB').resize((1024,1024)), dtype=np.float32)

        
    dp_mask = np.array(
        Image.open(dp_p).convert('L')           
             .resize((1024,1024), Image.NEAREST) 
    )
    sil_mask = np.array(
        Image.open(sil_p).convert('L')
             .resize((1024,1024), Image.NEAREST)
    )
    dp_binary = (dp_mask > 128).astype(np.uint8)
    sil_binary = (sil_mask > 128).astype(np.uint8)
    mask = (dp_binary & sil_binary).astype(np.uint8)
    
    img_r_masked = img_r * mask[...,None]
    gt_masked    = gt_img * mask[...,None]

    # compute metrics
    print(f"[DEBUG] SSIM inputs shapes: GT {gt_img.shape}, Rendered {img_r.shape}")
    try:
        ssim_val = ssim(gt_masked/255.0, img_r_masked/255.0,
                        channel_axis=2, win_size=7, data_range=1.0)
    except TypeError:
        ssim_val = ssim(gt_masked/255.0, img_r_masked/255.0,
                        multichannel=True, win_size=7, data_range=1.0)

    # LPIPS
    lp = get_lpips_model(device)
    im1 = torch.from_numpy(img_r_masked/255.0).float().permute(2,0,1).unsqueeze(0).to(device)*2-1
    im2 = torch.from_numpy(gt_masked/255.0).float().permute(2,0,1).unsqueeze(0).to(device)*2-1
    lpips_val = lp(im1, im2).item()

    # save render
    os.makedirs(save_folder, exist_ok=True)
    out_path = os.path.join(save_folder, f"{idn}_render.png")
    Image.fromarray(img_r).save(out_path)
    print(f"[INFO] Saved render for {idn} -> {out_path}")

    return (idn, ssim_val, lpips_val)


def main():
    args = parse_args()
    if os.path.exists(FAILED_LOG_PATH): os.remove(FAILED_LOG_PATH)
    os.makedirs(args.save_folder, exist_ok=True)

    gts = sorted([f for f in os.listdir(args.gt_folder) if f.lower().endswith(('.png','.jpg'))])
    inp = []
    for g in gts:
        idn = extract_id_number(g)
        if not idn: continue
        gt_p = os.path.join(args.gt_folder, g)
        tx_p  = find_file(idn, args.texture_folder)
        npz_p = find_file(idn, args.npz_folder)
        dp_p  = find_file(idn, args.dp_folder)
        sil_p = find_file(idn, args.sil_folder)
        if not (tx_p and npz_p and dp_p):
            print(f"[WARN] Missing texture or npz for {idn}")
            continue
        inp.append((idn, gt_p, tx_p, npz_p, dp_p, sil_p, args.save_folder))

    print(f"[INFO] Evaluating {len(inp)} samples")
    results = []
    for x in tqdm(inp):
        res = evaluate_one_sample(x)
        if res is not None:
            results.append(res)

    # write CSV
    with open(args.csv_path, 'w', newline='') as cf:
        wr = csv.writer(cf)
        wr.writerow(['ID','SSIM','LPIPS'])
        wr.writerows(results)
    print(f"[INFO] Done. Results at {args.csv_path}")

if __name__ == '__main__':
    main()
