import os
import argparse
import re
import logging
from glob import glob

import numpy as np
import pandas as pd
import torch
import lpips
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.fid import FrechetInceptionDistance

#def extract_id(fname):
   # m = re.search(r'id_(\d{8})', fname)
    #return m.group(1) if m else None

def extract_id(fname):
    m = re.match(r'^(\d+)_', fname)
    return m.group(1) if m else None



def compute_metrics(gt_img, sr_img, lpips_model, device):
    # to float
    gt = np.array(gt_img).astype(np.float32) / 255.0
    sr = np.array(sr_img).astype(np.float32) / 255.0
    if gt.shape != sr.shape:
        raise ValueError(f"Shape mismatch GT {gt.shape} vs SR {sr.shape}")
    # PSNR
    p = psnr(gt, sr, data_range=1.0)
    # SSIM
    s = ssim(gt, sr, data_range=1.0, channel_axis=2)
    # LPIPS
    gt_t = torch.from_numpy(gt.transpose(2,0,1)).unsqueeze(0).to(device)*2-1
    sr_t = torch.from_numpy(sr.transpose(2,0,1)).unsqueeze(0).to(device)*2-1
    with torch.no_grad():
        l = lpips_model(gt_t, sr_t).item()
    return p, s, l, gt_t, sr_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', required=True)
    parser.add_argument('--sr_dirs', nargs='+', required=True)
    parser.add_argument('--output_csv', default='sr_metrics_by_id.csv')
    parser.add_argument('--plot_png', default='sr_comparison.png')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # LPIPS and FID
    lpips_model = lpips.LPIPS(net='alex').to(device).eval()
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    # build GT map by id
    gt_paths = glob(os.path.join(args.gt_dir, '*.*'))
    gt_map = {extract_id(os.path.basename(p)): p for p in gt_paths if extract_id(os.path.basename(p))}
    logging.info(f"Found {len(gt_map)} GT images by id")

    records = []
    for sr_dir in args.sr_dirs:
        model_name = os.path.basename(sr_dir.rstrip('/'))
        logging.info(f"Evaluating {model_name}")
        sr_paths = glob(os.path.join(sr_dir, '*.*'))
        # reset lists
        psnrs, ssims, lpips_vals = [], [], []
        real = []
        fake = []
        fid_metric.reset()
        for sr_p in sr_paths:
            fid = extract_id(os.path.basename(sr_p))
            if fid not in gt_map:
                continue
            try:
                gt_img = Image.open(gt_map[fid]).convert('RGB')
                sr_img = Image.open(sr_p).convert('RGB')
                p, s, l, gt_t, sr_t = compute_metrics(gt_img, sr_img, lpips_model, device)
                psnrs.append(p)
                ssims.append(s)
                lpips_vals.append(l)
                with torch.no_grad():
                    real.append((gt_t*0.5+0.5)*255.0)
                    fake.append((sr_t*0.5+0.5)*255.0)
            except Exception as e:
                logging.warning(f"Metrics failed for ID {fid}: {e}")
                continue
        count = len(psnrs)
        if count==0:
            logging.warning(f"No valid for {model_name}, skipping")
            continue
        # FID compute
        real_batch = torch.cat(real, dim=0).to(torch.uint8)
        fake_batch = torch.cat(fake, dim=0).to(torch.uint8)
        fid_metric.update(real_batch, real=True)
        fid_metric.update(fake_batch, real=False)
        f = fid_metric.compute().item()
        records.append({'model':model_name,
                        'psnr':np.mean(psnrs), 'ssim':np.mean(ssims),
                        'lpips':np.mean(lpips_vals), 'fid':f})
    # save CSV
    df = pd.DataFrame(records)
    df.to_csv(args.output_csv, index=False)
    logging.info(f"Saved metrics to {args.output_csv}")

    # plot bar chart
    dfm = df.set_index('model')
    dfm[['psnr','ssim','lpips']].plot(kind='bar', subplots=False, figsize=(10,6))
    plt.title('SR Models Comparison')
    plt.ylabel('Value')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(args.plot_png)
    logging.info(f"Saved plot to {args.plot_png}")

if __name__=='__main__':
    main()
