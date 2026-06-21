import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Scan for result CSVs
csv_files = sorted(glob.glob("*results.csv"))
gt_name = 'PseudoGTresults.csv'  
if gt_name not in [os.path.basename(f) for f in csv_files]:
    raise FileNotFoundError(f"GT results file '{gt_name}' not found")

# 2. Read and compute per-file averages
data = []
for f in csv_files:
    model = os.path.splitext(os.path.basename(f))[0]
    df = pd.read_csv(f)
    data.append({
        'model': model,
        'avg_ssim': df['SSIM'].mean(),
        'avg_lpips': df['LPIPS'].mean(),
    })
summary = pd.DataFrame(data).set_index('model')

# 3. Compute GT baseline and deltas if needed
gt_ssim  = summary.loc['PseudoGTresults', 'avg_ssim']
gt_lpips = summary.loc['PseudoGTresults', 'avg_lpips']
summary['ssim_diff']  = summary['avg_ssim'] - gt_ssim
summary['lpips_diff'] = summary['avg_lpips'] - gt_lpips

# 4. Add inverse LPIPS for radar
summary['inv_lpips'] = 1.0 - summary['avg_lpips']

# 5. Save summary
summary.to_csv('summary_with_gt.csv')
print(summary)

# -------------------------------------------------------
# 6. Scatter plot: avg SSIM vs avg LPIPS (with GT star)
plt.figure(figsize=(6,6))
for model, row in summary.iterrows():
    if model == 'PseudoGTresults':
        # Plot GT with a black star and label it as 'GT'
        plt.scatter(row['avg_ssim'], row['avg_lpips'], 
                    c='k', marker='*', s=200, label='GT')
    else:
        # Plot other models with default circle marker
        plt.scatter(row['avg_ssim'], row['avg_lpips'], 
                    label=model)

plt.xlabel('Average SSIM')
plt.ylabel('Average LPIPS')
plt.title('Model Metrics vs GT')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('comparison_scatter_with_gt.png')
plt.close()



# -------------------------------------------------------
# 7. Radar chart: SSIM and (1-LPIPS)
metrics = ['avg_ssim', 'inv_lpips']
labels = [m for m in summary.index if m != 'PseudoGTresults']
N = len(metrics)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
# draw one line per model
for model in labels:
    vals = summary.loc[model, metrics].tolist()
    vals += vals[:1]  
    ax.plot(angles, vals, label=model, linewidth=2)
    ax.fill(angles, vals, alpha=0.15)

# set the category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(['SSIM', '1 - LPIPS'])
ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
ax.set_ylim(0,1)
ax.set_title("Radar Chart: SSIM vs (1 - LPIPS)", y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig('comparison_radar_with_gt.png')
plt.close()
