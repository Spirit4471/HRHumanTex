This project is build based on 
runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04.

So please use a UNIX environment, make sure you have a high-end NVIDIA GPU.

pip install -r requirements.txt

then download pretrained model.

https://drive.google.com/file/d/1XbQZ40yjD931UtpQlHCe6p2dfb-Zc_xc/view?usp=drive_link

in this google drive, there is a .zip file that include all 7 HRHumanTex pretrained texture inpainting model.

https://drive.google.com/file/d/1A9wWjPvlQyAgFOmN7n-qPsDWcExaYSfm/view?usp=sharing
this link is the inpainting results based on SHHQ dataset

https://drive.google.com/file/d/1rm9FQPSePl68rWBCs9vVOWi7927LF4YS/view?usp=sharing
this link is the inpainting results based on DeepFashion dataset

To train your own model:
you could follow the HRHT.ipynb.

Or start a new .ipynb:
set environments:

import os

os.environ["MODEL_NAME"] = "runwayml/stable-diffusion-v1-5"
os.environ["INSTANCE_DIR"] = "./SwinIR1024"
os.environ["CLASS_DIR"] = "./class_dir"
os.environ["OUTPUT_DIR"] = "./sd1.5-uv-1024"


cd scripts

Place your own UV maps in a folder. I used super-resolution enhanced SMPLitex dataset to train these inpainting model.

!accelerate launch --mixed_precision="fp16" train_dreambooth.py   --pretrained_model_name_or_path=$MODEL_NAME    --instance_data_dir=$INSTANCE_DIR   --output_dir=$OUTPUT_DIR   --class_data_dir=$CLASS_DIR   --with_prior_preservation --prior_loss_weight=1.0   --instance_prompt="a sks texturemap"   --class_prompt="a texturemap"   --resolution=1024   --train_batch_size=1   --gradient_accumulation_steps=8 --gradient_checkpointing   --learning_rate=1e-6   --lr_scheduler="constant"   --lr_warmup_steps=0   --num_class_images=10   --max_train_steps=2000   --checkpointing_steps=500   --train_text_encoder   --use_8bit_adam

you could convert your model into .ckpt file format with script:

for example:
!python convert_diffusers_to_original_stable_diffusion.py --model_path $PATH_TO_YOUR_TRAINED_MODEL  --checkpoint_path $OUTPUT_PATH/HRHumanTex_SwinIR.ckpt

Generate partial texturemap:

first install detectron2:
follow this:
https://detectron2.readthedocs.io/en/latest/tutorials/install.html

and run:


import os
os.environ["DETECTRON_PATH"] = "/workspace/HRHumanTex/scripts/detectron2"

!python image_to_densepose.py --detectron2 $DETECTRON_PATH --input_folder ./

After this, install SGHM: https://github.com/cxgincsu/SemanticGuidedHumanMatting

then using their pretrained weight.
run:

!python SemanticGuidedHumanMatting/test_image.py \
	--images-dir ./SHHQ_testdata/images  \
	--result-dir ./SHHQ_testdata/images-seg \
	--pretrained-weight SemanticGuidedHumanMatting/pretrained/SGHM-ResNet50.pth

then you could run:
!python compute_partial_texturemap.py --input_folder ./SHHQ_testdata


After this, you need to install StableDiffusion WebUI, 
https://github.com/AUTOMATIC1111/stable-diffusion-webui
 when you do the inpainting process ,you need to ensure StableDiffusion WebUI is running:


Launch stablediffusion webui:
~/stable-diffusion-webui$ ./webui.sh --disable-safe-unpickle --api

sdwebui install is sometimes complex:
first, you need to using python venv

python3.10 -m venv venv

activate venv

source venv/bin/activate

creat a new user

adduser sduser

Grant directory permissions to new user

chown -R sduser:sduser /workspace/SMPLitex

switch to new user

su - sduser

Activate the virtual environment and run the script:
cd /workspace/SMPLitex/scripts/stable-diffusion-webui
source venv/bin/activate
./webui.sh --disable-safe-unpickle --api

By following these steps you should be able to circumvent the limitation of running the webui.sh script as root and run the script successfully

but also, you need to change the webui.py 
find shared.demo.launch() function, and using share=True parameter, add server_name="0.0.0.0" .

then running:
!python inpaint_with_A1111.py --partial_textures ./SHHQ_testdata/uv-textures --masks ./SHHQ_testdata/uv-textures-masks --inpainted_textures ./SHHQ_testdata/inpaintedStableSR


finally, you could rendering the results:
!python render_results.py --textures ./SHHQ_testdata/inpaintedSMPLitex/


For evaluation: detaild in the evaluation.ipynb.

if you want to evaluation super-resolution method:
running example:
!python evaluate_sr.py \
  --gt_dir /workspace/HRHumanTex/scripts/SHHQ_testdata/PGT \
  --sr_dirs \
    /workspace/HRHumanTex/scripts/SHHQ_testdata/inpaintedBSRGAN \
    /workspace/HRHumanTex/scripts/SHHQ_testdata/inpaintedRealESRGAN \
    /workspace/HRHumanTex/scripts/SHHQ_testdata/inpaintedSwinIR \
    /workspace/HRHumanTex/scripts/SHHQ_testdata/inpaintedSRFormer \
    /workspace/HRHumanTex/scripts/SHHQ_testdata/inpaintedstablesr \
    /workspace/HRHumanTex/scripts/SHHQ_testdata/inpaintedDiffBIR \
    /workspace/HRHumanTex/scripts/SHHQ_testdata/inpaintedRCAN \
  --output_csv /workspace/HRHumanTex/scripts/SHHQ_testdata/metrics1024inpainting.csv


if you want to do rendering-based evaluation:
!python evaluate_texture_batch.py \
  --gt_folder /workspace/HRHumanTex/scripts/SHHQ_testdata/images \
  --texture_folder /workspace/HRHumanTex/scripts/SHHQ_testdata/inpaintedSwinIR \
  --npz_folder /workspace/HRHumanTex/scripts/SHHQ_testdata/pose_output_npz \
  --dp_folder /workspace/HRHumanTex/scripts/SHHQ_testdata/densepose \
  --sil_folder /workspace/HRHumanTex/scripts/SHHQ_testdata/images-seg \
  --save_folder /workspace/HRHumanTex/scripts/SHHQ_testdata/SwinIRrender \
  --csv_path /workspace/HRHumanTex/scripts/SHHQ_testdata/SwinIRRresults.csv



if you want to do rendering-based texture optimization:
!python optimize_uv_texture.py \
  --uv_template /workspace/HRHumanTex/sample-data/smpl_uv_20200910/smpl_uv.obj \
  --initial_texture /workspace/HRHumanTex/scripts/test_data/inpaintedDiffBIR/MEN-Shirts_Polos-id_00003706-01_1_front_texture_inpaint-003_img2img-000_cfg2.0_05042025-043616.png \
  --gt_image /workspace/HRHumanTex/scripts/test_data/ground_truth/MEN-Shirts_Polos-id_00003706-01_1_front.jpg \
  --npz_file /workspace/HRHumanTex/scripts/test_data/pose_output_npz/00003706.npz \
  --smpl_model_dir /workspace/HRHumanTex/sample-data/SMPL/models/smpl/SMPL_NEUTRAL.pkl \
  --out_dir /workspace/HRHumanTex/scripts/test_data/results \
  --lr 1e-2



running this scripts:
!python compare_results.py 

you could get all the rendering-based evaluation method in average.
for example:
avg_ssim  avg_lpips  ssim_diff  lpips_diff  inv_lpips
model                                                                   
BSRGANresults      0.755815   0.282977   0.000133   -0.000209   0.717023
DiffBIRresults     0.756864   0.283242   0.001182    0.000057   0.716758
PseudoGTresults    0.755682   0.283185   0.000000    0.000000   0.716815
RCANresults        0.755768   0.282734   0.000086   -0.000451   0.717266
RealESRGANresults  0.756765   0.283324   0.001083    0.000139   0.716676
StableSRresults    0.755864   0.282775   0.000182   -0.000410   0.717225
SwinIRresults      0.756157   0.282935   0.000475   -0.000250   0.717065




7 super resolution method I used in my thesis:
SwinIR and BSRGAN, I used Google Colab:
detailed in
swinir-demo-on-real-world-image-sr.ipynb

RealESRGAN:
detailed in
Copy of Real-ESRGAN Inference Demo.ipynb


I highly recommended using conda:

For DiffBIR: https://github.com/XPixelGroup/DiffBIR
follow their instruction
for example:
python -u inference.py \
--task sr \
--upscale 2 \
--version v2.1 \
--captioner none \
--cfg_scale 8 \
--noise_aug 0 \
--input /workspace/DiffBIR/input256 \
--output /workspace/DiffBIR/output512

StableSR: https://github.com/IceClear/StableSR
for example:

python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py \
  --config configs/stableSRNew/v2-finetune_text_T_512.yaml \
  --ckpt ./stablesr_000117.ckpt \
  --vqgan_ckpt ./vqgan_cfw_00011.ckpt \
  --init-img ./inputs/512img.png \
  --outdir ./results_up1024/ \
  --ddpm_steps 200 \
  --dec_w 0.5 \
  --colorfix_type wavelet \
  --upscale 2


SRFormer: https://github.com/HVision-NKU/SRFormer
for example:
PYTHONPATH=. python basicsr/infer_sr.py \
-opt options/test/SRFormer/test_SRFormer_DF2Ksrx2.yml \
--input_dir /workspace/SRFormer/512inputshhq \
--output_dir /workspace/SRFormer/1024outputshhq


RCAN:https://github.com/yulunzhang/RCAN

Because this RCAN repository is relative old, you need to modify the source code to running in modern GPU environment:
modify the RCAN dataloader:
find this line:
from torch._C import _set_worker_signal_handlers, _update_worker_pids, _remove_worker_pids
change to:
from torch._C import _set_worker_signal_handlers

because _update_worker_pids and _remove_worker_pid have been removed in new version PyTorch

after that you could running :
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

in the rcan conda environment, after this step, RCAN could running at CUDA11.1+ platform.


