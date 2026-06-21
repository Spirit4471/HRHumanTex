# dreambooth_trainer.py
import math, os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from accelerate import Accelerator
from diffusers import DiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import is_xformers_available

class DreamBoothTrainer:
    def __init__(self, args, unet, text_encoder, vae, tokenizer, noise_scheduler, optimizer, lr_scheduler, train_dataloader):
        self.args = args
        self.accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision)
        self.unet = unet
        self.text_encoder = text_encoder
        self.vae = vae
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.global_step = 0

    def setup(self):
        # xformers
        if is_xformers_available():
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print("Warning: xFormers not working:", e)

        # freeze
        self.vae.requires_grad_(False)
        if not self.args.train_text_encoder:
            self.text_encoder.requires_grad_(False)

        # gradient checkpoint
        if self.args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.args.train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

        # accelerate prepare
        if self.args.train_text_encoder:
            (self.unet, self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler) = self.accelerator.prepare(
                self.unet, self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler
            )
        else:
            (self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler) = self.accelerator.prepare(
                self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
            )

        # cast vae & text_encoder to half if needed
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        
        self.vae.to(self.accelerator.device, dtype=weight_dtype)
        if not self.args.train_text_encoder:
            self.text_encoder.to(self.accelerator.device, dtype=weight_dtype)

        # print info
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * self.num_update_steps_per_epoch
        
        print(f">> Setup complete. total steps: {self.args.max_train_steps}")

    def train_loop(self):
        self.setup()
        progress_bar = tqdm(range(self.args.max_train_steps), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Training steps")

        for epoch in range(self.args.num_train_epochs):
            self.unet.train()
            if self.args.train_text_encoder:
                self.text_encoder.train()
            
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    latents = self.vae.encode(batch["pixel_values"].to(dtype=self.accelerator.device_dtype)).latent_dist.sample()
                    latents = latents * 0.18215  # scaling

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device, dtype=torch.long)
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
                    model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

                    if self.args.with_prior_preservation:
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                        loss = loss + self.args.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), self.args.max_grad_norm)
                        if self.args.train_text_encoder:
                            self.accelerator.clip_grad_norm_(self.text_encoder.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.global_step += 1

                    if self.global_step % self.args.checkpointing_steps == 0 and self.accelerator.is_main_process:
                        ckpt_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
                        self.accelerator.save_state(ckpt_dir)
                        print(f"** Saved checkpoint at {ckpt_dir}")

                logs = {"loss": loss.item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if self.global_step >= self.args.max_train_steps:
                    break

            if self.global_step >= self.args.max_train_steps:
                break

        # final save
        if self.accelerator.is_main_process:
            pipe = DiffusionPipeline.from_pretrained(
                self.args.pretrained_model_name_or_path,
                unet=self.accelerator.unwrap_model(self.unet),
                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                revision=self.args.revision
            )
            pipe.save_pretrained(self.args.output_dir)
            print("** Final pipeline saved.")

        print("Training Done!")
        self.accelerator.end_training()
