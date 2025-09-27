import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import wandb
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SRTrainer:
    def __init__(self, model, device="cuda", lr=2e-4, weight_decay=1e-4,
                 max_epochs=21, sample_every_n_epochs=3, num_samples=10, **kwargs):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.sample_every_n_epochs = sample_every_n_epochs
        self.num_samples = num_samples

        # Loss & optimizer
        self.criterion = nn.L1Loss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        # Checkpoint
        self.best_loss = float("inf")
        self.best_state = None
        
        # For sampling
        self.sample_inputs = None
        self.sample_targets = None
        
        # Create output directory
        os.makedirs("outputs", exist_ok=True)
        
        # Initialize wandb
        self._init_wandb()

    def _init_wandb(self):
        """Initialize wandb logging with better error handling"""
        try:
            wandb_api_key = os.getenv('WANDB_API_KEY')
            if wandb_api_key:
                os.environ['WANDB_API_KEY'] = wandb_api_key
            
            project_name = os.getenv('WANDB_PROJECT', 'swinir-super-resolution')
            
            # Try without entity first - let wandb use default user workspace
            config = {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "max_epochs": self.max_epochs,
                "sample_every_n_epochs": self.sample_every_n_epochs,
                "num_samples": self.num_samples
            }
            
            # Initialize with minimal settings first
            wandb.init(
                project=project_name,
                name=f"swinir_x4_{self.max_epochs}epochs",
                config=config,
                # Don't specify entity - let it use default
                entity=None,
                # Add some settings to help with permission issues
                mode="online",  # Ensure online mode
                reinit=True     # Allow reinit if needed
            )
            print(f"[WandB] Successfully initialized project: {project_name}")
            
        except Exception as e:
            print(f"[WandB] Failed to initialize: {str(e)}")
            print("[WandB] Trying offline mode...")
            try:
                # Fallback to offline mode
                wandb.init(
                    project=project_name,
                    name=f"swinir_x4_{self.max_epochs}epochs",
                    config=config,
                    mode="offline"
                )
                print("[WandB] Initialized in offline mode")
            except Exception as e2:
                print(f"[WandB] Offline mode also failed: {str(e2)}")
                print("[WandB] Disabling wandb logging")
                wandb.init(mode="disabled")

    def _prepare_sample_data(self, val_loader):
        """Prepare fixed sample data for visualization"""
        if self.sample_inputs is not None:
            return
            
        print(f"[Sampling] Preparing {self.num_samples} fixed samples...")
        
        with torch.no_grad():
            # Get first batch from validation loader
            for lr_batch, hr_batch in val_loader:
                batch_size = lr_batch.shape[0]
                print(f"[Sampling] Found batch with {batch_size} samples, shapes: LR={lr_batch.shape}, HR={hr_batch.shape}")
                
                # Take up to num_samples from this batch
                num_to_take = min(self.num_samples, batch_size)
                
                self.sample_inputs = lr_batch[:num_to_take].to(self.device)
                self.sample_targets = hr_batch[:num_to_take].to(self.device)
                
                print(f"[Sampling] Successfully prepared {num_to_take} samples")
                print(f"[Sampling] Final shapes: LR={self.sample_inputs.shape}, HR={self.sample_targets.shape}")
                break  # Only take from first batch to avoid size mismatch
        
        if self.sample_inputs is None:
            print("[Sampling] Warning: No samples collected!")
            return

    def sample_images(self, epoch):
        """Generate and save sample images with error handling"""
        if self.sample_inputs is None:
            print("[Sampling] No sample data prepared, skipping...")
            return
            
        print(f"[Sampling] Generating samples for epoch {epoch}...")
        self.model.eval()
        
        with torch.no_grad():
            sample_srs = self.model(self.sample_inputs)
        
        # Create output directory for this epoch
        epoch_dir = f"outputs/epoch_{epoch:03d}"
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Convert tensors to images and save
        wandb_images = []
        for i in range(self.sample_inputs.shape[0]):
            # Convert to numpy arrays [0, 1] -> [0, 255]
            lr_img = self._tensor_to_image(self.sample_inputs[i])
            hr_img = self._tensor_to_image(self.sample_targets[i])
            sr_img = self._tensor_to_image(sample_srs[i])
            
            # Save individual images
            Image.fromarray(lr_img).save(f"{epoch_dir}/img{i+1:02d}_lr.jpg")
            Image.fromarray(hr_img).save(f"{epoch_dir}/img{i+1:02d}_hr.jpg")
            Image.fromarray(sr_img).save(f"{epoch_dir}/img{i+1:02d}_sr.jpg")
            
            # Create comparison for wandb
            lr_pil = Image.fromarray(lr_img)
            hr_pil = Image.fromarray(hr_img) 
            sr_pil = Image.fromarray(sr_img)
            
            # Resize LR to same size as HR for comparison
            hr_size = hr_pil.size
            lr_resized = lr_pil.resize(hr_size, Image.NEAREST)
            
            # Create side-by-side comparison
            combined_width = hr_size[0] * 3
            combined_height = hr_size[1]
            combined = Image.new('RGB', (combined_width, combined_height))
            
            combined.paste(lr_resized, (0, 0))
            combined.paste(sr_pil, (hr_size[0], 0))
            combined.paste(hr_pil, (hr_size[0] * 2, 0))
            
            wandb_images.append(wandb.Image(
                combined,
                caption=f"Sample {i+1}: LR (bicubic) | SR (ours) | HR (ground truth)"
            ))
        
        # Log to wandb with error handling
        try:
            wandb.log({
                "sample_images": wandb_images,
                "epoch": epoch
            })
            print(f"[Sampling] Logged {len(wandb_images)} images to wandb")
        except Exception as e:
            print(f"[WandB] Failed to log images: {str(e)}")
        
        print(f"[Sampling] Saved {len(wandb_images)} sample images to {epoch_dir}")

    def _tensor_to_image(self, tensor):
        """Convert tensor to numpy image array"""
        # tensor: [C, H, W] in range [0, 1]
        tensor = torch.clamp(tensor, 0, 1)
        img_np = tensor.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
        img_np = (img_np * 255).astype(np.uint8)
        return img_np

    def fit(self, train_loader, val_eval):
        """Training loop with progress bars and wandb logging"""
        
        # Prepare sample data from validation loader
        self._prepare_sample_data(val_eval.loader)
        
        print(f"[Training] Starting training for {self.max_epochs} epochs...")
        
        # Epoch progress bar
        epoch_pbar = tqdm(range(1, self.max_epochs + 1), desc="Training", unit="epoch")
        
        for epoch in epoch_pbar:
            self.model.train()
            total_loss = 0.0
            epoch_psnr = 0.0
            epoch_ssim = 0.0
            num_batches = 0
            
            # Training progress bar
            train_pbar = tqdm(
                train_loader, 
                desc=f"Epoch {epoch}/{self.max_epochs}", 
                leave=False,
                unit="batch"
            )
            
            for lr, hr in train_pbar:
                lr, hr = lr.to(self.device), hr.to(self.device)
                sr = self.model(lr)

                loss = self.criterion(sr, hr)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Calculate metrics
                with torch.no_grad():
                    psnr = self.psnr(sr, hr)
                    ssim = self.ssim(sr, hr)

                total_loss += loss.item()
                epoch_psnr += psnr.item()
                epoch_ssim += ssim.item()
                num_batches += 1
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'psnr': f'{psnr.item():.2f}',
                    'ssim': f'{ssim.item():.4f}'
                })

            # Calculate averages
            avg_train_loss = total_loss / num_batches
            avg_train_psnr = epoch_psnr / num_batches
            avg_train_ssim = epoch_ssim / num_batches

            # Validation
            self.model.eval()
            val_metrics = val_eval(self.model)
            
            # Log to wandb with error handling
            try:
                wandb.log({
                    "train_loss": avg_train_loss,
                    "train_psnr": avg_train_psnr,
                    "train_ssim": avg_train_ssim,
                    "val_loss": val_metrics.get('loss', 0),
                    "val_psnr": val_metrics['psnr'],
                    "val_ssim": val_metrics['ssim'],
                    "epoch": epoch
                })
            except Exception as e:
                print(f"[WandB] Failed to log metrics: {str(e)}")

            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'val_psnr': f'{val_metrics["psnr"]:.3f}',
                'val_ssim': f'{val_metrics["ssim"]:.4f}'
            })

            print(f"[Epoch {epoch}] Train: loss={avg_train_loss:.4f}, psnr={avg_train_psnr:.3f}, ssim={avg_train_ssim:.4f}")
            print(f"              Val: psnr={val_metrics['psnr']:.3f}, ssim={val_metrics['ssim']:.4f}")

            # Save best model
            if avg_train_loss < self.best_loss:
                self.best_loss = avg_train_loss
                self.best_state = self.model.state_dict().copy()
                print(f"[Checkpoint] New best model saved (loss: {avg_train_loss:.4f})")

            # Generate samples every N epochs
            if epoch % self.sample_every_n_epochs == 0:
                # Create checkpoints directory
                os.makedirs("/workspace/FIRE/checkpoints", exist_ok=True)
                # save the current model to folder checkpoints
                checkpoint_path = f"/workspace/FIRE/checkpoints/model_epoch_{epoch}.pth"
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"[Checkpoint] Model saved to {checkpoint_path}")
                self.sample_images(epoch)
        
        # Close progress bars
        epoch_pbar.close()
        
        # Final sampling
        if self.max_epochs % self.sample_every_n_epochs != 0:
            self.sample_images(self.max_epochs)
        
        # Finish wandb run
        try:
            wandb.finish()
            print("[WandB] Run finished successfully")
        except Exception as e:
            print(f"[WandB] Error finishing run: {str(e)}")
            
        print("[Training] Training completed!")

    def load_best(self):
        """Restore best checkpoint"""
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            print("[Trainer] Loaded best model weights.")
        else:
            print("[Trainer] No best model state found.")

def build_sr_trainer(**kwargs):
    """Hydra entrypoint"""
    return SRTrainer(**kwargs)