import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from hydra.utils import instantiate

@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    # 1) Loader
    loader = instantiate(cfg.loader)
    train_loader, val_loader, test_loader = loader.make()
    
    if cfg.verbose:
        print(f"[1] Data successfully loaded.")
        print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
        for lr, hr in train_loader:
            print(f"LR: {lr.shape} | HR: {hr.shape}")
            break

    # 2) Model
    model = instantiate(cfg.model, scale=loader.scale)

    if cfg.verbose:
        print(f"[2] Model successfully built.")
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {model.__class__.__name__} | #Params: {n_params}")

    # 3) Evaluators
    val_eval = instantiate(cfg.eval, loader=val_loader, device=device)
    test_eval = instantiate(cfg.eval, loader=test_loader, device=device)
    val_eval.set_model(model)
    test_eval.set_model(model)

    if cfg.verbose:
        print(f"[3] Evaluator successfully built.")

    # 4) Trainer
    trainer = instantiate(cfg.trainer, model=model, device=device)
    trainer.fit(train_loader, val_eval)
    trainer.load_best()

    if cfg.verbose:
        print(f"[4] Trainer successfully built and trained.")

    # 5) Final evaluation
    metrics = test_eval(model)
    print(f"Test PSNR={metrics['psnr']:.3f} SSIM={metrics['ssim']:.4f}")
    test_eval.show_samples(model)

    if cfg.verbose:
        print(f"[5] Final evaluation done.")

if __name__ == "__main__":
    main()