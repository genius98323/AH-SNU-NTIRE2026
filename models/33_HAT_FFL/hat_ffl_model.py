import torch
import numpy as np
from hat.archs.hat_arch import HAT

class Model:
    def __init__(self, model_path, device):
        self.device = device

        self.model = HAT(
            upscale=4, in_chans=3, img_size=64, window_size=16,
            compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
            overlap_ratio=0.5, img_range=1.,
            depths=[6,6,6,6,6,6], embed_dim=180,
            num_heads=[6,6,6,6,6,6], mlp_ratio=2,
            upsampler='pixelshuffle', resi_connection='1conv'
        ).to(device)

        ckpt = torch.load(model_path, map_location=device)
        state = ckpt.get('params_ema') or ckpt.get('params') or ckpt
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.no_grad()
    def process(self, img):
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).to(self.device)

        out = self.model(img)

        out = out.squeeze().permute(1,2,0).cpu().numpy()
        out = np.clip(out, 0, 1)
        out = (out * 255.0).round().astype(np.uint8)

        return out
