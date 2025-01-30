import math
import torch
import torch.nn as nn
from torcheval.metrics import PeakSignalNoiseRatio
from pytorch_msssim import ms_ssim
from compressai.registry import register_criterion


@register_criterion("RateDistortionLoss")
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, metric="mse", data_range=255.0, return_type="all", device=None):
        super().__init__()

        self.data_range = data_range
        self.device = device or torch.device("cpu")
        
        # Configurazione della metrica
        if metric == "mse":
            self.metric = nn.MSELoss()
        elif metric == "ms-ssim":
            self.metric = ms_ssim
        elif metric == "psnr":
            self.metric = PeakSignalNoiseRatio(data_range=self.data_range).to(self.device)
        else:
            raise NotImplementedError(f"{metric} is not implemented!")

        self.lmbda = lmbda
        self.return_type = return_type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        # Calcolo del bitrate (BPP loss)
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        # Calcolo della distorsione in base alla metrica
        if self.metric == ms_ssim:
            out["ms_ssim_loss"] = self.metric(output["x_hat"], target, data_range=self.data_range)
            distortion = 1 - out["ms_ssim_loss"]
        elif isinstance(self.metric, nn.MSELoss):
            out["mse_loss"] = self.metric(output["x_hat"], target)
            distortion = self.data_range**2 * out["mse_loss"]  # Conversione per PSNR
        elif isinstance(self.metric, PeakSignalNoiseRatio):
            self.metric.update(output["x_hat"], target)
            psnr_value = self.metric.compute()  # Calcola PSNR
            self.metric.reset()  # Resetta per il batch successivo
            distortion = -psnr_value  # PSNR è inversamente proporzionale alla distorsione
            out["psnr"] = psnr_value

        # Calcolo della perdita complessiva
        out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        

        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]
