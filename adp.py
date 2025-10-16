import torch

def project(a, b):
    dtype = a.dtype
    a, b = a.double(), b.double()
    b = torch.nn.functional.normalize(b, dim=[-1, -2, -3])
    a_par = (a * b).sum(dim=[-1, -2, -3], keepdim=True) * b
    a_orth = a - a_par
    return a_par.to(dtype), a_orth.to(dtype)


class AdaptiveProjectedGuidanceFunction:
    def __init__(self, momentum, eta, norm_threshold, adaptive_momentum=0, mode="normal"):
        self.eta = eta
        self.norm_threshold = norm_threshold
        self.current_step = 999.0
        self.init_momentum = momentum
        self.momentum = momentum
        self.running_average = 0.0
        self.mode = mode
        self.adaptive_momentum = adaptive_momentum

    def __call__(self, args):
        if "denoised" == self.mode:
            cond = args["cond_denoised"]
            uncond = args["uncond_denoised"]
        else:
            cond = args["cond"]
            uncond = args["uncond"]
        cfg_scale = args["cond_scale"]
        sigma = args["sigma"][0].item()
        step = args["model"].model_sampling.timestep(args["sigma"])[0].item()
        x_orig = args["input"]
        if self.mode == "vpred":
            sigma = step
            x = x_orig / (sigma * sigma + 1.0)
            cond = ((x - (x_orig - cond)) * (sigma**2 + 1.0) ** 0.5) / (sigma)
            uncond = ((x - (x_orig - uncond)) * (sigma**2 + 1.0) ** 0.5) / (sigma)

        if self.current_step < step:
            self.current_step = 999.0
            self.running_average = 0.0
            self.momentum = self.init_momentum
        else:
            scale = self.init_momentum
            if self.adaptive_momentum > 0:
                scale -= scale * (self.adaptive_momentum**4) * (1000 - step)
                if self.init_momentum < 0 and scale > 0:
                    scale = 0
                elif self.init_momentum > 0 and scale < 0:
                    scale = 0
                self.momentum = scale

        self.current_step = step

        diff = cond - uncond

        new_average = self.momentum * self.running_average
        self.running_average = diff + new_average
        diff = self.running_average

        if self.norm_threshold > 0.0:
            diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
            scale_factor = torch.minimum(torch.ones_like(diff), self.norm_threshold / diff_norm)
            diff = diff * scale_factor

        diff_parallel, diff_orthogonal = project(diff, cond)

        pred = cond + (cfg_scale - 1) * (diff_orthogonal + self.eta * diff_parallel)
        if "denoised" == self.mode:
            pred = x_orig - pred
        elif "vpred" == self.mode:
            pred = x_orig - (x - pred * sigma / (sigma * sigma + 1.0) ** 0.5)
        return pred


class AdaptiveProjectedGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"model": ("MODEL",)},
            "optional": {
                "momentum": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.01}),
                "eta": ("FLOAT", {"default": 1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "norm_threshold": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "mode": (["normal", "denoised", "vpred"],),
                "adaptive_momentum": ("FLOAT", {"default": 0.18, "min": 0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    CATEGORY = "_for_testing"

    def apply(self, model, momentum=0.5, eta=1.0, norm_threshold=15.0, mode="normal", adaptive_momentum=0.18):
        fn = AdaptiveProjectedGuidanceFunction(momentum, eta, norm_threshold, adaptive_momentum, mode)
        m = model.clone()
        m.set_model_sampler_cfg_function(fn)
        return (m,)