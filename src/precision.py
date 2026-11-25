"""CUDA mixed precision and scaled optimizer updates."""


def autocast_dtype(device):
    if device.type != "cuda":
        return None

    import torch

    with torch.cuda.device(device):
        if torch.cuda.is_bf16_supported(including_emulation=False):
            return torch.bfloat16
    return torch.float16


def optimizer_step(loss, optimizer, scheduler, parameters, scaler):
    import torch

    optimizer.zero_grad(set_to_none=True)
    if scaler is None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()
        return

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() >= previous_scale:
        scheduler.step()
