import os
from torchvision.transforms.functional import to_pil_image
import torch
from tqdm import tqdm


class PSNR:
    def __init__(self, max):
        self.max = max

    def __call__(self, mse):
        return 10 * torch.log10(self.max ** 2 / mse)


def Vimeo_PSNR(model, dataloader, valid_path, ep):
    model.eval()
    psnr_total = 0
    recon_loss = 0
    n_samples = 0
    ep = ep + 1  # add 1 to make it intuitive

    cur_valid_path = os.path.join(valid_path, f'{ep}')
    os.makedirs(cur_valid_path, exist_ok=True)
    psnr = PSNR(max=1.)

    for i, data in enumerate(tqdm(dataloader)):
        input_frames, target_frames, target_t, name = data
        input_frames = input_frames.cuda().float()
        target_frames = target_frames.cuda().float()
        target_t = target_t.cuda().float()
        b = target_frames.shape[0]
        n_samples += b

        with torch.no_grad():
            pred = model(input_frames, target_t)

            mse = (pred - target_frames) ** 2
            mse = torch.mean(mse.contiguous().view(b, -1), dim=1)

            psnr_total += psnr(mse).sum()
            recon_loss += mse.sum()

            # save reconstructed image
            for i in range(b):
                to_pil_image(pred[i].cpu()).save(os.path.join(cur_valid_path, f'{name[i]}.png'))

    avg_psnr = psnr_total / n_samples
    avg_recon_loss = recon_loss / n_samples
    print(f'Validation loss at Epoch {ep} === PSNR: {avg_psnr}\tMSE: {avg_recon_loss}')
    model.train()
    return avg_psnr, avg_recon_loss, cur_valid_path

