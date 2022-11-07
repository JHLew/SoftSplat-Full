import torch
from tqdm import tqdm
from lpips import LPIPS
from DISTS_pytorch import DISTS
from pytorch_msssim import SSIM, MS_SSIM


class Evaluate:
    def __init__(self) -> None:
        '''
        Evaluation with 5 different metrics: PSNR, SSIM, MS-SSIM, LPIPS, DISTS
        input must be in the range of [0, 1]
        '''
        self.get_psnr = PSNR()
        self.get_ssim = SSIM(size_average=False, channel=1, nonnegative_ssim=True)
        self.get_msssim = MS_SSIM(size_average=False, channel=1)
        self.get_lpips = LPIPS().to('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_dists = DISTS().to('cuda' if torch.cuda.is_available() else 'cpu')

        self.scaler = 255.0
        self.filter = torch.tensor([[65.481], [128.553], [24.966]]) / 255.
        self.b = 16.0

    def to_uint8(self, img):
        img = img * self.scaler
        img = torch.clamp(img.round(), 0, 255)
        return img

    # https://stackoverflow.com/questions/17892346/how-to-convert-rgb-yuv-rgb-both-ways
    def get_Y(self, img):
        img = img.permute(0, 2, 3, 1)  # b c h w -> b h w c
        y = torch.matmul(img, self.filter.to(img.device)) + self.b
        y = y.permute(0, 3, 1, 2)  #  b h w c -> b c h w
        return torch.clamp(y, 0, 255)

    def __call__(self, pred, gt):
        scores = dict()
        with torch.no_grad():
            pred_uint8, gt_uint8 = self.to_uint8(pred), self.to_uint8(gt)
            pred_y, gt_y = self.get_Y(pred_uint8), self.get_Y(gt_uint8)
            scores['psnr'] = self.get_psnr(pred_uint8, gt_uint8)
            scores['ssim'] = self.get_ssim(pred_y, gt_y)
            scores['ms_ssim'] = self.get_msssim(pred_y, gt_y)
            scores['lpips'] = self.get_lpips(pred, gt, normalize=True)
            scores['dists'] = self.get_dists(pred, gt)
        return scores


class PSNR:
    def __call__(self, pred, gt):
        b = gt.shape[0]
        se = (pred - gt) ** 2
        mse = torch.mean(se.view(b, -1), dim=1)
        return 10 * torch.log10((255. ** 2) / mse)


def validation(model, dataloader, writer, ep):
    model.eval()
    scores = dict()
    scores['psnr'] = 0
    scores['ssim'] = 0
    scores['ms_ssim'] = 0
    scores['lpips'] = 0
    scores['dists'] = 0
    n_samples = 0

    get_scores = Evaluate()
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            input_frames, target_frames, target_t, name = data
            input_frames = input_frames.cuda().float()
            target_frames = target_frames.cuda().float()
            target_t = target_t.cuda().float()
            n_samples += target_frames.shape[0]

            pred = model(input_frames, target_t)
            current_scores = get_scores(pred, target_frames)
            for key, value in current_scores.items():
                scores[key] += value.sum()

            # save reconstructed image
            if i < 10:
                writer.add_images(f'Validation Batch {i:04d} Pred', pred, ep)
                if ep == 0:
                    writer.add_images(f'Validation Batch {i:04d} GT/Inputs', input_frames.sum(dim=2) * 0.5, 0)
                    writer.add_images(f'Validation Batch {i:04d} GT', target_frames, 0)
    torch.cuda.empty_cache()
    for key, value in scores.items():
        scores[key] = value / n_samples

    writer.add_scalar('Eval/PSNR', scores['psnr'], ep)
    writer.add_scalar('Eval/SSIM', scores['ssim'], ep)
    writer.add_scalar('Eval/MS-SSIM', scores['ms_ssim'], ep)
    writer.add_scalar('Eval/LPIPS', scores['lpips'], ep)
    writer.add_scalar('Eval/DISTS', scores['dists'], ep)
    
    print(f"Validation at Epoch {ep} === PSNR: {scores['psnr'].item():.2f}\tSSIM: {scores['ssim'].item():.4f}\tMS-SSIM: {scores['ms_ssim'].item():.4f}\nLPIPS: {scores['lpips'].item():.4f}\tDISTS: {scores['dists'].item():.4f}")
    print('=======================================\n\n')
    model.train()
    return scores

