# SoftSplat-Full
Full model implementation of CVPR 2020 paper, ["Softmax Splatting for Video Frame Interpolation"](https://openaccess.thecvf.com/content_CVPR_2020/papers/Niklaus_Softmax_Splatting_for_Video_Frame_Interpolation_CVPR_2020_paper.pdf)

The full model's weights pretrained on Vimeo90k data can be downloaded from this [link](https://drive.google.com/file/d/1DKgZtbRmAycDDOk16IUT1NCKz_TQhcGF/view?usp=sharing).

My reproduced model shows a ```PSNR of 35.64``` in Vimeo90k.

Note that the original implementation of the paper is trained with the Laplacian pyramids, but I used the conventional pixel-wise L1 loss instead.

The model assumes to take a tensor in the shape of [B, C, T, H, W] where B is the batch size, C is the number of channels (3 RGB channels), T is the number of frames (2 frames: frame 0 & frame 1), H and W is the height and width of the frames.
```python
from SoftSplatModel import SoftSplatBaseline

frame0frame1 = torch.randn([1, 3, 2, 448, 256]).cuda()  # batch size 1, 3 RGB channels, 2 frame input, H x W of 448 x 256
target_t = torch.tensor([0.5]).cuda()  # target t=0.5

model = SoftSplatBaseline().cuda()
model.load_state_dict(torch.load('./SoftSplatBaseline_Vimeo.pth'))

framet = model(frame0frame1, target_t)
```
