# SoftSplat-Full
Full model implementation of CVPR 2020 paper, ["Softmax Splatting for Video Frame Interpolation"](https://openaccess.thecvf.com/content_CVPR_2020/papers/Niklaus_Softmax_Splatting_for_Video_Frame_Interpolation_CVPR_2020_paper.pdf)

<!--
The full model's weights pretrained on Vimeo90k data can be downloaded from this
[link](https://drive.google.com/file/d/1wtUFS68D8hVKRg-LFr7jAibg8KgvyrMZ/view?usp=sharing).
My reproduced model shows a ```PSNR of 35.59``` in Vimeo90k.
-->

Pre-trained weights for pre-defined Z [using alpha](https://drive.google.com/file/d/1-pFZZu8Fc5AN9JdKGns1ht2gE0ugsRkf/view?usp=sharing).

Pre-trained weights for fine-tuned Z [using a small neural network v](https://drive.google.com/file/d/1_x_3CxY1_f83spqHt5s-ZwsEtNCTULLy/view?usp=sharing).

Both weights approximately lead to a ``PSNR of 35.8`` in Vimeo90k. (reported score from the paper is 36.10)

The model assumes to take a tensor in the shape of [B, C, T, H, W] where B is the batch size, C is the number of channels (3 RGB channels), T is the number of frames (2 frames: frame 0 & frame 1), H and W is the height and width of the frames.
```python
from SoftSplatModel import SoftSplatBaseline

frame0frame1 = torch.randn([1, 3, 2, 448, 256]).cuda()  # batch size 1, 3 RGB channels, 2 frame input, H x W of 448 x 256
target_t = torch.tensor([0.5]).cuda()  # target t=0.5

model = SoftSplatBaseline().cuda()
model.load_state_dict(torch.load('./SoftSplat_predefinedZ.pth')['model'])

framet = model(frame0frame1, target_t)
```
