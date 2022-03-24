# SoftSplat-Full
Full model implementation of CVPR 2020 paper, "Softmax Splatting for Video Frame Interpolation"

The full model's weights pretrained on Vimeo90k data can be downloaded from this [link](https://drive.google.com/file/d/1DKgZtbRmAycDDOk16IUT1NCKz_TQhcGF/view?usp=sharing).

My reproduced model shows a PSNR of 35.64 in Vimeo90k.

Note that the original implementation of the paper is trained with the Laplacian pyramids, but I used the conventional pixel-wise L1 loss instead.

The model assumes to take a tensor in the shape of [B, C, T, H, W] where B is the batch size, C is the number of channels (3 RGB channels), T is the number of frames (2 frames: frame 0 & frame 1), H and W is the height and width of the frames.
