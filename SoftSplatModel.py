import torch
import torch.nn as nn
from OpticalFlow.PWCNet import PWCNet
from softsplat import Softsplat
from torch.nn.functional import interpolate, grid_sample
from einops import repeat


# convert [0, 1] to [-1, 1]
def preprocess(x):
    return x * 2 - 1


# convert [-1, 1] to [0, 1]
def postprocess(x):
    return torch.clamp((x + 1) / 2, 0, 1)


class BackWarp(nn.Module):
    def __init__(self, clip=True):
        super(BackWarp, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip = clip

    def forward(self, img, flow):
        b, c, h, w = img.shape
        gridY, gridX = torch.meshgrid(torch.arange(h), torch.arange(w))
        gridX, gridY = gridX.to(self.device), gridY.to(self.device)

        u = flow[:, 0]  # W
        v = flow[:, 1]  # H

        x = repeat(gridX, 'h w -> b h w', b=b).float() + u
        y = repeat(gridY, 'h w -> b h w', b=b).float() + v

        # normalize
        x = (x / w) * 2 - 1
        y = (y / h) * 2 - 1

        # stacking X and Y
        grid = torch.stack((x, y), dim=-1)

        # Sample pixels using bilinear interpolation.
        if self.clip:
            output = grid_sample(img, grid, mode='bilinear', align_corners=True, padding_mode='border')
        else:
            output = grid_sample(img, grid, mode='bilinear', align_corners=True)
        return output


class SoftSplatBaseline(nn.Module):
    def __init__(self):
        super(SoftSplatBaseline, self).__init__()
        self.flow_predictor = PWCNet()
        self.flow_predictor.load_state_dict(torch.load('./OpticalFlow/pwc-checkpoint.pt'))
        self.fwarp = Softsplat()
        self.bwarp = BackWarp(clip=False)
        self.feature_pyramid = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1),
                nn.PReLU(),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.PReLU()
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, 2, 1),
                nn.PReLU(),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.PReLU()
            ),
            nn.Sequential(
                nn.Conv2d(64, 96, 3, 2, 1),
                nn.PReLU(),
                nn.Conv2d(96, 96, 3, 1, 1),
                nn.PReLU()
            ),
        ])
        self.synth_net = GridNet()
        self.alpha = nn.Parameter(torch.randn(1))

    def forward(self, x, target_t):
        x = preprocess(x)
        b = x.shape[0]
        target_t = target_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fr0, fr1 = x[:, :, 0], x[:, :, 1]
        flow = self.flow_predictor(torch.cat([fr0, fr1], dim=0), torch.cat([fr1, fr0], dim=0))
        f_lv = torch.cat([fr0, fr1], dim=0)
        pyramid = []
        for feat_extractor_lv in self.feature_pyramid:
            f_lv = feat_extractor_lv(f_lv)
            pyramid.append(f_lv)

        # Z importance metric
        with torch.no_grad():
            brightness_diff = torch.sum(torch.abs(self.bwarp(torch.cat([fr1, fr0], dim=0), flow) - torch.cat([fr0, fr1], dim=0)), dim=1, keepdim=True)
        z = self.alpha * brightness_diff

        # warping
        n_lv = len(pyramid)
        warped_feat_pyramid = []
        for lv in range(n_lv):
            flow_lv = interpolate(flow, scale_factor=0.5 ** lv, mode='bilinear', align_corners=False) * (0.5 ** lv)
            flow01, flow10 = torch.split(flow_lv, b, dim=0)
            flow0t, flow1t = flow01 * target_t, flow10 * (1 - target_t)
            flowt = torch.cat([flow0t, flow1t], dim=0)
            f_lv = pyramid[lv]
            z_lv = interpolate(z, scale_factor=0.5 ** lv, mode='bilinear', align_corners=False)
            warped_f_lv = self.fwarp(f_lv, flowt, z_lv)
            warped_feat_pyramid.append(warped_f_lv)

        concat_warped_feat_pyramid = []
        for feat_lv in warped_feat_pyramid:
            feat0_lv, feat1_lv = torch.split(feat_lv, b, dim=0)
            feat_lv = torch.cat([feat0_lv, feat1_lv], dim=1)
            concat_warped_feat_pyramid.append(feat_lv)
        output = self.synth_net(concat_warped_feat_pyramid)
        return postprocess(output)


class GridNet(nn.Module):
    def __init__(self):
        super(GridNet, self).__init__()
        self.lateral = nn.ModuleList([
            nn.ModuleList([
                LateralBlock(32),
                LateralBlock(32),
                LateralBlock(32),
                LateralBlock(32),
                LateralBlock(32),
                LateralBlock(32),
            ]),
            nn.ModuleList([
                LateralBlock(64),
                LateralBlock(64),
                LateralBlock(64),
                LateralBlock(64),
                LateralBlock(64),
            ]),
            nn.ModuleList([
                LateralBlock(96),
                LateralBlock(96),
                LateralBlock(96),
                LateralBlock(96),
                LateralBlock(96),
            ])
        ])
        self.down = nn.ModuleList([
            nn.ModuleList([
                DownBlock(32, 64),
                DownBlock(32, 64),
                DownBlock(32, 64),
            ]),
            nn.ModuleList([
                DownBlock(64, 96),
                DownBlock(64, 96),
                DownBlock(64, 96),
            ])
        ])
        self.up = nn.ModuleList([
            nn.ModuleList([
                UpBlock(64, 32),
                UpBlock(64, 32),
                UpBlock(64, 32),
            ]),
            nn.ModuleList([
                UpBlock(96, 64),
                UpBlock(96, 64),
                UpBlock(96, 64),
            ])
        ])
        self.compress = nn.ModuleList([
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.Conv2d(192, 96, 3, 1, 1)
        ])
        self.to_RGB = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def forward(self, pyramid):
        x_0_0, x_1_0, x_2_0 = pyramid

        # compress dim 32 x 2 -> 32
        x_0_0 = self.compress[0](x_0_0)
        x_1_0 = self.compress[1](x_1_0)
        x_2_0 = self.compress[2](x_2_0)

        # first half: down & lateral
        x_0_1 = self.lateral[0][0](x_0_0)
        x_0_2 = self.lateral[0][1](x_0_1)
        x_0_3 = self.lateral[0][2](x_0_2)

        x_1_0 = x_1_0 + self.down[0][0](x_0_0)
        x_2_0 = x_2_0 + self.down[1][0](x_1_0)

        x_1_1 = self.lateral[1][0](x_1_0)
        x_2_1 = self.lateral[2][0](x_2_0)

        x_1_1 = x_1_1 + self.down[0][1](x_0_1)
        x_2_1 = x_2_1 + self.down[1][1](x_1_1)

        x_1_2 = self.lateral[1][1](x_1_1)
        x_2_2 = self.lateral[2][1](x_2_1)

        x_1_2 = x_1_2 + self.down[0][2](x_0_2)
        x_2_2 = x_2_2 + self.down[1][2](x_1_2)

        x_1_3 = self.lateral[1][2](x_1_2)
        x_2_3 = self.lateral[2][2](x_2_2)

        # second half: up & lateral
        x_2_4 = self.lateral[2][3](x_2_3)
        x_2_5 = self.lateral[2][4](x_2_4)

        x_1_3 = x_1_3 + self.up[1][0](x_2_3)
        x_0_3 = x_0_3 + self.up[0][0](x_1_3)

        x_1_4 = self.lateral[1][3](x_1_3)
        x_0_4 = self.lateral[0][3](x_0_3)

        x_1_4 = x_1_4 + self.up[1][1](x_2_4)
        x_0_4 = x_0_4 + self.up[0][1](x_1_4)

        x_1_5 = self.lateral[1][4](x_1_4)
        x_0_5 = self.lateral[0][4](x_0_4)

        x_1_5 = x_1_5 + self.up[1][2](x_2_5)
        x_0_5 = x_0_5 + self.up[0][2](x_1_5)

        # final synthesis
        output = self.lateral[0][5](x_0_5)
        output = self.to_RGB(output)
        return output


class LateralBlock(nn.Module):
    def __init__(self, dim):
        super(LateralBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )

    def forward(self, x):
        res = x
        x = self.layers(x)
        return x + res


class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DownBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_dim, out_dim, 3, 2, 1),
            nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        )

    def forward(self, x):
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(UpBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        )

    def forward(self, x):
        x = interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        return self.layers(x)


if __name__ == '__main__':
    '''
    Example Usage
    '''
    frame0frame1 = torch.randn([1, 3, 2, 448, 256]).cuda()  # batch size 1, 3 RGB channels, 2 frame input, H x W of 448 x 256
    target_t = torch.tensor([0.5]).cuda()
    model = SoftSplatBaseline().cuda()
    model.load_state_dict(torch.load('./ckpt/SoftSplatBaseline_Vimeo.pth'))

    with torch.no_grad():
        output = model(frame0frame1, target_t)
