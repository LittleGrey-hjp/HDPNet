import torch
import torch.nn as nn
from utils import intersect_dicts
import HourglassPvt2_Base


class LocalAttention(nn.Module):
    def __init__(self, dim, window_size):
        super(LocalAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(dim, num_heads=4)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(-1, C, self.window_size * self.window_size).permute(2, 0, 1)
        attn_output, _ = self.attention(x, x, x)  # [16, 2304, 384])
        attn_output = attn_output.permute(1, 2, 0).view(B, H // self.window_size, W // self.window_size, C, self.window_size, self.window_size) # [1, 48, 48, 384, 4, 4])
        attn_output = attn_output.permute(0, 3, 1, 4, 2, 5).contiguous() # ([1, 384, 48, 4, 48, 4])
        attn_output = attn_output.view(B, C, H, W) # [1, 384, 192, 192]
        return attn_output


class ImprovedModule(nn.Module):
    def __init__(self, dim):
        super(ImprovedModule, self).__init__()
        self.local_attn = LocalAttention(dim, window_size=4)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x + self.local_attn(x)
        x = self.relu(self.conv(x))
        return x


class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):  # num_state=384 num_node=16
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):  # x [16,384,16]
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class FIEM(nn.Module):
    def __init__(self, dim_in=32, dim_mid=16, mids=4, img_size=384, up=False):
        """
        dim_in : input channel
        dim_mid ：
        """
        super(FIEM, self).__init__()

        self.normalize = nn.LayerNorm(dim_in)
        self.img_size = img_size
        self.num_s = int(dim_mid)
        self.num_n = (mids * mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(dim_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(dim_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, dim_in, kernel_size=1, bias=False)

        if up:
            self.local_attn = ImprovedModule(dim_in)
        else:
            self.local_attn = nn.Identity()

    def forward(self, x1, x2):
        n, c, h, w = x1.size()
        x2 = self.local_attn(x2)
        x2 = torch.nn.functional.softmax(x2, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x1).view(n, self.num_s, -1)  # 1x1conv downsampling
        x_proj = self.conv_proj(x1)
        x_mask = x_proj * x2

        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)

        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x2.size()[2:])
        out = x1 + (self.conv_extend(x_state))

        return out


class UpSampling2x(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(UpSampling2x, self).__init__()
        temp_chs = out_chs * 4  # for PixelShuffle
        self.up_module = nn.Sequential(
            nn.Conv2d(in_chs, temp_chs, 1, bias=False),
            nn.BatchNorm2d(temp_chs),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)  # (B, C*r*r, H，w) reshape to (B, C, H*r，w*r)
        )

    def forward(self, features):
        return self.up_module(features)


class GroupFusion(nn.Module):
    def __init__(self, in_chs, out_chs, end=False):  # 768, 384
        super(GroupFusion, self).__init__()

        if end:
            tmp_chs = in_chs*2
        else:
            tmp_chs = in_chs
        self.gf1 = nn.Sequential(nn.Conv2d(in_chs * 2, in_chs, 1, bias=False),
                                 nn.BatchNorm2d(in_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(in_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(in_chs),
                                 nn.ReLU(inplace=True)

                                 )

        self.gf2 = nn.Sequential(nn.Conv2d(in_chs * 2, tmp_chs, 1, bias=False),
                                 nn.BatchNorm2d(tmp_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(tmp_chs, tmp_chs, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(tmp_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(tmp_chs, tmp_chs, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(tmp_chs),
                                 nn.ReLU(inplace=True)
                                 )

        self.gf3 = nn.Sequential(nn.Conv2d(in_chs * 2, tmp_chs, 1, bias=False),
                                 nn.BatchNorm2d(tmp_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(tmp_chs, tmp_chs, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(tmp_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(tmp_chs, tmp_chs, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(tmp_chs),
                                 nn.ReLU(inplace=True)
                                 )
        self.up2x_1 = UpSampling2x(tmp_chs, out_chs)
        self.up2x_2 = UpSampling2x(tmp_chs, out_chs)

    def forward(self, f_up1, f_up2, f_down1, f_down2):  # [768,24,24]
        fc1 = torch.cat((f_down1, f_down2), dim=1)
        f_tmp = self.gf1(fc1)

        out1 = self.gf2(torch.cat((f_tmp, f_up1), dim=1))
        out2 = self.gf3(torch.cat((f_tmp, f_up2), dim=1))

        return self.up2x_1(out1), self.up2x_2(out2)  # [384,48,48]


class OutPut(nn.Module):
    def __init__(self, in_chs):
        super(OutPut, self).__init__()
        self.out = nn.Sequential(UpSampling2x(in_chs, in_chs),
                                 nn.Conv2d(in_chs, in_chs, 1, bias=False),
                                 nn.BatchNorm2d(in_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_chs, 1, 1),
                                 nn.Sigmoid()
                                 )

    def forward(self, feat):
        return self.out(feat)


class Model(nn.Module):
    def __init__(self, ckpt_path, img_size=384):
        super(Model, self).__init__()
        # self.encoder = HourglassPvt_Base.Hourglass_vision_transformer_base()
        # self.encoder = HourglassVIT.Hourglass_vision_transformer()
        self.encoder = HourglassPvt2_Base.Hourglass_vision_transformer_base_v2()
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            # msg = self.encoder.load_state_dict(ckpt["model"], strict=False)
            # csd = ckpt['model']  # checkpoint state_dict as FP32
            csd = intersect_dicts(ckpt, self.encoder.state_dict())  # intersect
            msg = self.encoder.load_state_dict(csd, strict=False)  # load

            print("====================================")
            pt_name = ckpt_path.split('/')[-1]
            print(f'Transferred {len(csd)}/{len(self.encoder.state_dict())} items from {pt_name}')

        self.gf1_1 = GroupFusion(320, 128)
        self.gf1_2 = GroupFusion(128, 64)
        self.gf1_3 = GroupFusion(64, 64, end=True)

        # self.gf2_1 = GroupFusion(768, 384)
        self.gf2_2 = GroupFusion(128, 64)
        self.gf2_3 = GroupFusion(64, 64, end=True)

        self.out_F1 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True)
                                    )

        self.out_F2 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True)
                                    )

        self.sam1 = FIEM(128, 64, up=False)
        self.sam2 = FIEM(128, 64, up=False)
        self.conv_f = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, 3, padding=1,  bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True)
                                    )

        self.out1 = OutPut(in_chs=128)
        self.out2 = OutPut(in_chs=128)
        self.out3 = OutPut(in_chs=256)

    def cim_decoder(self, tokens):
        f = []
        size = [96, 96, 48, 48, 24, 24, 24, 24, 48, 48, 96, 96]
        for i in range(len(tokens)):  # [b,576,768] [b,2304,384]
            b, _, c = tokens[i].shape
            f.append(tokens[i].permute(0, 2, 1).view(b, c, size[i], size[i]).contiguous())

        f1_1, f1_2 = self.gf1_1(f[7], f[4], f[5], f[6])

        f2_1, f2_2 = self.gf1_2(f[9], f[8], f1_1, f1_2)
        f2_3, f2_4 = self.gf2_2(f[3], f[2], f1_1, f1_2)

        f3_1, f3_2 = self.gf1_3(f[11], f[10], f2_2, f2_1)
        f3_3, f3_4 = self.gf2_3(f[1], f[0], f2_3, f2_4)

        fout1 = self.out_F1(torch.cat([f3_1, f3_2], dim=1))
        fout2 = self.out_F2(torch.cat([f3_3, f3_4], dim=1))

        return fout1, fout2  # high, low

    def pred_outs(self, gpd_outs):
        return [self.out1(gpd_outs[0]), self.out2(gpd_outs[1]), self.out3(gpd_outs[2])]

    def forward(self, img):
        #
        B, C, H, W = img.size()
        x = self.encoder(img)  # include

        out_high, out_low = self.cim_decoder(x)
        out1 = self.sam1(out_high, out_low)
        out2 = self.sam2(out_low, out_high)
        out_f = self.conv_f(torch.cat([out1, out2], dim=1))

        out = self.pred_outs([out2, out1, out_f])  # low, high, fusion

        return out


if __name__=='__main__':
    from thop import profile
    x = torch.rand(1, 3, 384, 384)
    net = Model(None)
    flops, params = profile(net, (x,))
    print(f"Flops: {flops / 1e9:.4f} GFlops")
    print(f"Params: {params / 1e6:.4f} MParams")
