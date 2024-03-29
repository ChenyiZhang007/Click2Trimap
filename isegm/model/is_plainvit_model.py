import math
import torch.nn as nn
from isegm.utils.serialization import serialize
from .is_model import ISModel
from .modeling.models_vit import VisionTransformer, PatchEmbed
from .modeling.swin_transformer import SwinTransfomerSegHead
from networks import decoders
from networks.encoders.MatteFormer import MatteFormer
import os
import torch
import torch.nn as nn
from utils import CONFIG
import utils
from networks.ops import SpectralNorm

class SimpleFPN(nn.Module):
    def __init__(self, in_dim=768, out_dims=[128, 256, 512, 1024]):
        super().__init__()
        self.down_4_chan = max(out_dims[0]*2, in_dim // 2)
        self.down_4 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_4_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan),
            nn.GELU(),
            nn.ConvTranspose2d(self.down_4_chan, self.down_4_chan // 2, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan // 2),
            nn.Conv2d(self.down_4_chan // 2, out_dims[0], 1),
            nn.GroupNorm(1, out_dims[0]),
            nn.GELU()
        )
        self.down_8_chan = max(out_dims[1], in_dim // 2)
        self.down_8 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_8_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_8_chan),
            nn.Conv2d(self.down_8_chan, out_dims[1], 1),
            nn.GroupNorm(1, out_dims[1]),
            nn.GELU()
        )
        self.down_16 = nn.Sequential(
            nn.Conv2d(in_dim, out_dims[2], 1),
            nn.GroupNorm(1, out_dims[2]),
            nn.GELU()
        )
        self.down_32_chan = max(out_dims[3], in_dim * 2)
        self.down_32 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_32_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_32_chan),
            nn.Conv2d(self.down_32_chan, out_dims[3], 1),
            nn.GroupNorm(1, out_dims[3]),
            nn.GELU()
        )

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        x_down_4 = self.down_4(x)
        x_down_8 = self.down_8(x)
        x_down_16 = self.down_16(x)
        x_down_32 = self.down_32(x)

        return [x_down_4, x_down_8, x_down_16, x_down_32]
    
class SimpleFPN_single_scale(nn.Module):
    def __init__(self, in_dim=768, out_dims=[128, 256, 512, 1024]):
        super().__init__()
        self.down_4_chan = max(out_dims[0]*2, in_dim // 2)
        self.down_4 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_4_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan),
            nn.GELU(),
            nn.ConvTranspose2d(self.down_4_chan, self.down_4_chan // 2, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan // 2),
            nn.Conv2d(self.down_4_chan // 2, out_dims[0], 1),
            nn.GroupNorm(1, out_dims[0]),
            nn.GELU()
        )


        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        x_down_4 = self.down_4(x)
        return x_down_4

def get_generator(is_train=True):
    generator = Generator_MatteFormer(is_train=is_train)
    return generator

class Generator_MatteFormer(nn.Module):

    def __init__(self, is_train=True):

        super(Generator_MatteFormer, self).__init__()
        self.encoder = MatteFormer(embed_dim=96,
                                   depths=[2,2,6,2], # tiny-model
                                   num_heads=[3,6,12,24],
                                   window_size=7,
                                   mlp_ratio=4.0,
                                   qkv_bias=True,
                                   qk_scale=None,
                                   drop_rate=0.0,
                                   attn_drop_rate=0.0,
                                   drop_path_rate=0.3,
                                   patch_norm=True,
                                   use_checkpoint=False
                                   )
        # original
        self.decoder = decoders.__dict__['res_shortcut_decoder']()

        if is_train:
            self.init_pretrained_weight(pretrained_path=CONFIG.model.imagenet_pretrain_path)  # MatteFormer

    def init_pretrained_weight(self, pretrained_path=None):
        if not os.path.isfile(pretrained_path):
            print('Please Check your Pretrained weight path.. file not exist : {}'.format(pretrained_path))
            exit()

        weight = torch.load(pretrained_path)['model']

        # [1] get backbone weights
        weight_ = {}
        for i, (k, v) in enumerate(weight.items()):
            head = k.split('.')[0]
            if head in ['patch_embed', 'layers']:
                if 'attn_mask' in k:
                    print('[{}/{}] {} will be ignored'.format(i, len(weight.items()), k))
                    continue
                weight_.update({k: v})
            else:
                print('[{}/{}] {} will be ignored'.format(i, len(weight.items()), k))

        patch_embed_weight = weight_['patch_embed.proj.weight']
        patch_embed_weight_new = torch.nn.init.xavier_normal_(torch.randn(96, (3 + 3), 4, 4).cuda())
        patch_embed_weight_new[:, :3, :, :].copy_(patch_embed_weight)
        weight_['patch_embed.proj.weight'] = patch_embed_weight_new

        attn_layers = [k for k, v in weight_.items() if 'attn.relative_position_bias_table' in k]
        for layer_name in attn_layers:
            pos_bias = weight_[layer_name]
            n_bias, n_head = pos_bias.shape

            layer_idx, block_idx = int(layer_name.split('.')[1]), int(layer_name.split('.')[3])
            n_prior = block_idx + 1
            pos_bias_new = torch.nn.init.xavier_normal_(torch.randn(n_bias + n_prior*3, n_head))

            pos_bias_new[:n_bias, :] = pos_bias
            weight_[layer_name] = pos_bias_new

        attn_layers = [k for k, v in weight_.items() if 'attn.relative_position_index' in k]
        for layer_name in attn_layers:
            pos_index = weight_[layer_name]

            layer_idx, block_idx = int(layer_name.split('.')[1]), int(layer_name.split('.')[3])
            n_prior = block_idx + 1

            num_patch = 49
            last_idx = 169
            pos_index_new = torch.ones((num_patch, num_patch + n_prior*3)).long() * last_idx
            pos_index_new[:num_patch, :num_patch] = pos_index
            for i in range(n_prior):
                for j in range(3):
                    pos_index_new[:, num_patch + i*3 + j:num_patch + i*3 +j +1] = last_idx + i*3 + j
            weight_[layer_name] = pos_index_new

        self.encoder.load_state_dict(weight_, strict=False)
        print('load pretrained model done')

    def forward(self, image, trimap):
        inp = torch.cat((image, trimap), axis=1)
        x = self.encoder(inp, trimap)
        embedding = x[-1]
        outs = self.decoder(embedding, x[:-1])
        return outs


class PlainVitModel(ISModel):
    @serialize
    def __init__(
        self,
        backbone_params={},
        neck_params={}, 
        head_params={},
        random_split=False,
        **kwargs
        ):

        super().__init__(**kwargs)
        self.random_split = random_split

        self.patch_embed_coords = PatchEmbed(
            img_size= backbone_params['img_size'],
            patch_size=backbone_params['patch_size'], 
            in_chans=4 if self.with_prev_mask else 2, 
            embed_dim=backbone_params['embed_dim'],
        )

        self.backbone = VisionTransformer(**backbone_params)
        self.neck = SimpleFPN(**neck_params)
        # self.neck_1 = SimpleFPN_single_scale(**neck_params)
        # self.neck_2 = SimpleFPN_single_scale(**neck_params)
        # self.neck_3 = SimpleFPN_single_scale(**neck_params)
        # self.neck_6 = SimpleFPN_single_scale(**neck_params)
        self.head = SwinTransfomerSegHead(**head_params)

        self.inplanes = 128
        self.large_kernel = False
        out_dim = 128
        block = BasicBlock
        layers = [1,1,1,1]
        stride = 1

        # self.fusion_layer1 = self._make_layer(block, out_dim, layers[0], stride=stride)
        # self.fusion_layer2 = self._make_layer(block, out_dim, layers[1], stride=stride)
        # self.fusion_layer3 = self._make_layer(block, out_dim, layers[2], stride=stride)
        # self.fusion_layer4 = self._make_layer(block, out_dim, layers[3], stride=stride)

        

        # self.matting = Generator_MatteFormer(is_train=False)
        # checkpoint = torch.load('/home/zym/Desktop/yihan_temp/click2trimap/weights/matting_model.pth')
        # self.matting.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)
        # self.matting.cuda()
        # self.matting.eval()

    def backbone_forward(self, image, coord_features=None):
        coord_features = self.patch_embed_coords(coord_features)
        backbone_features = self.backbone.forward_backbone(image, coord_features, self.random_split)

        # Extract 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        B, N, C = backbone_features.shape
        grid_size = self.backbone.patch_embed.grid_size

        backbone_features = backbone_features.transpose(-1,-2).view(B, C, grid_size[0], grid_size[1])
        multi_scale_features = self.neck(backbone_features)

        return {'instances': self.head(multi_scale_features), 'instances_aux': None}
    
    def multi_neck(self, features):

        features = [features[0], features[1], features[2], features[5], features[-1]] ##select 1 2 3 6 12
        features = self.spatialize(features)
        # return [self.neck_1(features[0]), self.neck_2(features[1]), self.neck_3(features[2]), self.neck_6(features[3]), self.neck(features[4])]
        return [self.neck_1(features[0]), self.neck(features[4])]


    def spatialize(self, features):

        for i, feature in enumerate(features):
            B, N, C = feature.shape
            grid_size = self.backbone.patch_embed.grid_size
            features[i] = feature.transpose(-1,-2).view(B, C, grid_size[0], grid_size[1])
        
        return features
    
    def fusion(self, multi_block_features):

        # x = self.fusion_layer1(multi_block_features[0]) + multi_block_features[1]
        # x = self.fusion_layer2(x) + multi_block_features[2]
        # x = self.fusion_layer3(x) + multi_block_features[3]
        multi_block_features[-1][0] += multi_block_features[0]

        return multi_block_features[-1]

    def _make_layer(self, block, planes, blocks, stride=1):
            if blocks == 0:
                return nn.Sequential(nn.Identity())
            norm_layer = nn.BatchNorm2d
            upsample = None
            if stride != 1:
                upsample = nn.Sequential(
                    nn.UpsamplingNearest2d(scale_factor=2),
                    SpectralNorm(conv1x1(self.inplanes, planes * block.expansion)),
                    norm_layer(planes * block.expansion),
                )
            elif self.inplanes != planes * block.expansion:
                upsample = nn.Sequential(
                    SpectralNorm(conv1x1(self.inplanes, planes * block.expansion)),
                    norm_layer(planes * block.expansion),
                )

            layers = [block(self.inplanes, planes, stride, upsample, norm_layer, self.large_kernel)]
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, norm_layer=norm_layer, large_kernel=self.large_kernel))

            return nn.Sequential(*layers)

def conv5x5(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, groups=groups, bias=False, dilation=dilation)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, norm_layer=None, large_kernel=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.stride = stride
        conv = conv5x5 if large_kernel else conv3x3
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if self.stride > 1:
            self.conv1 = SpectralNorm(nn.ConvTranspose2d(inplanes, inplanes, kernel_size=4, stride=2, padding=1, bias=False))
        else:
            self.conv1 = SpectralNorm(conv(inplanes, inplanes))
        self.bn1 = norm_layer(inplanes)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = SpectralNorm(conv(inplanes, planes))
        self.bn2 = norm_layer(planes)
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.activation(out)

        return out

