import matplotlib
matplotlib.use('Agg')

import argparse
import tkinter as tk
from networks import model

from isegm.utils import exp
from isegm.inference import utils
from interactive_demo.app import InteractiveDemoApp

import torch
import torch.nn as nn
from utils import CONFIG
import utils as matting_utils
import os
from networks import decoders
from networks.encoders.MatteFormer import MatteFormer

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


def main():
    args = parse_args()

    torch.backends.cudnn.deterministic = True
    checkpoint_path = utils.find_checkpoint('.', args.checkpoint_c2t)
    c2t_model = utils.load_is_model(checkpoint_path, args.device, args.eval_ritm, cpu_dist_maps=True)



    # matting_model = Generator_MatteFormer(is_train=False)
    # checkpoint = torch.load(args.checkpoint_matting)
    # matting_model.load_state_dict(matting_utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)
    # matting_model.cuda()
    # matting_model.eval()

    matmodel = model.AEMatter()
    matmodel.load_state_dict(torch.load(args.checkpoint_matting,map_location='cpu')['model'])
    matmodel=matmodel.cuda()
    matmodel.eval()

    root = tk.Tk()
    root.minsize(960, 480)
    app = InteractiveDemoApp(root, args, c2t_model, matmodel)
    root.deiconify()
    app.mainloop()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint-c2t', type=str, default ='weights\click2trimap.pth',
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')

    parser.add_argument('--checkpoint-matting', type=str, default='weights\AEMFIX.ckpt',
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')

    parser.add_argument('--gpu', type=int, default=0,
                        help='Id of GPU to use.')

    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use only CPU for inference.')

    parser.add_argument('--limit-longest-size', type=int, default=800,
                        help='If the largest side of an image exceeds this value, '
                             'it is resized so that its largest side is equal to this value.')

    parser.add_argument('--cfg', type=str, default="config.yml",
                        help='The path to the config file.')

    parser.add_argument('--eval-ritm', action='store_true', default=False)

    args = parser.parse_args()
    if args.cpu:
        args.device =torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{args.gpu}')

    return args


if __name__ == '__main__':
    main()
