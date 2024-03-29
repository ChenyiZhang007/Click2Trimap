from isegm.utils.exp_imports.default import *
from isegm.model.modeling.transformer_helper.cross_entropy_loss import CrossEntropyLoss
import utils
from dataloader.data_generator import DataGenerator
from dataloader.image_file import ImageFileTrain, ImageFileTest
from dataloader.prefetcher import Prefetcher
# from trainers.trainer import Trainer
from utils import CONFIG
from torch.utils.data import DataLoader
import toml
from networks import decoders
from networks.encoders.MatteFormer import MatteFormer
MODEL_NAME = 'cocolvis_vit_huge448'



def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (448, 448)
    model_cfg.num_max_points = 24

    backbone_params = dict(
        img_size=model_cfg.crop_size,
        patch_size=(14,14),
        in_chans=3,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4, 
        qkv_bias=True,
    )

    neck_params = dict(
        in_dim = 1280,
        out_dims = [240, 480, 960, 1920],
    )

    head_params = dict(
        in_channels=[240, 480, 960, 1920],
        in_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=3,
        loss_decode=CrossEntropyLoss(),
        align_corners=False,
        upsample=cfg.upsample,
        channels={'x1': 256, 'x2': 128, 'x4': 64}[cfg.upsample],
    )

    model = PlainVitModel(
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
        random_split=cfg.random_split,
    )

    model.backbone.init_weights_from_pretrained(cfg.IMAGENET_PRETRAINED_MODELS.MAE_HUGE)
    model.to(cfg.device)

    return model, model_cfg







