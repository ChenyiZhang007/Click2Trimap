
import numpy as np


def compute_region_mse(pred, target, select):
    error_map = (pred - target) / 255.0
    loss = np.sum((error_map ** 2) * (select)) / (np.sum(select) + 1e-8)
    return loss


def compute_region_sad(pred, target, select):
    error_map = np.abs((pred - target) / 255.0)
    loss = np.sum(error_map * (select))
    loss /= 1000
    return loss

def compute_mse_loss(pred, target, trimap, trimap_gt):

    fg = trimap_gt == 255
    bg = trimap_gt == 0
    uk = trimap_gt == 128
    whole = trimap_gt != 127

    loss_fg = compute_region_mse(pred, target, fg)
    loss_bg = compute_region_mse(pred, target, bg)
    loss_uk = compute_region_mse(pred, target, uk)
    loss_global = compute_region_mse(pred, target, whole)



    # select_1 = trimap == 128
    # select_2 = trimap_2 == 128
    # select = trimap != 127

    # select = np.logical_or(select_1, select_2)
    # select = select_2

    # error_map = (pred - target) / 255.0
    # loss = np.sum((error_map ** 2) * (select)) / (np.sum(select) + 1e-8)

    return [loss_fg, loss_bg, loss_uk, loss_global]


def compute_sad_loss(pred, target, trimap, trimap_gt):
    fg = trimap_gt == 255
    bg = trimap_gt == 0
    uk = trimap_gt == 128
    whole = trimap_gt != 127

    loss_fg = compute_region_sad(pred, target, fg)
    loss_bg = compute_region_sad(pred, target, bg)
    loss_uk = compute_region_sad(pred, target, uk)
    loss_global = compute_region_sad(pred, target, whole)

    return [loss_fg, loss_bg, loss_uk, loss_global]

