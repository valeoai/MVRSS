"""A lot of functions used in our pipelines"""
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from mvrss.utils import MVRSS_HOME
from mvrss.losses.soft_dice import SoftDiceLoss
from mvrss.losses.coherence import CoherenceLoss
from mvrss.loaders.dataloaders import Rescale, Flip, HFlip, VFlip


def get_class_weights(signal_type):
    """Load class weights for custom loss

    PARAMETERS
    ----------
    signal_type: str
        Supported: 'range_doppler', 'range_angle'

    RETURNS
    -------
    weights: numpy array
    """
    weight_path = MVRSS_HOME / 'config_files'
    if signal_type in ('range_angle'):
        file_name = 'ra_weights.json'
    elif signal_type in ('range_doppler'):
        file_name = 'rd_weights.json'
    else:
        raise ValueError('Signal type {} is not supported.'.format(signal_type))
    file_path = weight_path / file_name
    with open(file_path, 'r') as fp:
        weights = json.load(fp)
    weights = np.array([weights['background'], weights['pedestrian'],
                        weights['cyclist'], weights['car']])
    weights = torch.from_numpy(weights)
    return weights


def transform_masks_viz(masks, nb_classes):
    """Used for visualization"""
    masks = masks.unsqueeze(1)
    masks = (masks.float()/nb_classes)
    return masks


def get_metrics(metrics, loss, losses=None):
    """Structure the metric results

    PARAMETERS
    ----------
    metrics: object
        Contains statistics recorded during inference
    loss: tensor
        Loss value
    losses: list
        List of loss values

    RETURNS
    -------
    metrics_values: dict
    """
    metrics_values = dict()
    metrics_values['loss'] = loss.item()
    if isinstance(losses, list):
        metrics_values['loss_ce'] = losses[0].item()
        metrics_values['loss_dice'] = losses[1].item()
    acc, acc_by_class = metrics.get_pixel_acc_class()  # harmonic_mean=True)
    prec, prec_by_class = metrics.get_pixel_prec_class()
    recall, recall_by_class = metrics.get_pixel_recall_class()  # harmonic_mean=True)
    miou, miou_by_class = metrics.get_miou_class()  # harmonic_mean=True)
    dice, dice_by_class = metrics.get_dice_class()
    metrics_values['acc'] = acc
    metrics_values['acc_by_class'] = acc_by_class.tolist()
    metrics_values['prec'] = prec
    metrics_values['prec_by_class'] = prec_by_class.tolist()
    metrics_values['recall'] = recall
    metrics_values['recall_by_class'] = recall_by_class.tolist()
    metrics_values['miou'] = miou
    metrics_values['miou_by_class'] = miou_by_class.tolist()
    metrics_values['dice'] = dice
    metrics_values['dice_by_class'] = dice_by_class.tolist()
    return metrics_values


def normalize(data, signal_type, norm_type='local'):
    """
    Method to normalise the radar views

    PARAMETERS
    ----------
    data: numpy array
        Radar view (batch)
    signal_type: str
        Type of radar view
        Supported: 'range_doppler', 'range_angle' and 'angle_doppler'
    norm_type: str
        Type of normalisation to apply
        Supported: 'local', 'tvt'

    RETURNS
    -------
    norm_data: numpy array
        normalised radar view
    """
    if norm_type in ('local'):
        min_value = torch.min(data)
        max_value = torch.max(data)
        norm_data = torch.div(torch.sub(data, min_value), torch.sub(max_value, min_value))
        return norm_data

    elif signal_type == 'range_doppler':
        if norm_type == 'tvt':
            file_path = MVRSS_HOME / 'config_files' / 'rd_stats_all.json'
        else:
            raise TypeError('Global type {} is not supported'.format(norm_type))
        with open(file_path, 'r') as fp:
            rd_stats = json.load(fp)
        min_value = torch.tensor(rd_stats['min_val'])
        max_value = torch.tensor(rd_stats['max_val'])

    elif signal_type == 'range_angle':
        if norm_type == 'tvt':
            file_path = MVRSS_HOME / 'config_files' / 'ra_stats_all.json'
        else:
            raise TypeError('Global type {} is not supported'.format(norm_type))
        with open(file_path, 'r') as fp:
            ra_stats = json.load(fp)
        min_value = torch.tensor(ra_stats['min_val'])
        max_value = torch.tensor(ra_stats['max_val'])

    elif signal_type == 'angle_doppler':
        if norm_type == 'tvt':
            file_path = MVRSS_HOME / 'config_files' / 'ad_stats_all.json'
        else:
            raise TypeError('Global type {} is not supported'.format(norm_type))
        with open(file_path, 'r') as fp:
            ad_stats = json.load(fp)
        min_value = torch.tensor(ad_stats['min_val'])
        max_value = torch.tensor(ad_stats['max_val'])

    else:
        raise TypeError('Signal {} is not supported.'.format(signal_type))

    norm_data = torch.div(torch.sub(data, min_value),
                          torch.sub(max_value, min_value))
    return norm_data


def define_loss(signal_type, custom_loss, device):
    """
    Method to define the loss to use during training

    PARAMETERS
    ----------
    signal_type: str
        Type of radar view
        Supported: 'range_doppler', 'range_angle' or 'angle_doppler'
    custom loss: str
        Short name of the custom loss to use
        Supported: 'wce', 'sdice', 'wce_w10sdice' or 'wce_w10sdice_w5col'
        Default: Cross Entropy is used for any other str
    devide: str
        Supported: 'cuda' or 'cpu'
    """
    if custom_loss == 'wce':
        weights = get_class_weights(signal_type)
        loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
    elif custom_loss == 'sdice':
        loss = SoftDiceLoss()
    elif custom_loss == 'wce_w10sdice':
        weights = get_class_weights(signal_type)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        loss = [ce_loss, SoftDiceLoss(global_weight=10.)]
    elif custom_loss == 'wce_w10sdice_w5col':
        weights = get_class_weights(signal_type)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        loss = [ce_loss, SoftDiceLoss(global_weight=10.), CoherenceLoss(global_weight=5.)]
    else:
        loss = nn.CrossEntropyLoss()
    return loss


def get_transformations(transform_names, split='train', sizes=None):
    """Create a list of functions used for preprocessing

    PARAMETERS
    ----------
    transform_names: list
        List of str, one for each transformation
    split: str
        Split currently used
    sizes: int or tuple (optional)
        Used for rescaling
        Default: None
    """
    transformations = list()
    if 'rescale' in transform_names:
        transformations.append(Rescale(sizes))
    if 'flip' in transform_names and split == 'train':
        transformations.append(Flip(0.5))
    if 'vflip' in transform_names and split == 'train':
        transformations.append(VFlip())
    if 'hflip' in transform_names and split == 'train':
        transformations.append(HFlip())
    return transformations


def mask_to_img(mask):
    """Generate colors per class, only 3 classes are supported"""
    mask_img = np.zeros((mask.shape[0],
                         mask.shape[1], 3), dtype=np.uint8)
    mask_img[mask == 1] = [255, 0, 0]
    mask_img[mask == 2] = [0, 255, 0]
    mask_img[mask == 3] = [0, 0, 255]
    mask_img = Image.fromarray(mask_img)
    return mask_img


def get_qualitatives(outputs, masks, paths, seq_name, quali_iter, signal_type=None):
    """
    Method to get qualitative results

    PARAMETERS
    ----------
    outputs: torch tensor
        Predicted masks
    masks: torch tensor
        Ground truth masks
    paths: dict
    seq_name: str
    quali_iter: int
        Current iteration on the dataset
    signal_type: str

    RETURNS
    -------
    quali_iter: int
    """
    if signal_type:
        folder_path = paths['logs'] / signal_type / seq_name[0]
    else:
        folder_path = paths['logs'] / seq_name[0]
    folder_path.mkdir(parents=True, exist_ok=True)
    outputs = torch.argmax(outputs, axis=1).cpu().numpy()
    masks = torch.argmax(masks, axis=1).cpu().numpy()
    for i in range(outputs.shape[0]):
        mask_img = mask_to_img(masks[i])
        mask_path = folder_path / 'mask_{}.png'.format(quali_iter)
        mask_img.save(mask_path)
        output_img = mask_to_img(outputs[i])
        output_path = folder_path / 'output_{}.png'.format(quali_iter)
        output_img.save(output_path)
        quali_iter += 1
    return quali_iter


def count_params(model):
    """Count trainable parameters of a PyTorch Model"""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_parameters])
    return nb_params
