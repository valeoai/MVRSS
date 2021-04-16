"""Class to test a model"""
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from mvrss.utils.functions import transform_masks_viz, get_metrics, normalize, define_loss, get_transformations, get_qualitatives
from mvrss.utils.paths import Paths
from mvrss.utils.metrics import Evaluator
from mvrss.loaders.dataloaders import CarradaDataset


class Tester:
    """
    Class to test a model

    PARAMETERS
    ----------
    cfg: dict
        Configuration parameters used for train/test
    visualizer: object or None
        Add a visulization during testing
        Default: None
    """

    def __init__(self, cfg, visualizer=None):
        self.cfg = cfg
        self.visualizer = visualizer
        self.model = self.cfg['model']
        self.nb_classes = self.cfg['nb_classes']
        self.annot_type = self.cfg['annot_type']
        self.process_signal = self.cfg['process_signal']
        self.w_size = self.cfg['w_size']
        self.h_size = self.cfg['h_size']
        self.n_frames = self.cfg['nb_input_channels']
        self.batch_size = self.cfg['batch_size']
        self.device = self.cfg['device']
        self.custom_loss = self.cfg['custom_loss']
        self.transform_names = self.cfg['transformations'].split(',')
        self.norm_type = self.cfg['norm_type']
        self.paths = Paths().get()
        self.test_results = dict()

    def predict(self, net, seq_loader, iteration=None, get_quali=False, add_temp=False):
        """
        Method to predict on a given dataset using a fixed model

        PARAMETERS
        ----------
        net: PyTorch Model
            Network to test
        seq_loader: DataLoader
            Specific to the dataset used for test
        iteration: int
            Iteration used to display visualization
            Default: None
        get_quali: boolean
            If you want to save qualitative results
            Default: False
        add_temp: boolean
            Is the data are considered as a sequence
            Default: False
        """
        net.eval()
        transformations = get_transformations(self.transform_names, split='test',
                                              sizes=(self.w_size, self.h_size))
        rd_criterion = define_loss('range_doppler', self.custom_loss, self.device)
        ra_criterion = define_loss('range_angle', self.custom_loss, self.device)
        nb_losses = len(rd_criterion)
        running_losses = list()
        rd_running_losses = list()
        rd_running_global_losses = [list(), list()]
        ra_running_losses = list()
        ra_running_global_losses = [list(), list()]
        coherence_running_losses = list()
        rd_metrics = Evaluator(num_class=self.nb_classes)
        ra_metrics = Evaluator(num_class=self.nb_classes)
        if iteration:
            rand_seq = np.random.randint(len(seq_loader))
        with torch.no_grad():
            for i, sequence_data in enumerate(seq_loader):
                seq_name, seq = sequence_data
                path_to_frames = self.paths['carrada'] / seq_name[0]
                frame_dataloader = DataLoader(CarradaDataset(seq,
                                                             self.annot_type,
                                                             path_to_frames,
                                                             self.process_signal,
                                                             self.n_frames,
                                                             transformations,
                                                             add_temp),
                                              shuffle=False,
                                              batch_size=self.batch_size,
                                              num_workers=4)
                if iteration and i == rand_seq:
                    rand_frame = np.random.randint(len(frame_dataloader))
                if get_quali:
                    quali_iter_rd = self.n_frames-1
                    quali_iter_ra = self.n_frames-1
                for j, frame in enumerate(frame_dataloader):
                    rd_data = frame['rd_matrix'].to(self.device).float()
                    ra_data = frame['ra_matrix'].to(self.device).float()
                    ad_data = frame['ad_matrix'].to(self.device).float()
                    rd_mask = frame['rd_mask'].to(self.device).float()
                    ra_mask = frame['ra_mask'].to(self.device).float()
                    rd_data = normalize(rd_data, 'range_doppler', norm_type=self.norm_type)
                    ra_data = normalize(ra_data, 'range_angle', norm_type=self.norm_type)
                    if self.model == 'tmvanet':
                        ad_data = normalize(ad_data, 'angle_doppler', norm_type=self.norm_type)
                        rd_outputs, ra_outputs = net(rd_data, ra_data, ad_data)
                    else:
                        rd_outputs, ra_outputs = net(rd_data, ra_data)
                    rd_outputs = rd_outputs.to(self.device)
                    ra_outputs = ra_outputs.to(self.device)

                    if get_quali:
                        quali_iter_rd = get_qualitatives(rd_outputs, rd_mask, self.paths,
                                                         seq_name, quali_iter_rd, 'range_doppler')
                        quali_iter_ra = get_qualitatives(ra_outputs, ra_mask, self.paths,
                                                         seq_name, quali_iter_ra, 'range_angle')

                    rd_metrics.add_batch(torch.argmax(rd_mask, axis=1).cpu(),
                                         torch.argmax(rd_outputs, axis=1).cpu())
                    ra_metrics.add_batch(torch.argmax(ra_mask, axis=1).cpu(),
                                         torch.argmax(ra_outputs, axis=1).cpu())

                    if nb_losses < 3:
                        # Case without the CoL
                        rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                     for c in rd_criterion]
                        rd_loss = torch.mean(torch.stack(rd_losses))
                        ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                                     for c in ra_criterion]
                        ra_loss = torch.mean(torch.stack(ra_losses))
                        loss = torch.mean(rd_loss + ra_loss)
                    else:
                        # Case with the CoL
                        # Select the wCE and wSDice
                        rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                     for c in rd_criterion[:2]]
                        rd_loss = torch.mean(torch.stack(rd_losses))
                        ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                                     for c in ra_criterion[:2]]
                        ra_loss = torch.mean(torch.stack(ra_losses))
                        # Coherence loss
                        coherence_loss = rd_criterion[2](rd_outputs, ra_outputs)
                        loss = torch.mean(rd_loss + ra_loss + coherence_loss)

                    running_losses.append(loss.data.cpu().numpy()[()])
                    rd_running_losses.append(rd_loss.data.cpu().numpy()[()])
                    rd_running_global_losses[0].append(rd_losses[0].data.cpu().numpy()[()])
                    rd_running_global_losses[1].append(rd_losses[1].data.cpu().numpy()[()])
                    ra_running_losses.append(ra_loss.data.cpu().numpy()[()])
                    ra_running_global_losses[0].append(ra_losses[0].data.cpu().numpy()[()])
                    ra_running_global_losses[1].append(ra_losses[1].data.cpu().numpy()[()])
                    if nb_losses > 2:
                        coherence_running_losses.append(coherence_loss.data.cpu().numpy()[()])

                    if iteration and i == rand_seq:
                        if j == rand_frame:
                            rd_pred_masks = torch.argmax(rd_outputs, axis=1)[:5]
                            ra_pred_masks = torch.argmax(ra_outputs, axis=1)[:5]
                            rd_gt_masks = torch.argmax(rd_mask, axis=1)[:5]
                            ra_gt_masks = torch.argmax(ra_mask, axis=1)[:5]
                            rd_pred_grid = make_grid(transform_masks_viz(rd_pred_masks,
                                                                         self.nb_classes))
                            ra_pred_grid = make_grid(transform_masks_viz(ra_pred_masks,
                                                                         self.nb_classes))
                            rd_gt_grid = make_grid(transform_masks_viz(rd_gt_masks,
                                                                       self.nb_classes))
                            ra_gt_grid = make_grid(transform_masks_viz(ra_gt_masks,
                                                                       self.nb_classes))
                            self.visualizer.update_multi_img_masks(rd_pred_grid, rd_gt_grid,
                                                                   ra_pred_grid, ra_gt_grid,
                                                                   iteration)
            self.test_results = dict()
            self.test_results['range_doppler'] = get_metrics(rd_metrics, np.mean(rd_running_losses),
                                                             [np.mean(sub_loss) for sub_loss
                                                              in rd_running_global_losses])
            self.test_results['range_angle'] = get_metrics(ra_metrics, np.mean(ra_running_losses),
                                                           [np.mean(sub_loss) for sub_loss
                                                            in ra_running_global_losses])
            if nb_losses > 2:
                self.test_results['coherence_loss'] = np.mean(coherence_running_losses).item()
            self.test_results['global_acc'] = (1/2)*(self.test_results['range_doppler']['acc']+
                                                     self.test_results['range_angle']['acc'])
            self.test_results['global_prec'] = (1/2)*(self.test_results['range_doppler']['prec']+
                                                      self.test_results['range_angle']['prec'])
            self.test_results['global_dice'] = (1/2)*(self.test_results['range_doppler']['dice']+
                                                      self.test_results['range_angle']['dice'])

            rd_metrics.reset()
            ra_metrics.reset()
        return self.test_results

    def write_params(self, path):
        """Write quantitative results of the Test"""
        with open(path, 'w') as fp:
            json.dump(self.test_results, fp)

    def set_device(self, device):
        """Set device used for test (supported: 'cuda', 'cpu')"""
        self.device = device

    def set_annot_type(self, annot_type):
        """Set annotation type to test on (specific to CARRADA)"""
        self.annot_type = annot_type
