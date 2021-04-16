"""Classes to load Carrada dataset"""
import os
import numpy as np
from skimage import transform
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from mvrss.loaders.dataset import Carrada
from mvrss.utils.paths import Paths


class SequenceCarradaDataset(Dataset):
    """DataLoader class for Carrada sequences"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.seq_names = list(self.dataset.keys())

    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, idx):
        seq_name = self.seq_names[idx]
        return seq_name, self.dataset[seq_name]


class CarradaDataset(Dataset):
    """DataLoader class for Carrada sequences
    Load frames, only for semantic segmentation
    Specific to load several frames at the same time (sub sequences)
    Aggregated Tensor Input + Multiple Output

    PARAMETERS
    ----------
    dataset: SequenceCarradaDataset object
    annotation_type: str
        Supported annotations are 'sparse', 'dense'
    path_to_frames: str
        Path to the frames of a given sequence (folder of the sequence)
    process_signal: boolean
        Load signal w/ or w/o processing (power, log transform)
    n_frame: int
        Number of frames used for each sample
    transformations: list of functions
        Preprocessing or data augmentation functions
        Default: None
    add_temp: boolean
        Formating the input tensors as sequences
        Default: False
    """

    def __init__(self, dataset, annotation_type, path_to_frames, process_signal,
                 n_frames, transformations=None, add_temp=False):
        self.dataset = dataset
        self.annotation_type = annotation_type
        self.path_to_frames = Path(path_to_frames)
        self.process_signal = process_signal
        self.n_frames = n_frames
        self.transformations = transformations
        self.add_temp = add_temp
        self.dataset = self.dataset[self.n_frames-1:]  # remove n first frames
        self.path_to_annots = self.path_to_frames / 'annotations' / self.annotation_type

    def transform(self, frame, is_vflip=False, is_hflip=False):
        """
        Method to apply preprocessing / data augmentation functions

        PARAMETERS
        ----------
        frame: dict
            Contains the matrices and the masks on which we want to apply the transformations
        is_vfilp: boolean
            If you want to apply a vertical flip
            Default: False
        is_hfilp: boolean
            If you want to apply a horizontal flip
            Default: False

        RETURNS
        -------
        frame: dict
        """
        if self.transformations is not None:
            for function in self.transformations:
                if isinstance(function, VFlip):
                    if is_vflip:
                        frame = function(frame)
                    else:
                        continue
                if isinstance(function, HFlip):
                    if is_hflip:
                        frame = function(frame)
                    else:
                        continue
                if not isinstance(function, VFlip) and not isinstance(function, HFlip):
                    frame = function(frame)
        return frame

    def __len__(self):
        """Number of frames per sequence"""
        return len(self.dataset)

    def __getitem__(self, idx):
        init_frame_name = self.dataset[idx][0]
        frame_id = int(init_frame_name)
        frame_names = [str(f_id).zfill(6) for f_id in range(frame_id-self.n_frames+1, frame_id+1)]
        rd_matrices = list()
        ra_matrices = list()
        ad_matrices = list()
        rd_mask = np.load(os.path.join(self.path_to_annots, init_frame_name,
                                       'range_doppler.npy'))
        ra_mask = np.load(os.path.join(self.path_to_annots, init_frame_name,
                                       'range_angle.npy'))
        for frame_name in frame_names:
            if self.process_signal:
                rd_matrix = np.load(os.path.join(self.path_to_frames,
                                                 'range_doppler_processed',
                                                 frame_name + '.npy'))
                ra_matrix = np.load(os.path.join(self.path_to_frames,
                                                 'range_angle_processed',
                                                 frame_name + '.npy'))
                ad_matrix = np.load(os.path.join(self.path_to_frames,
                                                 'angle_doppler_processed',
                                                 frame_name + '.npy'))
            else:
                rd_matrix = np.load(os.path.join(self.path_to_frames,
                                                 'range_doppler_raw',
                                                 frame_name + '.npy'))
                ra_matrix = np.load(os.path.join(self.path_to_frames,
                                                 'range_angle_raw',
                                                 frame_name + '.npy'))
                ad_matrix = np.load(os.path.join(self.path_to_frames,
                                                 'angle_doppler_raw',
                                                 frame_name + '.npy'))

            rd_matrices.append(rd_matrix)
            ra_matrices.append(ra_matrix)
            ad_matrices.append(ad_matrix)

        # Apply the same transfo to all representations
        if np.random.uniform(0, 1) > 0.5:
            is_vflip = True
        else:
            is_vflip = False
        if np.random.uniform(0, 1) > 0.5:
            is_hflip = True
        else:
            is_hflip = False

        rd_matrix = np.dstack(rd_matrices)
        rd_matrix = np.rollaxis(rd_matrix, axis=-1)
        rd_frame = {'matrix': rd_matrix, 'mask': rd_mask}
        rd_frame = self.transform(rd_frame, is_vflip=is_vflip, is_hflip=is_hflip)
        if self.add_temp:
            if isinstance(self.add_temp, bool):
                rd_frame['matrix'] = np.expand_dims(rd_frame['matrix'], axis=0)
            else:
                assert isinstance(self.add_temp, int)
                rd_frame['matrix'] = np.expand_dims(rd_frame['matrix'],
                                                    axis=self.add_temp)

        ra_matrix = np.dstack(ra_matrices)
        ra_matrix = np.rollaxis(ra_matrix, axis=-1)
        ra_frame = {'matrix': ra_matrix, 'mask': ra_mask}
        ra_frame = self.transform(ra_frame, is_vflip=is_vflip, is_hflip=is_hflip)
        if self.add_temp:
            if isinstance(self.add_temp, bool):
                ra_frame['matrix'] = np.expand_dims(ra_frame['matrix'], axis=0)
            else:
                assert isinstance(self.add_temp, int)
                ra_frame['matrix'] = np.expand_dims(ra_frame['matrix'],
                                                    axis=self.add_temp)

        ad_matrix = np.dstack(ad_matrices)
        ad_matrix = np.rollaxis(ad_matrix, axis=-1)
        # Fill fake mask just to apply transform
        ad_frame = {'matrix': ad_matrix, 'mask': rd_mask.copy()}
        ad_frame = self.transform(ad_frame, is_vflip=is_vflip, is_hflip=is_hflip)
        if self.add_temp:
            if isinstance(self.add_temp, bool):
                ad_frame['matrix'] = np.expand_dims(ad_frame['matrix'], axis=0)
            else:
                assert isinstance(self.add_temp, int)
                ad_frame['matrix'] = np.expand_dims(ad_frame['matrix'],
                                                    axis=self.add_temp)

        frame = {'rd_matrix': rd_frame['matrix'], 'rd_mask': rd_frame['mask'],
                 'ra_matrix': ra_frame['matrix'], 'ra_mask': ra_frame['mask'],
                 'ad_matrix': ad_frame['matrix']}

        return frame


class Rescale:
    """Rescale the image in a sample to a given size.

    PARAMETERS
    ----------
    output_size: tuple or int
        Desired output size. If tuple, output is
        matched to output_size. If int, smaller of image edges is matched
        to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, frame):
        matrix, rd_mask, ra_mask = frame['matrix'], frame['rd_mask'], frame['ra_mask']
        h, w = matrix.shape[1:]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        # transform.resize induce a smoothing effect on the values
        # transform only the input data
        matrix = transform.resize(matrix, (matrix.shape[0], new_h, new_w))
        return {'matrix': matrix, 'rd_mask': rd_mask, 'ra_mask': ra_mask}


class Flip:
    """
    Randomly flip the matrix with a proba p
    """

    def __init__(self, proba):
        assert proba <= 1.
        self.proba = proba

    def __call__(self, frame):
        matrix, mask = frame['matrix'], frame['mask']
        h_flip_proba = np.random.uniform(0, 1)
        if h_flip_proba < self.proba:
            matrix = np.flip(matrix, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        v_flip_proba = np.random.uniform(0, 1)
        if v_flip_proba < self.proba:
            matrix = np.flip(matrix, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()
        return {'matrix': matrix, 'mask': mask}


class HFlip:
    """
    Randomly horizontal flip the matrix with a proba p
    """

    def __init__(self):
        pass

    def __call__(self, frame):
        matrix, mask = frame['matrix'], frame['mask']
        matrix = np.flip(matrix, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()
        return {'matrix': matrix, 'mask': mask}


class VFlip:
    """
    Randomly vertical flip the matrix with a proba p
    """

    def __init__(self):
        pass

    def __call__(self, frame):
        matrix, mask = frame['matrix'], frame['mask']
        matrix = np.flip(matrix, axis=2).copy()
        mask = np.flip(mask, axis=2).copy()
        return {'matrix': matrix, 'mask': mask}


def test_sequence():
    dataset = Carrada().get('Train')
    dataloader = DataLoader(SequenceCarradaDataset(dataset), batch_size=1,
                            shuffle=False, num_workers=0)
    for i, data in enumerate(dataloader):
        seq_name, seq = data
        if i == 0:
            seq = [subseq[0] for subseq in seq]
            assert seq_name[0] == '2019-09-16-12-52-12'
            assert '000163' in seq
            assert '001015' in seq
        else:
            break


def test_carradadataset():
    paths = Paths().get()
    n_frames = 3
    dataset = Carrada().get('Train')
    seq_dataloader = DataLoader(SequenceCarradaDataset(dataset), batch_size=1,
                                shuffle=True, num_workers=0)
    for _, data in enumerate(seq_dataloader):
        seq_name, seq = data
        path_to_frames = paths['carrada'] / seq_name[0]
        frame_dataloader = DataLoader(CarradaDataset(seq,
                                                     'dense',
                                                     path_to_frames,
                                                     process_signal=True,
                                                     n_frames=n_frames),
                                      shuffle=False,
                                      batch_size=1,
                                      num_workers=0)
        for _, frame in enumerate(frame_dataloader):
            assert list(frame['rd_matrix'].shape[2:]) == [256, 64]
            assert list(frame['ra_matrix'].shape[2:]) == [256, 256]
            assert list(frame['ad_matrix'].shape[2:]) == [256, 64]
            assert frame['rd_matrix'].shape[1] == n_frames
            assert list(frame['rd_mask'].shape[2:]) == [256, 64]
            assert list(frame['ra_mask'].shape[2:]) == [256, 256]
        break


def test_subflip():
    paths = Paths().get()
    n_frames = 3
    dataset = Carrada().get('Train')
    seq_dataloader = DataLoader(SequenceCarradaDataset(dataset), batch_size=1,
                                shuffle=True, num_workers=0)
    for _, data in enumerate(seq_dataloader):
        seq_name, seq = data
        path_to_frames = paths['carrada'] / seq_name[0]
        frame_dataloader = DataLoader(CarradaDataset(seq,
                                                     'dense',
                                                     path_to_frames,
                                                     process_signal=True,
                                                     n_frames=n_frames),
                                      shuffle=False,
                                      batch_size=1,
                                      num_workers=0)
        for _, frame in enumerate(frame_dataloader):
            rd_matrix = frame['rd_matrix'][0].cpu().detach().numpy()
            rd_mask = frame['rd_mask'][0].cpu().detach().numpy()
            rd_frame_test = {'matrix': rd_matrix,
                             'mask': rd_mask}
            rd_frame_vflip = VFlip()(rd_frame_test)
            rd_matrix_vflip = rd_frame_vflip['matrix']
            rd_frame_hflip = HFlip()(rd_frame_test)
            rd_matrix_hflip = rd_frame_hflip['matrix']
            assert rd_matrix[0][0][0] == rd_matrix_vflip[0][0][-1]
            assert rd_matrix[0][0][-1] == rd_matrix_vflip[0][0][0]
            assert rd_matrix[0][0][0] == rd_matrix_hflip[0][-1][0]
            assert rd_matrix[0][-1][0] == rd_matrix_hflip[0][0][0]
        break

    
if __name__ == '__main__':
    # test_sequence()
    # test_carradadataset()
    test_subflip()
