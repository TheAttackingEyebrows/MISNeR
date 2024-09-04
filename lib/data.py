#!/usr/bin/env python3

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import pdb
import imageio
import time
import random
import imageio
import nibabel as nib
import torch.nn.functional as F
from lib.utils import *
from matplotlib import pyplot as plt
from PIL import Image


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample
    ):
        self.subsample = subsample
        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)


    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, self.npyfiles[idx]
        )
        return unpack_sdf_samples(filename, self.subsample), idx, self.npyfiles[idx]


class RGBA2SDF(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        is_train=False,
        num_views = 1,
    ):
        self.subsample = subsample
        self.is_train = is_train
        self.data_source = data_source
        self.num_views = num_views
        self.npyfiles =  get_instance_filenames(data_source, split)

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):

        mesh_name = self.npyfiles[idx].split(".npz")[0]

        # fetch sdf samples
        sdf_filename = os.path.join(
            self.data_source, self.npyfiles[idx]
        )
        sdf_samples = unpack_sdf_samples(sdf_filename,  self.subsample)

        if self.is_train:
            # reset seed for random sampling training data (see https://github.com/pytorch/pytorch/issues/5059)
            np.random.seed( int(time.time()) + idx)
            id = np.random.randint(0, self.num_views)
        else:
            np.random.seed(idx)
            id = np.random.randint(0, self.num_views)

        view_id = '{0:02d}'.format(id)

        image_filename = os.path.join(self.data_source, mesh_name.replace("samples", "renders"), view_id + ".png")
        RGBA = unpack_images(image_filename)

        # fetch cameras
        metadata_filename = os.path.join(self.data_source, mesh_name.replace("samples", "renders"), "rendering_metadata.txt")
        intrinsic, extrinsic = get_camera_matrices(metadata_filename, id)

        return sdf_samples, RGBA, intrinsic, extrinsic, mesh_name


#for pelvis
class Volume2SDF(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        is_train=False,
        num_views = 1,
    ):
        self.subsample = subsample
        self.is_train = is_train
        self.data_source = data_source
        self.num_views = num_views
        self.npyfiles =  get_instance_filenames(data_source, split)
        # self.npyfiles.sort()

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):

        mesh_name = self.npyfiles[idx].split(".npz")[0]
        # # heatmap_number = "/mnt/HDD1/Gary/meshSDF/data/chaos/heatmap/" + mesh_name.split("_")[5] + ".nii.gz"
        # # heatmap_number = "/mnt/HDD1/Gary/meshSDF/data/kidney_cropped/heatmap/" + mesh_name.split("-")[1] + ".nii.gz"
        # heatmap_filename = os.path.join(self.data_source, mesh_name.replace("samples_normalized", "heatmap").replace("dataset3_colon_","").replace("_mask_4label.npz",".nii.gz"))
        # heatmap = nib.nifti1.load(heatmap_filename).get_fdata()
        #
        # heatmap = torch.from_numpy(heatmap)

        heatmap = 0


        # fetch sdf samples
        sdf_filename = os.path.join(
            self.data_source, self.npyfiles[idx]
        )
        # sdf_samples = get_sdf_samples(sdf_filename, heatmap, self.subsample)

        sdf_samples = unpack_sdf_samples(sdf_filename, self.subsample)

        # image_filename = os.path.join(self.data_source, mesh_name.replace("normalized_samples", "renders").replace("labels", "volume")) +".nii.gz"
        # image_filename = os.path.join(self.data_source, mesh_name.replace("normalized_samples", "renders").replace("true_", ""))[:-18] +"data.nii.gz"
        image_filename = os.path.join(self.data_source, mesh_name.replace("normalized_samples", "renders_256").replace("dataset3_", "").replace("_mask_4label", "")) +".nii.gz"
        img = nib.nifti1.load(image_filename).get_fdata()

        x = np.float32(img)
        mean_x = np.mean(x)
        std_x = np.std(x)
        x = (x - mean_x) / std_x
        x = torch.from_numpy(x)

        x = torch.unsqueeze(x,0)


        return sdf_samples, x, mesh_name, heatmap


class XRay2SDF(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        is_train=False,
        num_views = 1,
    ):
        self.subsample = subsample
        self.is_train = is_train
        self.data_source = data_source
        self.num_views = num_views
        self.npyfiles =  get_instance_filenames(data_source, split)
        # self.npyfiles.sort()

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):

        mesh_name = self.npyfiles[idx].split(".npz")[0].replace("normalized_samples","X-Rays_256") + ".nii.gz"

        # fetch sdf samples
        sdf_filename = os.path.join(
            self.data_source, self.npyfiles[idx]
        )

        sdf_samples = unpack_sdf_samples(sdf_filename, self.subsample)
        sdf_samples = sdf_samples

        image_filename = os.path.join(
            self.data_source, mesh_name
        )

        # im_frame = Image.open(image_filename)
        # img = np.array(im_frame)
        #
        #
        # x = np.float32(img)
        # x = x/255
        # x = torch.from_numpy(x)
        #
        # x = torch.unsqueeze(x,0)

        img = nib.nifti1.load(image_filename).get_fdata()

        x = np.float32(img)
        x_min = x.min()
        x_max = x.max()
        x = (x - x_min) / (x_max-x_min)
        x = torch.from_numpy(x)

        x = torch.unsqueeze(x, 0)
        x = torch.squeeze(x, -1)


        return sdf_samples, x, mesh_name

class XRay2CT(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        is_train=False,
        num_views = 1,
    ):
        self.subsample = subsample
        self.is_train = is_train
        self.data_source = data_source
        self.num_views = num_views
        self.npyfiles =  get_instance_filenames(data_source, split)

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):

        xray_name = self.npyfiles[idx].split(".npz")[0].replace("normalized_samples","X-Rays_256") + ".nii.gz"

        xray_filename = os.path.join(
            self.data_source, xray_name
        )

        xray = nib.nifti1.load(xray_filename).get_fdata()

        x = np.float32(xray)
        x_min = x.min()
        x_max = x.max()
        x = (x - x_min) / (x_max-x_min)
        x = torch.from_numpy(x)

        x = torch.unsqueeze(x, 0)
        x = torch.squeeze(x, -1)

        ct_name = xray_name.replace("X-Rays", "CT")

        ct_filename = os.path.join(
            self.data_source, ct_name
        )

        ct = nib.nifti1.load(ct_filename).get_fdata()

        y = np.float32(ct)
        y_min = y.min()
        y_max = y.max()
        y = (y - y_min) / (y_max-y_min)
        y = torch.from_numpy(y)

        y = torch.unsqueeze(y, 0)
        y = torch.squeeze(y, -1)

        return y, x, xray_name


class Xwitch2SDF(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        is_train=False,
        num_views = 1,
    ):
        self.subsample = subsample
        self.is_train = is_train
        self.data_source = data_source
        self.num_views = num_views
        self.npyfiles =  get_instance_filenames(data_source, split)
        # self.npyfiles.sort()

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        # fetch sdf samples
        mesh_name = self.npyfiles[idx].split(".npz")[0].replace("normalized_samples", "X-Rays_256") + ".nii.gz"

        sdf_filename = os.path.join(
            self.data_source, self.npyfiles[idx]
        )
        # sdf_samples = get_sdf_samples(sdf_filename, heatmap, self.subsample)

        sdf_samples = unpack_sdf_samples(sdf_filename, self.subsample)

        xray_name = self.npyfiles[idx].split(".npz")[0].replace("normalized_samples", "X-Rays_256") + ".nii.gz"

        xray_filename = os.path.join(
            self.data_source, xray_name
        )

        xray = nib.nifti1.load(xray_filename).get_fdata()

        x = np.float32(xray)
        x_min = x.min()
        x_max = x.max()
        x = (x - x_min) / (x_max - x_min)
        x = torch.from_numpy(x)

        x = torch.unsqueeze(x, 0)
        x = torch.squeeze(x, -1)

        ct_name = xray_name.replace("X-Rays_256", "CT_rotated")

        ct_filename = os.path.join(
            self.data_source, ct_name
        )

        ct = nib.nifti1.load(ct_filename).get_fdata()

        y = np.float32(ct)
        y_min = y.min()
        y_max = y.max()
        y = (y - y_min) / (y_max - y_min)
        y = torch.from_numpy(y)

        y = torch.unsqueeze(y, 0)
        y = torch.squeeze(y, -1)

        return sdf_samples, x, y, mesh_name