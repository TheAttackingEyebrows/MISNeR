#!/usr/bin/env python3

import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import json
import time
import pdb
import torch.nn as nn
import imageio
import numpy as np
import torch.nn.functional as F
from lib.models.decoder import *
from lib.models.encoder import *
from torchvision.utils import save_image
from lib.mesh import convert_sdf_samples_to_ply, convert_sdf_samples_to_mesh
from matplotlib import pyplot as plt
from network import *



import lib
import lib.workspace as ws
from lib.utils import *

device = "cuda:0"


def optimize_mesh(
    para, image, map, N=256, max_batch=32 ** 3, offset=None, scale=None, output_mesh = False, filename = None
):
    start = time.time()
    ply_filename = filename

    para.eval()
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())

    # samples = torch.zeros(N ** 3, 4)
    #
    # # transform first 3 columns
    # # to be the x, y, z index
    # samples[:, 2] = overall_index % N
    # samples[:, 1] = (overall_index.long() / N) % N
    # samples[:, 0] = ((overall_index.long() / N) / N) % N
    #
    # # transform first 3 columns
    # # to be the x, y, z coordinate
    # samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    # samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    # samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    #
    # num_samples = N ** 3
    #
    # samples.requires_grad = False
    #
    # head = 0
    #
    # while head < num_samples:
    #     sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
    #     samples[head : min(head + max_batch, num_samples), 3] = (
    #         decode_sdf(para, image, sample_subset.unsqueeze(0))
    #         .squeeze(1)
    #         .detach()
    #         .cpu()
    #     )
    #     head += max_batch
    #
    samples = para(image)
    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    s_min = sdf_values.min()


    if output_mesh is False:

        convert_sdf_samples_to_ply(
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            ply_filename + ".ply",
            offset,
            scale,
        )
        return

    else:

        verts, faces = convert_sdf_samples_to_mesh(
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            offset,
            scale,
        )

        # first fetch bins that are activated
        k = ((verts[:, 2] -  voxel_origin[2])/voxel_size).astype(int)
        j = ((verts[:, 1] -  voxel_origin[1])/voxel_size).astype(int)
        i = ((verts[:, 0] -  voxel_origin[0])/voxel_size).astype(int)
        # find points around
        next_samples = i*N*N + j*N + k
        next_samples_ip = np.minimum(i+1,N-1)*N*N + j*N + k
        next_samples_jp = i*N*N + np.minimum(j+1,N-1)*N + k
        next_samples_kp = i*N*N + j*N + np.minimum(k+1,N-1)
        next_samples_im = np.maximum(i-1,0)*N*N + j*N + k
        next_samples_jm = i*N*N + np.maximum(j-1,0)*N + k
        next_samples_km = i*N*N + j*N + np.maximum(k-1,0)

        next_indices = np.concatenate((next_samples,next_samples_ip, next_samples_jp,next_samples_kp,next_samples_im,next_samples_jm, next_samples_km))

        return verts, faces, samples, next_indices


def main_function(experiment_directory, continue_from,  iterations, marching_cubes_resolution, regularize):

    specs = ws.load_experiment_specifications(experiment_directory)

    print("Reconstruction from experiment description: \n" + ' '.join([str(elem) for elem in specs["Description"]]))

    data_source = specs["DataSource"]
    test_split_file = specs["TestSplit"]

    # arch_encoder = __import__("lib.models." + specs["NetworkEncoder"], fromlist=["ResNet3D"])
    # arch_decoder = __import__("lib.models." + specs["NetworkDecoder"], fromlist=["DeepSDF"])
    # latent_size = specs["CodeLength"]
    #
    # encoder = arch_encoder.ResNet3D(latent_size, specs["Depth"], norm_type = specs["NormType"]).cuda()
    # decoder = arch_decoder.DeepSDF(latent_size, **specs["NetworkSpecs"]).cuda()
    #
    # encoder = torch.nn.DataParallel(encoder)
    # decoder = torch.nn.DataParallel(decoder)

    # para = ShapeNet256Vox().cuda(device)
    network = MISNeR(device)

    print("testing with {} GPU(s)".format(torch.cuda.device_count()))

    num_samp_per_scene = specs["SamplesPerScene"]
    with open(test_split_file, "r") as f:
        test_split = json.load(f)

    sdf_dataset_test = lib.data.Volume2SDF(
        data_source, test_split, num_samp_per_scene, is_train=False, num_views = specs["NumberOfViews"]
    )
    torch.manual_seed(int( time.time() * 1000.0 ))
    sdf_loader_test = data_utils.DataLoader(
        sdf_dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )

    num_scenes = len(sdf_loader_test)
    print("There are {} scenes".format(num_scenes))

    print('Loading epoch "{}"'.format(continue_from))

    model_epoch = ws.load_para_parameters(
        "experiments/VSD", continue_from, network
    )
    network.eval()

    optimization_meshes_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(continue_from)
    )

    if not os.path.isdir(optimization_meshes_dir):
        os.makedirs(optimization_meshes_dir)

    for sdf_data, image, name, map in sdf_loader_test:

        out_name = name[0].split("/")[-1]

        print('Reconstructing {}...'.format(out_name))

        verts, faces, samples, next_indices, id = lib.mesh.create_mesh(network, image, map, N=512, output_mesh = True, filename = os.path.join(optimization_meshes_dir,out_name + ".nii.gz"))
        # verts, faces, samples, next_indices = optimize_mesh(para, image, map, N=128, output_mesh = True)
        if id == 0:
            print("skip!")
            continue
        # store raw output
        mesh_filename = os.path.join(optimization_meshes_dir, out_name + "predicted.ply")
        lib.mesh.write_verts_faces_to_file(verts, faces, mesh_filename)


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        default="experiments/VSD",
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        default="latest",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--resolution",
        default=128,
        help="Marching cubes resolution for reconstructed surfaces.",
    )
    arg_parser.add_argument(
        "--iterations",
        default=100,
        help="Number of refinement iterations.",
    )
    arg_parser.add_argument("--regularize", default=0.0, help="L2 regularization weight on latent vector")

    args = arg_parser.parse_args()
    main_function(args.experiment_directory, args.continue_from, int(args.iterations), int(args.resolution), float(args.regularize))
