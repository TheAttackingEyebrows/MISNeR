from lib.utils import *
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
import logging
import plyfile
import skimage.measure
import time
import torch
import pdb
import lib


def convert_sdf_samples_to_mesh(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to vertices,faces

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function is adapted from https://github.com/facebookresearch/DeepSDF
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # return mesh
    return mesh_points, faces


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function is taken from https://github.com/facebookresearch/DeepSDF
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file
    write_verts_faces_to_file(verts, faces, ply_filename_out)


def write_verts_faces_to_file(verts, faces, ply_filename_out):

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(verts[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)


def closer_sdf_samples(filename, subsample=None):

    npz = np.load(filename)
    if subsample is None:
        return npz

    pos_tensor = remove_nans(torch.from_numpy(npz["pos"].astype(float)))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"].astype(float)))

    # split the sample into half
    half = int(subsample / 2)

    # random_pos = (torch.rand(half).cpu() * pos_tensor.shape[0]).long()
    # random_neg = (torch.rand(half).cpu() * neg_tensor.shape[0]).long()
    #
    # sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    # sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    pos_sort, pos_index = torch.sort(pos_tensor[:,-1])
    neg_sort, neg_index = torch.sort(neg_tensor[:,-1],descending=True)

    sample_pos = torch.index_select(pos_tensor, 0, pos_index[:half])
    sample_neg = torch.index_select(neg_tensor, 0, neg_index[:half])

    samples = torch.cat([sample_pos, sample_neg], 0).float()

    return samples

N=32
max_batch= 8 ** 3
offset=None
scale=None
sdf_filename = "/mnt/HDD1/Gary/meshSDF/data/pelvis/samples/true_dataset6_CLINIC_0059_mask_4label_part_1.npz"
sdf_samples = closer_sdf_samples(sdf_filename, 32768)



mesh_filename = "/mnt/HDD1/Gary/meshSDF/closer"



# NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
voxel_origin = [-1, -1, -1]
voxel_size = 2.0 / (N - 1)

samples = sdf_samples

sdf_values = samples[:, 3]
sdf_values = sdf_values.reshape(N, N, N)

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


lib.mesh.write_verts_faces_to_file(verts, faces, mesh_filename)

