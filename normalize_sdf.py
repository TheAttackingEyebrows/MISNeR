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





def compute_unit_sphere_transform(mesh):
    """
    returns translation and scale, which is applied to meshes before computing their SDF cloud
    """
    # the transformation applied by mesh_to_sdf.scale_to_unit_sphere(mesh)
    translation = -mesh.bounding_box.centroid
    scale = 1 / np.max(np.linalg.norm(mesh.vertices + translation, axis=1))
    return translation, scale


# mesh1 = trimesh.load("/mnt/HDD1/Gary/meshSDF/experiments/pelvis/Reconstructions/latest/true_dataset6_CLINIC_0033_mask_4label_part_1/predicted.ply")
# coord1 = mesh1.vertices




# mesh2 = trimesh.load("/mnt/HDD1/Gary/meshSDF/data/pelvis/meshes/true_dataset6_CLINIC_0013_mask_4label_part_1.obj")
# coord2 = mesh2.vertices
# faces2 = mesh2.faces
#
# translation, scale = compute_unit_sphere_transform(mesh2)
# coord2 = coord2 - translation/scale
# lib.mesh.write_verts_faces_to_file(coord2, faces2, "013_resized_normalized_1.ply")



# mesh = trimesh.load("/mnt/HDD1/Gary/meshSDF/data/pelvis/meshes/true_dataset6_CLINIC_0013_mask_4label_part_1.obj")
# voxel_grid = trimesh.voxel.creation.voxelize(mesh,1/128)
# voxel_grid = voxel_grid.matrix
# print(voxel_grid.shape)
# plt.imshow(voxel_grid[:,35,:])
# plt.show()


# mesh = o3d.io.read_triangle_mesh("/mnt/HDD1/Gary/meshSDF/data/pelvis/meshes/true_dataset6_CLINIC_0013_mask_4label_part_1.obj")
# voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
#                                                               voxel_size=0.1)
# voxels = voxel_grid.get_voxels()  # returns list of voxels
# indices = np.stack(list(vx.grid_index for vx in voxels))
#
# voxels = voxel_grid.get_voxels()  # returns list of voxels
# indices = np.stack(list(vx.grid_index for vx in voxels))
# print(indices)


mesh_path = ""
sdf_path = ""
target_path = ""
gt_path = ""

meshes = [os.path.join(mesh_path, dir) for dir in os.listdir(mesh_path)]
sdfs = [os.path.join(sdf_path, dir) for dir in os.listdir(sdf_path)]
meshes.sort()
sdfs.sort()

for i in range(len(sdfs)):
    mesh = trimesh.load(meshes[i])
    sdf = np.load(sdfs[i])
    pos_tensor = remove_nans(torch.from_numpy(sdf["pos"].astype(float)))
    neg_tensor = remove_nans(torch.from_numpy(sdf["neg"].astype(float)))
    translation, scale = compute_unit_sphere_transform(mesh)
    new_pos = torch.zeros_like(pos_tensor)
    new_pos[:,:3] = pos_tensor[:,:3] / scale - translation
    new_pos[:,3] = pos_tensor[:,3] / scale
    # new_pos[:, :3] = torch.flip(new_pos[:,:3],[1,])
    # pos_file = new_pos.tofile('pos')

    new_neg = torch.zeros_like(neg_tensor)
    new_neg[:,:3] = neg_tensor[:,:3] / scale - translation
    new_neg[:,3] = neg_tensor[:,3] / scale
    # new_neg[:, :3] = torch.flip(new_neg[:,:3],[1,])
    # neg_file = new_neg.tofile('neg')
    file_name = os.path.join(target_path, sdfs[i].split("/")[-1])
    np.savez(file_name,pos=new_pos,neg=new_neg)
    print(file_name)