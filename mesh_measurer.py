import math

import numpy as np
import surface_distance
from surface_distance.surface_distance import metrics
import nibabel as nib
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
from os import listdir
import os
import trimesh

import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import (
    knn_points,
    knn_gather,
    sample_points_from_meshes
)
from pytorch3d.io import load_obj, load_ply
from pytorch3d.structures import Meshes, Pointclouds
from scipy.spatial.distance import directed_hausdorff
from lib.metric_utils import _point_mesh_face_distance_unidirectional

from lib.loss import dice_coeff



def chamfer_directed(A, B):


    N1 = A.shape[1]
    N2 = B.shape[1]

    if N1 > 0 and N2 > 0:
        y1 = A[:, :, None].repeat(1, 1, N2, 1)
        y2 = B[:, None].repeat(1, N1, 1, 1)

        diff = torch.sum((y1 - y2) ** 2, dim=3)

        loss, _ = torch.min(diff, dim=2)

        loss = torch.mean(loss)
    else:
        loss = torch.Tensor([float("Inf")]).cuda() if A.is_cuda else torch.Tensor([float("Inf")])

    return loss

def chamfer_symmetric(A, B):
    N1 = A.shape[1]
    N2 = B.shape[1]
    y1 = A[:, :, None].repeat(1, 1, N2, 1)
    y2 = B[:, None].repeat(1, N1, 1, 1)

    diff = torch.sum((y1 - y2) ** 2, dim=3)

    loss1, _ = torch.min(diff, dim=1)
    loss2, _ = torch.min(diff, dim=2)

    loss = torch.sum(loss1) + torch.sum(loss2)
    return loss

def chamfer_weighted_symmetric(A, B):

    N1 = A.shape[1]
    N2 = B.shape[1]
    y1 = A[:, :, None].repeat(1, 1, N2, 1)
    y2 = B[:, None].repeat(1, N1, 1, 1)

    diff = torch.sum((y1 - y2) ** 2, dim=3)

    loss1, _ = torch.min(diff, dim=1)
    loss2, _ = torch.min(diff, dim=2)
    loss = torch.mean(loss1) + torch.mean(loss2)
    return loss

def iterative_chamfer_symmetric(A,B):
    diff = []
    A = A.cuda(0)*511
    B = B.cuda(0)*511
    for i in range(A.shape[0]):
        p = A[i].unsqueeze(0)

        pX = p.repeat(B.shape[0],1)
        d = (pX-B)**2
        d = torch.sqrt(torch.min(d))
        diff.append(d)
    return sum(diff)/len(diff)


def chamfer_weighted_symmetric_with_dtf(A, B, B_dtf):

    N1 = A.shape[1]
    N2 = B.shape[1]
    y1 = A[:, :, None].repeat(1, 1, N2, 1)
    y2 = B[:, None].repeat(1, N1, 1, 1)

    diff = torch.sum((y1 - y2) ** 2, dim=3)
    loss1, _ = torch.min(diff, dim=1)
    # loss2, _ = torch.min(diff, dim=2)
    A_ = A[:, :, None, None]
    loss2 = F.grid_sample(B_dtf, A_, mode='bilinear', padding_mode='border', align_corners=True)

    loss = torch.mean(loss1) + torch.mean(loss2)
    return loss



def AverageDistanceScore(pred, gt, n_m_classes,
                         padded_coordinates=(-1.0, -1.0, -1.0)):
    """ Compute point-to-mesh distance between prediction and ground truth. """

    padded_coordinates = torch.Tensor(padded_coordinates).cuda()

    # Ground truth

    # Back to original coordinate space
    gt_vertices, gt_faces= gt.vertices, gt.faces
    ndims = gt_vertices.shape[-1]

    # Prediction: Only consider mesh of last step
    pred_vertices, pred_faces = pred.vertices, pred.faces
    # pred_vertices = pred_vertices[-1].view(n_m_classes, -1, ndims)
    # pred_faces = pred_faces[-1].view(n_m_classes, -1, ndims)

    # Iterate over structures
    assd_all = []
    for pred_v, pred_f, gt_v, gt_f in zip(
        pred_vertices,
        pred_faces,
        gt_vertices.cuda(),
        gt_faces.cuda()
    ):

        # Prediction
        pred_mesh = Meshes([pred_v], [pred_f])
        pred_pcl = sample_points_from_meshes(pred_mesh, 100000)
        pred_pcl = Pointclouds(pred_pcl)

        # Remove padded vertices from gt
        gt_v = gt_v[~torch.isclose(gt_v, padded_coordinates).all(dim=1)]
        gt_mesh = Meshes([gt_v], [gt_f])
        gt_pcl = sample_points_from_meshes(gt_mesh, 100000)
        gt_pcl = Pointclouds(gt_pcl)

        # Compute distance
        P2G_dist = _point_mesh_face_distance_unidirectional(
            gt_pcl, pred_mesh
        ).cpu().numpy()
        G2P_dist = _point_mesh_face_distance_unidirectional(
            pred_pcl, gt_mesh
        ).cpu().numpy()

        assd2 = (P2G_dist.sum() + G2P_dist.sum()) / float(P2G_dist.shape[0] + G2P_dist.shape[0])

        assd_all.append(assd2)

    return assd_all


evals_path = "/media/gary/0E5A9BCD5A9BB047/Gary/projects/meshSDF/experiments/Xwitch-demo-MSD_T10/Reconstructions/latest"
gts_path = "/media/gary/0E5A9BCD5A9BB047/Gary/projects/meshSDF/data/MSD_T10/meshes-cropped"

evals = [file for file in listdir(evals_path) if ".ply" in file]
print(evals)

assd = []
hausdorff = []
chamfer = []
for path in evals:
    print(path)
    # gt_path = os.path.join(gts_path, path.replace("pred", "true"))
    gt_path = os.path.join(gts_path, path.replace(".nii.gzpredicted.ply", ".obj"))
    # gt_path = os.path.join(gts_path, path.replace("predicted.ply", ".obj"))

    eval_path = os.path.join(evals_path, path)
    # pred_verts, pred_faces, gt_aux = load_obj(eval_path)
    # pred_mesh = Meshes(verts=[pred_verts], faces=[pred_faces.verts_idx]).cuda()
    pred_verts, pred_faces = load_ply(eval_path)
    pred_mesh = Meshes(verts=[pred_verts], faces=[pred_faces]).cuda()


    gt_verts, gt_faces, gt_aux = load_obj(gt_path)
    gt_mesh = Meshes(verts=[gt_verts], faces=[gt_faces.verts_idx]).cuda()

    pred_pcl = sample_points_from_meshes(pred_mesh, 10000)
    pred_pcl = Pointclouds(pred_pcl)
    gt_pcl = sample_points_from_meshes(gt_mesh, 10000)
    gt_pcl = Pointclouds(gt_pcl)

    # Compute distance
    P2G_dist = _point_mesh_face_distance_unidirectional(
        gt_pcl, pred_mesh
    )
    G2P_dist = _point_mesh_face_distance_unidirectional(
        pred_pcl, gt_mesh
    )
    assd2 = (P2G_dist.sum() + G2P_dist.sum()) / float(P2G_dist.shape[0] + G2P_dist.shape[0])
    assd.append(assd2.cpu().numpy())

    d = max(directed_hausdorff(pred_verts, gt_verts)[0], directed_hausdorff(gt_verts, pred_verts)[0])
    hausdorff.append(d)
    n_pred_verts =pred_verts.unsqueeze(0)
    n_gt_verts = gt_verts.unsqueeze(0)
    cd = chamfer_distance(n_pred_verts.cuda(), n_gt_verts.cuda())[0].cpu().item()
    chamfer.append(cd)


#
#
# evals_path = "/media/gary/0E5A9BCD5A9BB047/Gary/projects/voxel2mesh-master/experiments/MSD_T10Experiment_005/trial_3/best_performance3/mesh"
# gts_path = "/media/gary/0E5A9BCD5A9BB047/Gary/projects/voxel2mesh-master/experiments/MSD_T10Experiment_005/trial_3/best_performance3/mesh"
#
# evals = [file for file in listdir(evals_path) if "pred" in file]
# print(evals)
#
# assd = []
# hausdorff = []
# chamfer = []
# for path in evals:
#     print(path)
#     gt_path = os.path.join(gts_path, path.replace("pred", "true"))
#     eval_path = os.path.join(evals_path, path)
#     pred_verts, pred_faces, gt_aux = load_obj(eval_path)
#     pred_mesh = Meshes(verts=[pred_verts], faces=[pred_faces.verts_idx]).cuda()
#     # pred_verts, pred_faces = load_ply(eval_path)
#     # pred_mesh = Meshes(verts=[pred_verts], faces=[pred_faces]).cuda()
#
#
#     gt_verts, gt_faces, gt_aux = load_obj(gt_path)
#     gt_mesh = Meshes(verts=[gt_verts], faces=[gt_faces.verts_idx]).cuda()
#
#     pred_pcl = sample_points_from_meshes(pred_mesh, 10000)
#     pred_pcl = Pointclouds(pred_pcl)
#     gt_pcl = sample_points_from_meshes(gt_mesh, 10000)
#     gt_pcl = Pointclouds(gt_pcl)
#
#     # Compute distance
#     P2G_dist = _point_mesh_face_distance_unidirectional(
#         gt_pcl, pred_mesh
#     )
#     G2P_dist = _point_mesh_face_distance_unidirectional(
#         pred_pcl, gt_mesh
#     )
#     assd2 = (P2G_dist.sum() + G2P_dist.sum()) / float(P2G_dist.shape[0] + G2P_dist.shape[0])
#     assd.append(assd2.cpu().numpy())
#
#     d = max(directed_hausdorff(pred_verts, gt_verts)[0], directed_hausdorff(gt_verts, pred_verts)[0])
#     hausdorff.append(d)
#     n_pred_verts =pred_verts.unsqueeze(0)
#     n_gt_verts = gt_verts.unsqueeze(0)
#     cd = chamfer_distance(n_pred_verts.cuda(), n_gt_verts.cuda())[0].cpu().item()
#     chamfer.append(cd)


avg_assd = sum(assd)/len(assd)
avg_hausdorff = sum(hausdorff)/len(hausdorff)
avg_chamfer = sum(chamfer)/len(chamfer)
print(avg_assd)

print(avg_hausdorff)

print(avg_chamfer)






