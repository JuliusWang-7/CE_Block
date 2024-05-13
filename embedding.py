import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import os

def local_pairwise_distances(model, x, y):
    """Computes pairwise squared l2 distances using a local search window.

    Optimized implementation using correlation_cost.

    Args:
    x: Float32 tensor of shape [batch, feature_dim, height, width].
    y: Float32 tensor of shape [batch, feature_dim, height, width].
    max_distance: Integer, the maximum distance in pixel coordinates
      per dimension which is considered to be in the search window.

    Returns:
    Float32 distances tensor of shape
      [batch, (2 * max_distance + 1) ** 2, height, width].
    """
    # d[i,j] = (x[i] - y[j]) * (x[i] - y[j])'
    # = sum(x[i]^2, -1) + sum(y[j]^2, -1) - 2 * x[i] * y[j]'

    corr = model.corr(x, y)
    xs = torch.sum(x * x, dim=1, keepdim=True)
    ys = torch.sum(y * y, dim=1, keepdim=True)
    ones_ys = torch.ones_like(ys)
    ys = model.corr(ones_ys, ys)
    d = xs.half() + ys - 2 * corr     # feature dist

    # add div
    d = d / d.shape[1]

    # Boundary should be set to Inf.
    boundary = torch.eq(model.corr(ones_ys, ones_ys), 0)

    d = torch.where(boundary, torch.ones_like(d).fill_(np.float('inf')), d)
    return d


def local_previous_frame_nearest_neighbor_features_per_object(model,
        prev_frame_embedding, query_embedding, prev_frame_labels,
        max_distance=9, save_cost=True, device=None):
    """Computes nearest neighbor features while only allowing local matches.

    Args:
      prev_frame_embedding: Tensor of shape [batch, embedding_dim, height, width],
        the embedding vectors for the last frame.
      query_embedding: Tensor of shape [batch, embedding_dim, height, width],
        the embedding vectors for the query frames.
      prev_frame_labels: Tensor of shape [batch, 1, height, width], the class labels of
        the previous frame.
      gt_ids: Int Tensor of shape [n_objs] of the sorted unique ground truth
        ids in the first frame.
      max_distance: Integer, the maximum distance allowed for local matching.

    Returns:
      nn_features: A float32 np.array of nearest neighbor features of shape
        [batch, (2*d+1)**2, height,width].
    """

    assert device is not None, "Device should not be none."

    if save_cost:
        d = local_pairwise_distances(model, query_embedding, prev_frame_embedding)  # shape:(batch, (2*d+1)**2, height, width)
    else:
        # Slow fallback in case correlation_cost is not available.
        pass
    d = (torch.sigmoid(d) - 0.5) * 2
    batch = prev_frame_embedding.size()[0]
    height = prev_frame_embedding.size()[-2]
    width = prev_frame_embedding.size()[-1]

    # Create offset versions of the mask.
    if save_cost:
        # New, faster code with cross-correlation via correlation_cost.
        # Due to padding we have to add 1 to the labels.
        # offset_labels = models.corr4(torch.ones((1, 1, height, width)).to(device), torch.unsqueeze((prev_frame_labels.permute((2, 0, 1)) + 1).to(device), 0))
        offset_labels = model.corr(torch.ones((batch, 1, height, width)).to(device), (prev_frame_labels + 1).to(device))
        # offset_labels = offset_labels.permute((2, 3, 1, 0))
        # Subtract the 1 again and round.
        offset_labels = torch.round(offset_labels - 1)
        offset_masks = torch.eq(offset_labels, 1).type(torch.uint8)
        # shape:(batch,(2*d+1)**2, height,width)
    else:
        # Slower code, without dependency to correlation_cost
        pass

    pad = torch.ones((batch, (2 * max_distance + 1) ** 2, height, width)).type(torch.half).to(device)
    # shape:(batch, (2*d+1)**2, height,width)
    d_tiled = d.half()  # shape:(batch, (2*d+1)**2, height,width)

    d_masked = torch.where(offset_masks, d_tiled, pad)
    dists = torch.min(d_masked, dim=1, keepdim=True)[0].float()  # shape:(batch, 1, height,width)
    return dists

def get_logits_with_matching(model,
                             features,
                             reference_labels,
                             ref_len,
                             correlation_window_size,
                             save_cost):
    height = features.size(2)
    width = features.size(3)

    embedding = model.embedding_conv2d(features)
    # shape:(batch, embedding_dimension(128), 128, 128)
    label = reference_labels
    # shape:(batch, 1, 128, 128)

    ref_embedding = embedding.clone()
    ref_embedding[ref_len:] = embedding[:-ref_len]
    ref_embedding[:ref_len] = embedding[ref_len: 2 * ref_len]

    ref_label = label.clone()
    ref_label[ref_len:] = label[:-ref_len]
    ref_label[:ref_len] = label[ref_len: 2 * ref_len]

    '''for debug'''
    # embedding = embedding[:4, 0, 0, 0]
    # print(embedding)
    # ref_embedding = embedding.clone()     # [-0.6149, -0.5467, -0.6859, -0.6736]
    # ref_embedding[ref_len:] = embedding[:-ref_len]
    # ref_embedding[:ref_len] = embedding[ref_len: 2 * ref_len]
    # print(ref_embedding)                  # [-0.5467, -0.6149, -0.5467, -0.6859]

    coor_info = local_previous_frame_nearest_neighbor_features_per_object(
        model,
        ref_embedding,
        embedding,
        ref_label,
        max_distance=correlation_window_size,
        save_cost=save_cost,
        device=model.device)

    features_n_concat = torch.cat([features, label, coor_info], dim=1)
    # shape:(batch,feat_dim(256) + 1 + 1, 128, 128)
    out_embedding = model.embedding_seg_conv2d(features_n_concat)
    # shape:(batch,embedding_dimension(128), 128, 128)

    return out_embedding, coor_info
