# -*- encoding: utf-8 -*-
import copy
import math

import cv2
import numpy as np


def bbox_decode(heat, wh, reg=None, K=100):
    """bbox组成：[V1, V2, V3, V4]
    V1~V4: bbox的4个坐标点
    """
    batch = heat.shape[0]
    heat, keep = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.reshape(batch, K, 2)
        xs = xs.reshape(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.reshape(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.reshape(batch, K, 1) + 0.5
        ys = ys.reshape(batch, K, 1) + 0.5

    wh = _tranpose_and_gather_feat(wh, inds)
    wh = wh.reshape(batch, K, 8)
    clses = clses.reshape(batch, K, 1).astype(np.float32)
    scores = scores.reshape(batch, K, 1)

    bboxes = np.concatenate(
        [
            xs - wh[..., 0:1],
            ys - wh[..., 1:2],
            xs - wh[..., 2:3],
            ys - wh[..., 3:4],
            xs - wh[..., 4:5],
            ys - wh[..., 5:6],
            xs - wh[..., 6:7],
            ys - wh[..., 7:8],
        ],
        axis=2,
    )
    detections = np.concatenate([bboxes, scores, clses], axis=2)
    return detections, inds


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = max_pool(heat, kernel_size=kernel, stride=1, padding=pad)
    keep = hmax == heat
    return heat * keep, keep


def max_pool(img, kernel_size, stride, padding):
    h, w = img.shape[2:]
    img = np.pad(
        img,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        "constant",
        constant_values=0,
    )

    res_h = ((h + 2 - kernel_size) // stride) + 1
    res_w = ((w + 2 - kernel_size) // stride) + 1
    res = np.zeros((img.shape[0], img.shape[1], res_h, res_w))
    for i in range(res_h):
        for j in range(res_w):
            temp = img[
                :,
                :,
                i * stride : i * stride + kernel_size,
                j * stride : j * stride + kernel_size,
            ]
            res[:, :, i, j] = temp.max()
    return res


def _topk(scores, K=40):
    batch, cat, height, width = scores.shape

    topk_scores, topk_inds = find_topk(scores.reshape(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = topk_inds / width
    topk_xs = np.float32(np.int32(topk_inds % width))

    topk_score, topk_ind = find_topk(topk_scores.reshape(batch, -1), K)
    topk_clses = np.int32(topk_ind / K)
    topk_inds = _gather_feat(topk_inds.reshape(batch, -1, 1), topk_ind).reshape(
        batch, K
    )
    topk_ys = _gather_feat(topk_ys.reshape(batch, -1, 1), topk_ind).reshape(batch, K)
    topk_xs = _gather_feat(topk_xs.reshape(batch, -1, 1), topk_ind).reshape(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def find_topk(a, k, axis=-1, largest=True, sorted=True):
    if axis is None:
        axis_size = a.size
    else:
        axis_size = a.shape[axis]
    assert 1 <= k <= axis_size

    a = np.asanyarray(a)
    if largest:
        index_array = np.argpartition(a, axis_size - k, axis=axis)
        topk_indices = np.take(index_array, -np.arange(k) - 1, axis=axis)
    else:
        index_array = np.argpartition(a, k - 1, axis=axis)
        topk_indices = np.take(index_array, np.arange(k), axis=axis)

    topk_values = np.take_along_axis(a, topk_indices, axis=axis)
    if sorted:
        sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
        if largest:
            sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)

        sorted_topk_values = np.take_along_axis(
            topk_values, sorted_indices_in_topk, axis=axis
        )
        sorted_topk_indices = np.take_along_axis(
            topk_indices, sorted_indices_in_topk, axis=axis
        )
        return sorted_topk_values, sorted_topk_indices
    return topk_values, topk_indices


def _gather_feat(feat, ind):
    dim = feat.shape[2]
    ind = np.broadcast_to(ind[:, :, None], (ind.shape[0], ind.shape[1], dim))
    feat = _gather_np(feat, 1, ind)
    return feat


def _gather_np(data, dim, index):
    """
    Gathers values along an axis specified by dim.
    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    :param dim: The axis along which to index
    :param index: A tensor of indices of elements to gather
    :return: tensor of gathered values
    """
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1 :]
    data_xsection_shape = data.shape[:dim] + data.shape[dim + 1 :]
    if idx_xsection_shape != data_xsection_shape:
        raise ValueError(
            "Except for dimension "
            + str(dim)
            + ", all dimensions of index and data should be the same size"
        )

    if index.dtype != np.int64:
        raise TypeError("The values of index must be integers")

    data_swaped = np.swapaxes(data, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.take_along_axis(data_swaped, index_swaped, axis=0)
    return np.swapaxes(gathered, 0, dim)


def _tranpose_and_gather_feat(feat, ind):
    feat = np.ascontiguousarray(np.transpose(feat, [0, 2, 3, 1]))
    feat = feat.reshape(feat.shape[0], -1, feat.shape[3])
    feat = _gather_feat(feat, ind)
    return feat


def gbox_decode(mk, st_reg, reg=None, K=400):
    """gbox的组成：[V1, P1, P2, P3, P4]
    P1~P4: 四个框的中心点
    V1: 四个框的交点
    """
    batch = mk.shape[0]
    mk, keep = _nms(mk)
    scores, inds, clses, ys, xs = _topk(mk, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.reshape(batch, K, 2)
        xs = xs.reshape(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.reshape(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.reshape(batch, K, 1) + 0.5
        ys = ys.reshape(batch, K, 1) + 0.5

    scores = scores.reshape(batch, K, 1)
    clses = clses.reshape(batch, K, 1).astype(np.float32)
    st_Reg = _tranpose_and_gather_feat(st_reg, inds)

    bboxes = np.concatenate(
        [
            xs - st_Reg[..., 0:1],
            ys - st_Reg[..., 1:2],
            xs - st_Reg[..., 2:3],
            ys - st_Reg[..., 3:4],
            xs - st_Reg[..., 4:5],
            ys - st_Reg[..., 5:6],
            xs - st_Reg[..., 6:7],
            ys - st_Reg[..., 7:8],
        ],
        axis=2,
    )
    return np.concatenate([xs, ys, bboxes, scores, clses], axis=2), keep


def transform_preds(coords, center, scale, output_size, rot=0):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, rot, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def bbox_post_process(bbox, c, s, h, w):
    for i in range(bbox.shape[0]):
        bbox[i, :, 0:2] = transform_preds(bbox[i, :, 0:2], c[i], s[i], (w, h))
        bbox[i, :, 2:4] = transform_preds(bbox[i, :, 2:4], c[i], s[i], (w, h))
        bbox[i, :, 4:6] = transform_preds(bbox[i, :, 4:6], c[i], s[i], (w, h))
        bbox[i, :, 6:8] = transform_preds(bbox[i, :, 6:8], c[i], s[i], (w, h))
    return bbox


def gbox_post_process(gbox, c, s, h, w):
    for i in range(gbox.shape[0]):
        gbox[i, :, 0:2] = transform_preds(gbox[i, :, 0:2], c[i], s[i], (w, h))
        gbox[i, :, 2:4] = transform_preds(gbox[i, :, 2:4], c[i], s[i], (w, h))
        gbox[i, :, 4:6] = transform_preds(gbox[i, :, 4:6], c[i], s[i], (w, h))
        gbox[i, :, 6:8] = transform_preds(gbox[i, :, 6:8], c[i], s[i], (w, h))
        gbox[i, :, 8:10] = transform_preds(gbox[i, :, 8:10], c[i], s[i], (w, h))
    return gbox


def nms(dets, thresh):
    if len(dets) < 2:
        return dets

    index_keep, keep = [], []
    for i in range(len(dets)):
        box = dets[i]
        if box[-1] < thresh:
            break

        max_score_index = -1
        ctx = (dets[i][0] + dets[i][2] + dets[i][4] + dets[i][6]) / 4
        cty = (dets[i][1] + dets[i][3] + dets[i][5] + dets[i][7]) / 4

        for j in range(len(dets)):
            if i == j or dets[j][-1] < thresh:
                break

            x1, y1 = dets[j][0], dets[j][1]
            x2, y2 = dets[j][2], dets[j][3]
            x3, y3 = dets[j][4], dets[j][5]
            x4, y4 = dets[j][6], dets[j][7]
            a = (x2 - x1) * (cty - y1) - (y2 - y1) * (ctx - x1)
            b = (x3 - x2) * (cty - y2) - (y3 - y2) * (ctx - x2)
            c = (x4 - x3) * (cty - y3) - (y4 - y3) * (ctx - x3)
            d = (x1 - x4) * (cty - y4) - (y1 - y4) * (ctx - x4)
            if all(x > 0 for x in (a, b, c, d)) or all(x < 0 for x in (a, b, c, d)):
                if dets[i][8] > dets[j][8] and max_score_index < 0:
                    max_score_index = i
                elif dets[i][8] < dets[j][8]:
                    max_score_index = -2
                    break

        if max_score_index > -1:
            index_keep.append(max_score_index)
        elif max_score_index == -1:
            index_keep.append(i)

    keep = [dets[index_keep[i]] for i in range(len(index_keep))]
    return np.array(keep)


def group_bbox_by_gbox(
    bboxes, gboxes, score_thred=0.3, v2c_dist_thred=2, c2v_dist_thred=0.5
):
    def point_in_box(box, point):
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        x3, y3, x4, y4 = box[4], box[5], box[6], box[7]
        ctx, cty = point[0], point[1]
        a = (x2 - x1) * (cty - y1) - (y2 - y1) * (ctx - x1)
        b = (x3 - x2) * (cty - y2) - (y3 - y2) * (ctx - x2)
        c = (x4 - x3) * (cty - y3) - (y4 - y3) * (ctx - x3)
        d = (x1 - x4) * (cty - y4) - (y1 - y4) * (ctx - x4)
        if all(x > 0 for x in (a, b, c, d)) or all(x < 0 for x in (a, b, c, d)):
            return True
        return False

    def get_distance(pt1, pt2):
        return math.sqrt(
            (pt1[0] - pt2[0]) * (pt1[0] - pt2[0])
            + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1])
        )

    dets = copy.deepcopy(bboxes)
    sign = np.zeros((len(dets), 4))

    for gbox in gboxes:
        if gbox[10] < score_thred:
            break

        vertex = [gbox[0], gbox[1]]
        for i in range(4):
            center = [gbox[2 * i + 2], gbox[2 * i + 3]]
            if get_distance(vertex, center) < v2c_dist_thred:
                continue

            for k, bbox in enumerate(dets):
                if bbox[8] < score_thred:
                    break

                if sum(sign[k]) == 4:
                    continue

                w = (abs(bbox[6] - bbox[0]) + abs(bbox[4] - bbox[2])) / 2
                h = (abs(bbox[3] - bbox[1]) + abs(bbox[5] - bbox[7])) / 2
                m = max(w, h)
                if point_in_box(bbox, center):
                    min_dist, min_id = 1e4, -1
                    for j in range(4):
                        dist = get_distance(vertex, [bbox[2 * j], bbox[2 * j + 1]])
                        if dist < min_dist:
                            min_dist = dist
                            min_id = j

                    if (
                        min_id > -1
                        and min_dist < c2v_dist_thred * m
                        and sign[k][min_id] == 0
                    ):
                        bboxes[k][2 * min_id] = vertex[0]
                        bboxes[k][2 * min_id + 1] = vertex[1]
                        sign[k][min_id] = 1
    return bboxes
