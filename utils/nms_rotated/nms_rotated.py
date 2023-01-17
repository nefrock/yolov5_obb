import time

import torch
from numpy import pi

if __name__.startswith('utils'):
    from .nms_rotated_wrapper import obb_nms
else:
    from nms_rotated_wrapper import obb_nms


def non_max_suppression_obb(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=1500):
    """Runs Non-Maximum Suppression (NMS) on inference results_obb
    Args:
        prediction (tensor): (b, n_all_anchors, [cx cy l s obj num_cls theta_cls])
        agnostic (bool): True = NMS will be applied between elements of different categories
        labels : () or

    Returns:
        list of detections, len=batch_size, on (n,7) tensor per image [xylsθ, conf, cls] θ ∈ [-pi/2, pi/2)
    """

    nc = prediction.shape[2] - 5 - 180  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    class_index = nc + 5

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    max_wh = 4096 # min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 30.0  # seconds to quit after
    # redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros((0, 7), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence, (tensor): (n_conf_thres, [cx cy l s obj num_cls theta_cls])

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:class_index] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        _, theta_pred = torch.max(x[:, class_index:], 1,  keepdim=True) # [n_conf_thres, 1] θ ∈ int[0, 179]
        theta_pred = (theta_pred - 90) / 180 * pi # [n_conf_thres, 1] θ ∈ [-pi/2, pi/2)

        # Detections matrix nx7 (xyls, θ, conf, cls) θ ∈ [-pi/2, pi/2)
        if multi_label:
            i, j = (x[:, 5:class_index] > conf_thres).nonzero(as_tuple=False).T # ()
            x = torch.cat((x[i, :4], theta_pred[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:class_index].max(1, keepdim=True)
            x = torch.cat((x[:, :4], theta_pred, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 6:7] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 5].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 6:7] * (0 if agnostic else max_wh)  # classes
        rboxes = x[:, :5].clone()
        rboxes[:, :2] = rboxes[:, :2] + c # rboxes (offset by class)
        scores = x[:, 5]  # scores
        _, i = obb_nms(rboxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output
