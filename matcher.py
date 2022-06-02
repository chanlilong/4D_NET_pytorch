# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from box_ops import box_cxcywh_to_xyxy, generalized_box_iou
import sys
sys.path.append("/home/conda/RAID_5_14TB/Rotated_IoU/")
sys.path.append("/home/conda/RAID_5_14TB/mmdetection3d/")
# from mmdet3d.ops.iou3d import boxes_iou_bev
from mmdet3d.core.bbox.iou_calculators import bbox_overlaps_3d 
from oriented_iou_loss import cal_diou,cal_diou_3d
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 8,
                 cost_giou: float = 3):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        
    def convert_to_axisaligned(self,xywlr):
        axis_aligned_bboxes = []


        for bbox in xywlr:
            x,y,w_pix,l_pix,r = bbox

            corners = torch.tensor([[ -0.5*w_pix, -0.5*l_pix],
                                    [  0.5*w_pix, -0.5*l_pix],
                                    [  0.5*w_pix,  0.5*l_pix],
                                    [ -0.5*w_pix,  0.5*l_pix]]).to(xywlr.device)

            #============[ROTATE]==========#
            c, s = torch.cos(r), torch.sin(r)
            R = torch.tensor(((c, -s), (s, c))).to(xywlr.device)
            corners_rot = torch.matmul(R,corners.T).T

            #=====[TRANSLATE]====#
            corners_rot[:,0] += x
            corners_rot[:,1] += y

            xyxy = torch.tensor([corners_rot[:,0].min(),corners_rot[:,1].min(),corners_rot[:,0].max(),corners_rot[:,1].max()],dtype=torch.float32).to(xywlr.device)
            axis_aligned_bboxes.append(xyxy.reshape(1,-1))
            
        return torch.cat(axis_aligned_bboxes,0).to(xywlr.device)

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])
            
            #print(tgt_ids.shape,tgt_bbox.shape)
            # Compute the classification cost.
            alpha = 0.25
            gamma = 2
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            #cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
            #                                 box_cxcywh_to_xyxy(tgt_bbox))
            #out_bbox -> x,y,z,l,w,h,r
            
            
            # xyxy1 = box_cxcywh_to_xyxy(torch.cat([out_bbox[...,0:2],out_bbox[...,3:5]],-1).view(-1,4))
            # xyxy2 = box_cxcywh_to_xyxy(torch.cat([tgt_bbox[...,0:2],tgt_bbox[...,3:5]],-1).view(-1,4))
            # xyxyr1 = torch.cat([xyxy1,(out_bbox[...,6:]*2-1)*np.pi],-1).view(-1,5)
            # xyxyr2 = torch.cat([xyxy2,(tgt_bbox[...,6:]*2-1)*np.pi],-1).view(-1,5)
            # cost_giou = -boxes_iou_bev(xyxyr1,xyxyr2)
            
            # b1 = torch.cat([out_bbox[...,0:6],(out_bbox[...,6:7]*2-1)*np.pi],-1) #x,y,z,w,l,h,r
            # b2 = torch.cat([tgt_bbox[...,0:6],(tgt_bbox[...,6:7]*2-1)*np.pi],-1) 
            
            try:
                b1 = torch.cat([out_bbox[...,0:6],(out_bbox[...,6:7])],-1) #x,y,z,l,w,h,rz
                b2 = torch.cat([tgt_bbox[...,0:6],(tgt_bbox[...,6:7])],-1)  #"x","y","z","l","w","h","yaw"
                cost_giou = -bbox_overlaps_3d(b1,b2, coordinate='lidar')
                # cost_giou, _ = cal_diou_3d(b1.unsqueeze(0),b2.unsqueeze(0),enclosing_type="smallest")

                # + self.cost_giou * cost_giou
                C = self.cost_bbox * cost_bbox + self.cost_class * cost_class+ self.cost_giou * cost_giou
                C = C.view(bs, num_queries, -1).cpu()

                sizes = [len(v["boxes"]) for v in targets]
                # max_val = torch.maximum(cost_bbox,cost_class).max()
                # indices = [linear_sum_assignment(torch.nan_to_num(c[i],max_val,max_val,max_val)) for i, c in enumerate(C.split(sizes, -1))]
                indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
            except:
                print("hungarian giou match failed")
                # b1 = torch.cat([out_bbox[...,0:6],(out_bbox[...,6:7])],-1) #x,y,z,w,l,h,r
                # b2 = torch.cat([tgt_bbox[...,0:6],(tgt_bbox[...,6:7])],-1) 
                # cost_giou = -bbox_overlaps_3d(b1,b2, coordinate='lidar')

                # + self.cost_giou * cost_giou
                C = self.cost_bbox * cost_bbox + self.cost_class * cost_class
                C = C.view(bs, num_queries, -1).cpu()

                sizes = [len(v["boxes"]) for v in targets]
                indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

class Criterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, losses=['labels','cardinality','boxes'], focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher()
        self.losses = losses
        self.focal_alpha = focal_alpha
        
        self.weight_dict = {"loss_ce":self.matcher.cost_class,
                            "loss_bbox":self.matcher.cost_bbox,
                            "loss_giou":self.matcher.cost_giou}

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        
        target_classes_onehot = target_classes_onehot[:,:,:-1]
        
        
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        
        
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        #Bbox Loss
        loss_bbox = F.smooth_l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes


        #3DIOU Loss
        # b1 = torch.cat([src_boxes[...,0:3],src_boxes[...,4:6],src_boxes[...,3:4],(src_boxes[...,-1:]*2-1)*np.pi],-1) #x,y,z,w,h,l,r
        # b2 = torch.cat([target_boxes[...,0:3],target_boxes[...,4:6],target_boxes[...,3:4],(target_boxes[...,-1:]*2-1)*np.pi],-1) #x,y,z,w,h,l,r
        #x0,y1,z2,l3,w4,h5,r6 
        b1 = torch.cat([src_boxes[...,0:3],src_boxes[...,4:6],src_boxes[...,3:4],src_boxes[...,-1:]],-1) #x,y,z,w,h,l,r
        b2 = torch.cat([target_boxes[...,0:3],target_boxes[...,4:6],target_boxes[...,3:4],target_boxes[...,-1:]],-1) #x,y,z,w,h,l,r
        loss_giou, _ = cal_diou_3d(b1.unsqueeze(0),b2.unsqueeze(0),enclosing_type="smallest") #for 3D GIOU (x,y,z,w,h,l,alpha) shape: B,N,7
        
#         b1 = torch.cat([src_boxes[...,0:6],(src_boxes[...,6:7])],-1) #x,y,z,l,w,h,rz
#         b2 = torch.cat([target_boxes[...,0:6],(target_boxes[...,6:7])],-1) 
#         loss_giou = 1-torch.diag(bbox_overlaps_3d(b1,b2, coordinate='lidar'))
        
        #2DIOU Loss
        # xyxy1 = box_cxcywh_to_xyxy(torch.cat([src_boxes[...,0:2],src_boxes[...,3:5]],-1).view(-1,4))
        # xyxy2 = box_cxcywh_to_xyxy(torch.cat([target_boxes[...,0:2],target_boxes[...,3:5]],-1).view(-1,4))
        # xyxyr1 = torch.cat([xyxy1,(src_boxes[...,6:]*2-1)*np.pi],-1).view(-1,5)
        # xyxyr2 = torch.cat([xyxy2,(target_boxes[...,6:]*2-1)*np.pi],-1).view(-1,5)
        # loss_bev = 1-torch.diag(boxes_iou_bev(xyxyr1,xyxyr2))
        
        
        losses['loss_giou'] = ((loss_giou.sum() / num_boxes) )
        
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        return losses