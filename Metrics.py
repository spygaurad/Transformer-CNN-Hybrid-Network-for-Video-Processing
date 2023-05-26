import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim

class DiceLoss(nn.Module):

    def __init__(self, num_classes=8):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0
        self.classes = 3
        self.ignore_index = None
        self.eps = 1e-7

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc




class JaccardScore(nn.Module):

    def __init__(self):
        super(JaccardScore, self).__init__()
    
    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = torch.logical_and(y_true, y_pred)
        union = torch.logical_or(y_true, y_pred)
        iou_score = torch.sum(intersection) / torch.sum(union)  
        return iou_score




class MixedLoss(nn.Module):
  def __init__(self, alpha, beta):
    super(MixedLoss, self).__init__()
    self.alpha = alpha
    self.beta = beta

  def forward(self, y_pred, y_true):
    # y_pred and y_true are of shape (batch_size, channels, height, width)
    # compute the MS-SSIM loss
    msssim_loss = 1 - ms_ssim(y_pred, y_true)
    # compute the L2 loss
    l2_loss = nn.MSELoss()(y_pred, y_true)
    # return the mixed loss
    return self.alpha*msssim_loss + self.beta*l2_loss