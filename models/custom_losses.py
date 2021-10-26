from numpy import nan
import  torch
import torch.nn as nn

class MaskedL1(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_im, target_im, mask, num_ch):
        if num_ch > 1:
            mask = torch.cat([mask for _ in range(num_ch)], dim=1)
        neg_mask = torch.logical_not(mask)
        
        diff = torch.abs(input_im - target_im)
        
        diff_in = diff[mask]
        # loss_in = torch.sum(diff_in) / torch.numel(diff_in)
        loss_in = torch.sum(diff_in) / torch.numel(diff)
        loss_in = torch.nan_to_num(loss_in)
            
        diff_out = diff[neg_mask]
        # loss_out = torch.sum(diff_out) / torch.count_nonzero(diff_out)
        loss_out = torch.sum(diff_out) / torch.numel(diff)
        loss_out = torch.nan_to_num(loss_out)
            
        return loss_in, loss_out
    
class BackgroundColor(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, fake_im, target_im, mask):
        mask = torch.cat([mask, mask, mask], 1)
        target_mean = torch.mean(target_im[mask], 1)
        fake_mean = torch.mean(fake_im[mask], 1)
        return torch.abs(target_mean, fake_mean) / len(fake_mean)
