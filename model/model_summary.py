import torch
from torchsummary import summary
import os

def model_summary(model_name):
    device = torch.device('cuda:0')
    if(model_name == 'ESPCN'):
        from ESPCN import ESPCN
        net = ESPCN(n_colors=3, scale=2)
        net.to(device)
        print(summary(net, (3, 240, 360),device='cuda'))

    elif(model_name == 'ESPCN_modified'):
        from ESPCN_modified import ESPCN_modified
        net = ESPCN_modified(n_colors=3, scale=2)
        net.to(device)
        print(summary(net, (3, 240, 360),device='cuda'))

    elif(model_name == 'ESPCN_multiframe'):
        from ESPCN_multiframe import ESPCN_multiframe
        net = ESPCN_multiframe(n_colors=3, scale=2, n_sequence=3)
        net.to(device)
        print(summary(net, (3, 3, 240, 360),device='cuda'))
    else:
        pass

if __name__ =="__main__":
    # model_summary("ESPCN")
    # model_summary("ESPCN_modified")
    model_summary("ESPCN_multiframe")