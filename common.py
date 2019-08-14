
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
#import PIL
#import cv2
import math
from segmentor_v1 import DenseNet
from metrics import dice

##Select the Nvidia card
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'1'
#from torch.autograd import Variable
##----------------------------------Common Settings----------------------------
##Fix seed to reproduce result
#random.seed(1234)
#torch.manual_seed(1234)

##Network setting
pre_trained=True
##Optimization
num_epoch = 20000
lr_S = 2e-4
lr_D = 2e-5
momentum_S=0.9
momentum_D=0.9
step_size_S = 5000
step_size_D = 5000
beta1=0.9
beta2=0.999
batch_train = 4
##CUDNN
cudnn.enabled = True
cudnn.benchmark=True

##Data setting
#xdim = 164
#ydim = 144
#zdim = 192
data_dm = 2
ignore_label = 9
num_classes= 4
crop_size = (64, 64, 64)
## Note
checkpoint_name= 'model_3d_denseseg_v1'
note_S='Seg_3ddenseseg(Adam lr_S: ' + str(lr_S) + ',w_decay:1e-4' + 'beta:' +str(beta1)+ ',' + str(beta2) + ',' + 'step:' + str(step_size_S) + ' , lr_step)'
note_D='Seg_3ddenseseg(Adam lr_S: ' + str(lr_S) + ',w_decay:1e-4' + 'beta:' +str(beta1)+ ',' + str(beta2) + ',' + 'step:' + str(step_size_S) + ' , lr_step)'

num_checkpoint='20000'
note= str(num_checkpoint) +'_' + checkpoint_name
#Testing
checkpoint='./checkpoints/'+str(num_checkpoint) +'_' + checkpoint_name + '.pth'

#---------------------------------------------------------------------------------
print('@%s:  ' % os.path.basename(__file__))

if 1:
    torch.backends.cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = True
    print ('\tset cuda environment')
    print ('\t\ttorch.__version__              =', torch.__version__)
    print ('\t\ttorch.version.cuda             =', torch.version.cuda)
    print ('\t\ttorch.backends.cudnn.version() =', torch.backends.cudnn.version())
    try:
        print ('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =',os.environ['CUDA_VISIBLE_DEVICES'])
        NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    except Exception:
        print ('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =','None')
        NUM_CUDA_DEVICES = 1

    print ('\t\ttorch.cuda.device_count()      =', torch.cuda.device_count())
    print ('\t\ttorch.cuda.current_device()    =', torch.cuda.current_device())


print('')

#---------------------------------------------------------------------------------

##----------------------------------Common Functions----------------------------
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def make_one_hot(labels, num_classes):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x D x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x D x H x W, where C is class number. One-hot encoded.
    '''
    labels_extend=labels.clone()
    labels_extend.unsqueeze_(1)
    #labels_extend[labels_extend > num_classes] = num_classes
    one_hot = torch.cuda.FloatTensor(labels_extend.size(0), num_classes, labels_extend.size(2), labels_extend.size(3), labels_extend.size(4)).zero_()
    one_hot.scatter_(1, labels_extend, 1) #Copy 1 to one_hot at dim=1
    #target = one_hot[:, :num_classes]#ignore the ignored class
    return one_hot



def one_hot(labels):
    labels = labels.data.cpu().numpy()
    one_hot = np.zeros((labels.shape[0], num_classes, labels.shape[1], labels.shape[2],labels.shape[3]), dtype=labels.dtype)
    # handle ignore labels
    for class_id in range(num_classes):
        one_hot[:, class_id,...] = (labels==class_id)
    return torch.FloatTensor(one_hot)

def image_show(name, image, resize=5):
    H,W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 3D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor) * \
           (1 - abs(og[2] - center) / factor)

    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size),
                      dtype=np.float64)
    f = math.ceil(kernel_size / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(kernel_size):
        for j in range(kernel_size):
            for k in range(kernel_size):
                weight[0, 0, i, j, k] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c)) *  (1 - math.fabs(k / f - c))
    #weight[range(in_channels), range(out_channels), :, :, :] = filt
    for c in range(1, in_channels):
        weight[c, 0, :, :, :] = weight[0, 0, :, :, :]
    return torch.from_numpy(weight).float()

def fill_up_weights(up):
    w = up.weight.data
    #print (w)
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            for k in range(w.size(4)):
                w[0, 0, i, j, k] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))* (1 - math.fabs(k / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :,:] = w[0, 0, :, :,:]
    #print (w)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(lr_S, i_iter, num_epoch, 0.9)
    optimizer.param_groups[0]['lr'] = lr
    #if len(optimizer.param_groups) > 1 :
    #    optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(lr_D, i_iter, num_epoch, 0.9)
    optimizer.param_groups[0]['lr'] = lr
    #if len(optimizer.param_groups) > 1 :
     #   optimizer.param_groups[1]['lr'] = lr * 10

# def fill_up_weights(up):
#     w = up.weight.data
#     f = math.ceil(w.size(2) / 2)
#     c = (2 * f - 1 - f % 2) / (2. * f)
#     for i in range(w.size(2)):
#         for j in range(w.size(3)):
#             for k in range(w.size(4)):
#                 w[0, 0, i, j, k] = \
#                     (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c)) *  (1 - math.fabs(k / f - c))
#     for c in range(1, w.size(0)):
#         w[c, 0, :, :, :] = w[0, 0, :, :, :]
#     print (w)
