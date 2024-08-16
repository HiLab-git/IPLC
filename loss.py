#loss function for UPL-SFDA
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2

def curvature_loss(output):
    probability_map = output[:, 1, :, :]
    probability_map2 = output[:, 2, :, :]
    probability_map2 = probability_map + probability_map2
    probability_map3 = output[:, 3, :, :]
    sobel_x = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
    device = output.device
    sobel_x = sobel_x.to(device)
    curvature_loss = 0
    curvature_loss2 = 0
    curvature_loss3 = 0
    for depth in range(probability_map.size(0)):
        # 获取当前深度的概率分布
        prob_map_depth = probability_map[depth, :, :]
        prob_map_depth = F.conv2d(prob_map_depth.view(1, 1, 256, 256), sobel_x.view(1, 1, 3, 3))
        edge_intensity = torch.sqrt(prob_map_depth ** 2)
        gradient_x = F.conv2d(prob_map_depth, torch.Tensor([[-1, 0, 1]]).view(1, 1, 1, 3).to(device), padding=(0, 1))
        gradient_y = F.conv2d(prob_map_depth, torch.Tensor([[-1], [0], [1]]).view(1, 1, 3, 1).to(device), padding=(1, 0))
        H_xx = F.conv2d(gradient_x, torch.Tensor([[-1, 2, -1]]).view(1, 1, 1, 3).to(device), padding=(0, 1))
        H_yy = F.conv2d(gradient_y, torch.Tensor([[-1], [2], [-1]]).view(1, 1, 3, 1).to(device), padding=(1, 0))
        H_xy = F.conv2d(gradient_x, torch.Tensor([[-1, 0, 1]]).view(1, 1, 1, 3).to(device), padding=(0, 1))
        H_yx = F.conv2d(gradient_y, torch.Tensor([[-1], [0], [1]]).view(1, 1, 3, 1).to(device), padding=(1, 0))
        curvature = (H_xx * (1 + gradient_y) ** 2 - 2 * H_xy * gradient_x * gradient_y + H_yy * (
                    1 + gradient_x) ** 2) / 2 * (1 + gradient_x ** 2 + gradient_y ** 2) ** (3 / 2)
        negative_curvature = torch.nn.functional.relu(-curvature)  #
        average_negative_curvature = torch.sum(negative_curvature) / torch.sum(negative_curvature != 0).float()
        curvature_loss += average_negative_curvature
    for depth in range(probability_map2.size(0)):
        prob_map_depth2 = probability_map2[depth, :, :]
        prob_map_depth2 = F.conv2d(prob_map_depth2.view(1, 1, 256, 256), sobel_x.view(1, 1, 3, 3))
        gradient_x = F.conv2d(prob_map_depth2, torch.Tensor([[-1, 0, 1]]).view(1, 1, 1, 3).to(device), padding=(0, 1))
        gradient_y = F.conv2d(prob_map_depth2, torch.Tensor([[-1], [0], [1]]).view(1, 1, 3, 1).to(device), padding=(1, 0))
        H_xx = F.conv2d(gradient_x, torch.Tensor([[-1, 2, -1]]).view(1, 1, 1, 3).to(device), padding=(0, 1))
        H_yy = F.conv2d(gradient_y, torch.Tensor([[-1], [2], [-1]]).view(1, 1, 3, 1).to(device), padding=(1, 0))
        H_xy = F.conv2d(gradient_x, torch.Tensor([[-1, 0, 1]]).view(1, 1, 1, 3).to(device), padding=(0, 1))
        H_yx = F.conv2d(gradient_y, torch.Tensor([[-1], [0], [1]]).view(1, 1, 3, 1).to(device), padding=(1, 0))
        curvature = (H_xx * (1 + gradient_y) ** 2 - 2 * H_xy * gradient_x * gradient_y + H_yy * (
                    1 + gradient_x) ** 2) / 2 * (1 + gradient_x ** 2 + gradient_y ** 2) ** (3 / 2)
        negative_curvature = torch.nn.functional.relu(-curvature)  #
        average_negative_curvature = torch.sum(negative_curvature) / torch.sum(negative_curvature != 0).float()
        curvature_loss2 += average_negative_curvature
    for depth in range(probability_map3.size(0)):
        prob_map_depth3 = probability_map3[depth, :, :]
        prob_map_depth3 = F.conv2d(prob_map_depth3.view(1, 1, 256, 256), sobel_x.view(1, 1, 3, 3))
        gradient_x = F.conv2d(prob_map_depth3, torch.Tensor([[-1, 0, 1]]).view(1, 1, 1, 3).to(device), padding=(0, 1))
        gradient_y = F.conv2d(prob_map_depth3, torch.Tensor([[-1], [0], [1]]).view(1, 1, 3, 1).to(device), padding=(1, 0))
        H_xx = F.conv2d(gradient_x, torch.Tensor([[-1, 2, -1]]).view(1, 1, 1, 3).to(device), padding=(0, 1))
        H_yy = F.conv2d(gradient_y, torch.Tensor([[-1], [2], [-1]]).view(1, 1, 3, 1).to(device), padding=(1, 0))
        H_xy = F.conv2d(gradient_x, torch.Tensor([[-1, 0, 1]]).view(1, 1, 1, 3).to(device), padding=(0, 1))
        H_yx = F.conv2d(gradient_y, torch.Tensor([[-1], [0], [1]]).view(1, 1, 3, 1).to(device), padding=(1, 0))
        curvature = (H_xx * (1 + gradient_y) ** 2 - 2 * H_xy * gradient_x * gradient_y + H_yy * (
                    1 + gradient_x) ** 2) / 2 * (1 + gradient_x ** 2 + gradient_y ** 2) ** (3 / 2)
        negative_curvature = torch.nn.functional.relu(-curvature)  #
        average_negative_curvature = torch.sum(negative_curvature) / torch.sum(negative_curvature != 0).float()
        curvature_loss3 += average_negative_curvature
    loss = curvature_loss + curvature_loss2 + curvature_loss3
    return loss
def calculate_exact_curvature(contour):
    contour = contour[:, 0, :].T
    # Calculate derivatives of the contour points
    if len(contour[0]) < 3:
        return np.zeros_like(contour[0])
    edge_order = min(2, len(contour[0]) - 1)
    dx = np.gradient(contour[0], edge_order= edge_order)
    dy = np.gradient(contour[1], edge_order= edge_order)
    ddx = np.gradient(dx, edge_order= edge_order)
    ddy = np.gradient(dy, edge_order= edge_order)
    # Calculate curvature using the formula: curvature = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)

    return curvature
def dice_weight_loss(predict,target,weight):
    target = target.float()*weight
    predict = predict*weight
    smooth = 1e-4
    intersect = torch.sum(predict*target)
    dice = (2 * intersect + smooth)/(torch.sum(target)+torch.sum(predict*predict)+smooth)
    loss = 1.0 - dice
    return loss
def dice_loss(predict,target, ent_dice_weight_map):
    target = target.float()
    smooth = 1e-4
    intersect = torch.sum(ent_dice_weight_map*predict*target*ent_dice_weight_map)
    dice = (2 * intersect + smooth)/(torch.sum(target*ent_dice_weight_map)+torch.sum(predict*ent_dice_weight_map)+smooth)
    # dice = (2 * intersect + smooth)/(torch.sum(target)+torch.sum(predict*predict)+smooth)

    loss = 1.0 - dice
    return loss
class diceLoss_weight(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        self.n_classes = n_classes
    def one_hot_encode(self,input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            tmp = (input_tensor==i) * torch.ones_like(input_tensor)
            tensor_list.append(tmp)
        output_tensor = torch.cat(tensor_list,dim=1)
        return output_tensor.float()
    
    def forward(self,inputs,target,weight):
        x_shape = list(target.shape)
        if(len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            target = torch.transpose(target, 1, 2)
            target = torch.reshape(target, new_shape)
        target = self.one_hot_encode(target)
        
        assert inputs.shape == target.shape,(target.shape,inputs.shape)
        class_wise_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            diceloss = dice_weight_loss(inputs[:,i,:,:], target[:,i,:,:],weight)
            class_wise_dice.append(diceloss)
            loss += diceloss
        return loss/self.n_classes

class DiceLoss(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        self.n_classes = n_classes
        
    def one_hot_encode(self,input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            tmp = (input_tensor==i) * torch.ones_like(input_tensor)
            tensor_list.append(tmp)
        output_tensor = torch.cat(tensor_list,dim=1)
        return output_tensor.float()
    
    def forward(self,inputs, target, ent_dice_weight_map, one_hot):
        x_shape = list(target.shape)
        if(len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            target = torch.transpose(target, 1, 2)
            target = torch.reshape(target, new_shape)

        if one_hot:
            target = self.one_hot_encode(target)
        assert inputs.shape == target.shape,'size must match'
        class_wise_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            diceloss = dice_loss(inputs[:,i,:,:], target[:,i,:,:], ent_dice_weight_map)
            class_wise_dice.append(diceloss)
            loss += diceloss
        return loss/self.n_classes

class Ce_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,input,target):
        inputs = F.softmax(input,dim=1)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        loss = 0
        for i in range(0,input.shape[0]):
            loss += self.ce_loss(input[i].unsqueeze(0),target)
        return loss

class DiceLoss_n(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        self.n_classes = n_classes
    
    def forward(self,input,target,weight=None,softmax=True):
        if softmax:
            inputs = F.softmax(input,dim=1)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.shape == target.shape,'size must match'
        class_wise_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            diceloss = dice_loss(inputs[:,i], target[:,i])
            class_wise_dice.append(diceloss)
            loss += diceloss * weight[i]
        return loss/self.n_classes

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.eps = 1e-4
        self.num_classes = num_classes

    def forward(self, predict, target):
        weight = []
        for c in range(self.num_classes):
            weight_c = torch.sum(target == c).float()
            weight.append(weight_c)
        weight = torch.tensor(weight).to(target.device)
        weight = 1 - weight / (torch.sum(weight))
        weight[0] = 0.0
        target = target.argmax(axis=1)
        wce_loss = F.cross_entropy(predict, target.long(), weight)
        return wce_loss

class DiceLoss_weight(nn.Module):
    def __init__(self,num_classes,alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.diceloss = diceLoss_weight(self.num_classes)
    def forward(self,predict,label,weight):
        x_shape = list(label.shape)
        if(len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            x = torch.transpose(label, 1, 2)
            label = torch.reshape(x, new_shape)
        loss = self.diceloss(predict,label,weight) 
        return loss
class DiceCeLoss(nn.Module):
     #predict : output of model (i.e. no softmax)[N,C,*]
     #target : gt of img [N,1,*]
    def __init__(self,num_classes,alpha=1.0):
        '''
        calculate loss:
            celoss + alpha*celoss
            alpha : default is 1
        '''
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.diceloss = DiceLoss(self.num_classes)
        self.celoss = WeightedCrossEntropyLoss(self.num_classes)
        
    def forward(self,predict,label,one_hot):
        
        x_shape = list(label.shape)
        if(len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            x = torch.transpose(label, 1, 2)
            label = torch.reshape(x, new_shape)
        celoss = self.celoss.to(label.device)
        diceloss = self.diceloss(predict,label,one_hot)
        celoss = self.celoss(predict,label)
        loss = diceloss + celoss
        return loss
        