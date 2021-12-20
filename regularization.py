import torch
import torchvision


class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
 
    def to(self, device):
        self.device = device
        super().to(device)
        return self
 
    def forward(self):
        self.weight_list = self.get_weight(self.model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss
 
    def get_weight(self, model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
 
    def regularization_loss(self, weight_list, weight_decay, p=2):
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
        reg_loss = weight_decay * reg_loss
        return reg_loss