import torch
import torch.nn.functional as F
import torch.nn as nn

softmax = nn.Softmax(dim=1)

inputs = torch.randn(8, 8)
input_softmax = softmax(inputs)
# print(input_softmax)


label=torch.tensor([0,1,2,3,4,5,6,7],
               dtype=torch.int
            ).long()


label_one_hot = F.one_hot(label, num_classes=8)

def CE_Loss(output, target):
    return torch.mean((-1)*torch.sum(torch.log(output)*target, dim=1))

criterion = torch.nn.CrossEntropyLoss(reduction='mean')
loss=criterion(inputs,label)
print(loss.item())
print(CE_Loss(input_softmax,label_one_hot))