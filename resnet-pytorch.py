
import torchvision.models as models
import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

""" resnet.eval()
Yes, it hurts the testing accuracy. If you use resnet.eval(), batch normalization layer uses running average/variance instead of mini-batch statistics. You can improve the performance when using resnet.eval() by changing the momentum coefficient in batch normalization layer.
It is recommended to change nn.BatchNorm2d(16) to nn.BatchNorm2d(16, momentum=0.01). The default value of the momentum is 0.1.
"""
img_path="./dog.jpg"

image_transform = transforms.Compose([
    transforms.Scale([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ]),
])

resnet = models.resnet152(pretrained=True)
resnet = resnet.eval()
img_vec = image_transform(Image.open(img_path).convert('RGB')).unsqueeze(0)
img_vec = resnet(Variable(img_vec)).data.squeeze(0).cpu().numpy()
np.savetxt("vec.txt",img_vec)

