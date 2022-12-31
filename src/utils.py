from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms as transforms
import cv2
import numpy as np
from settings import DEVICE, SIZE

# mean and std of ImageNet to use pre-trained VGG
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Image Transforms
loader = T.Compose([
  T.Resize(SIZE),
  T.CenterCrop(SIZE),
  T.ToTensor()
])

unloader = T.ToPILImage()

normalize = transforms.Normalize(mean=IMAGENET_MEAN,
                                 std=IMAGENET_STD)

denormalize = transforms.Normalize(mean=[-mean/std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/std for std in IMAGENET_STD])								 

def load_image(path):
	image = loader(Image.open(path)).unsqueeze(0)
	return image.to(DEVICE, torch.float)

def save_image(tensor, path):
	image = unloader(tensor.cpu().clone().squeeze(0))
	image.save(path)

def get_transformer(imsize=SIZE, cropsize=SIZE):
    # print("get_transformer")
    transformer = []
    if imsize:
        transformer.append(transforms.Resize(imsize))
    if cropsize:
        transformer.append(transforms.CenterCrop(cropsize)),
    transformer.append(transforms.ToTensor())
    # transformer.append(normalize)
    # print("transformer", transformer)
    return transforms.Compose(transformer)

def imloadFrame(frame):
    # print("imloadFrame")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = frame.transpose()
    # print("frame type: ", type(frame))
    # print("frame size: ", frame.shape)
    # frame = torch.from_numpy(frame)

    image2 = Image.fromarray(frame)

    transformer = get_transformer()
    # frame.unsqueeze(0)
    new_frame = transformer(image2)
    # print("new_frame type: ", type(new_frame))
    return new_frame.unsqueeze(0)

def imsaveframe(tensor, writer=None):
    # print("imsaveframe, tensor type: ", type(tensor))
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = torchvision.utils.make_grid(tensor)    
    tensor = denormalize(tensor).clamp_(0.0, 1.0)
    # convert tensor to PIL image
    frame = np.asarray(transforms.ToPILImage()(tensor)) 
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if(writer is not None): 
        writer.write(frame)
    return frame