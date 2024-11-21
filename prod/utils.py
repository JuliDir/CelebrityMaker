import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def load_dataset(path='data/preprocesed_images'):
    return datasets.ImageFolder(root=path, transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
]))

def load_model(model, checkpoint_path='prod/preprocessed_weights.pth', device='cpu'):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_interp(v1, v2, n):
  if not v1.shape == v2.shape:
    raise Exception('Diferent vector size')

  v1 = v1.to("cpu")
  v2 = v2.to("cpu")

  return np.array([np.linspace(v1[i], v2[i], n+2) for i in range(v1.shape[0])]).T

def model_interp(model, dataset, index1, index2, size = 10):
  img1 = dataset[index1][0].to("cpu").unsqueeze(0)
  img2 = dataset[index2][0].to("cpu").unsqueeze(0)

  with torch.no_grad():
    img1_compressed = model.encoder(img1)[0]
    img2_compressed = model.encoder(img2)[0]
    interps = get_interp(img1_compressed, img2_compressed, size)

    interps = torch.tensor(interps).to("cpu").squeeze()
    interps = interps.permute(1, 0)

    decoded_interps = model.decoder(interps)

  return decoded_interps

def show_interp(imgs, index1, index2, dataset, titles=None, scale=1.5):
  figsize = (12 * scale, 1 * scale)
  _, axes = plt.subplots(1, 12, figsize=figsize)
  axes = axes.flatten()
  for i, (ax, img) in enumerate(zip(axes, imgs)):
    try:
      img = img.detach().numpy()
    except:
      pass
    if i==0:
      ax.set_title(titles[0])
      ax.imshow(dataset[index1][0].permute(1,2,0).cpu().detach().numpy())
    elif i==11:
      ax.set_title(titles[1])
      ax.imshow(dataset[index2][0].permute(1,2,0).cpu().detach().numpy())
    else:
      ax.imshow(img)
      ax.axes.get_xaxis().set_visible(False)
      ax.axes.get_yaxis().set_visible(False)
  return axes