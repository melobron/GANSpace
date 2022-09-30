import torch
from torchvision.transforms import transforms
from sklearn.decomposition import PCA

import gradio as gr
import math
import numpy as np
import pickle
import cv2

from models.StyleGAN2 import StyledGenerator
from explore import get_mean_style


def save_pca_components(model_path, title='pca', gpu_num=0, random_seed=0, latent_size=100000, n_components=100):
    # Device
    device = torch.device('cuda:{}'.format(gpu_num))

    # Random Seed
    torch.manual_seed(random_seed)

    # Generator
    generator = StyledGenerator().to(device)
    generator.load_state_dict(torch.load('{}'.format(model_path), map_location=device))

    # Style
    latent = torch.randn(latent_size, 512).to(device)
    style = generator.get_style(latent)
    style = style.detach().cpu()

    # PCA
    transformer = PCA(n_components=n_components, svd_solver='full')
    transformer.fit(X=style)

    components = transformer.components_
    std = np.dot(components, style.T).std(axis=1)
    idx = np.argsort(std)[::-1]
    std = std[idx]
    components[:] = components[idx]
    data = {'std': std, 'components': components}

    # Save pickle data
    with open('./pickle_data/components({}).pickle'.format(title), 'wb') as f:
        pickle.dump(data, f)


def save_mean_style(model_path, title='Dog', gpu_num=0, random_seed=0, style_mean_num=10):
    # Device
    device = torch.device('cuda:{}'.format(gpu_num))

    # Random Seed
    torch.manual_seed(random_seed)

    # Generator
    generator = StyledGenerator().to(device)
    generator.load_state_dict(torch.load('{}'.format(model_path), map_location=device))

    # Mean style
    mean_style = get_mean_style(generator, device, style_mean_num=style_mean_num).cpu()
    data = {'mean_style': mean_style}

    # Save pickle data
    with open('./pickle_data/mean_style({}).pickle'.format(title), 'wb') as f:
        pickle.dump(data, f)


def show(random_seed, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10):
    # Device
    device = torch.device('cuda:{}'.format(gpu_num))

    # Random Seed
    torch.manual_seed(random_seed)

    # Generator
    generator = StyledGenerator().to(device)
    generator.load_state_dict(torch.load('{}'.format(model_path), map_location=device))

    # Style
    latent = torch.randn(1, 512).to(device)
    style = generator.get_style(latent)
    style = style.detach().cpu()

    # Components
    with open('./pickle_data/components(pca).pickle', 'rb') as f:
        data = pickle.load(f)
    components = data['pickle_data']

    with open('./pickle_data/mean_style(Dog).pickle') as f:
        data = pickle.load(f)
    mean_style = data['mean_style'].to(device)

    # Explore
    controls = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]
    for i, control in enumerate(controls):
        style += components[i, :] * control
    style = torch.unsqueeze(style, dim=0).type(torch.FloatTensor).to(device)
    img = generator.forward_from_style(style, step=step, alpha=1, mean_style=mean_style)

    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    transform = transforms.Compose([
        transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
    ])

    img = transform(img)
    img = img.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0., 1.)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


if __name__ == "__main__":
    # # Save PCA pickle data
    # save_pca_components(model_path='./pre-trained/Dog(FreezeD).pth', title='pca', latent_size=100000, n_components=100)

    # # Save mean style pickle data
    # save_mean_style(model_path='./pre-trained/Dog(FreezeD).pth', title='Dog', gpu_num=0, random_seed=0, style_mean_num=10)

    # Demo
    gpu_num = 0
    model_path = './pre-trained/Dog(FreezeD).pth'
    img_size = 256
    step = int(math.log(img_size, 2)) - 2

    demo = gr.Interface(
        fn=show,
        inputs=[gr.Slider(-1, 1) for _ in range(11)],
        outputs=gr.Image(type='pil'),
        title='Face Generation',
        description='Exploration of StyleGAN feature space via GANSpace'
    )

    demo.launch()
