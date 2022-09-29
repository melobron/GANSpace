import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from sklearn.decomposition import PCA

import os
import argparse
import random
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

from models.StyleGAN2 import StyledGenerator


@torch.no_grad()
def get_mean_style(generator, device, style_mean_num):
    mean_style = None

    for _ in range(style_mean_num):
        style = generator.mean_style(torch.randn(1024, 512).to(device))
        if mean_style is None:
            mean_style = style
        else:
            mean_style += style

    mean_style /= style_mean_num
    return mean_style




if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Explore Latent')

    parser.add_argument('--gpu_num', default=0, type=int)
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--model_path', default='./pre-trained/FFHQ(pretrained).model', type=str)
    # parser.add_argument('--model_path', default='./pre-trained/Dog(FreezeD).pth', type=str)
    parser.add_argument('--dataset_name', default='FFHQ', type=str)  # FFHQ, Dog
    parser.add_argument('--img_size', default=256, type=int)  # Pre-trained model suited for 256

    # Mean Style
    parser.add_argument('--style_mean_num', default=10, type=int)  # Style mean calculation for Truncation trick
    parser.add_argument('--alpha', default=1, type=float)  # Fix=1: No progressive growing
    parser.add_argument('--style_weight', default=0.7, type=float)  # 0: Mean of FFHQ, 1: Independent

    # PCA
    parser.add_argument('--batch_size', default=100000, type=int)
    parser.add_argument('--n_components', default=100, type=int)  # Number of eigenvectors

    # Transformations
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--mean', type=tuple, default=(0.5, 0.5, 0.5))
    parser.add_argument('--std', type=tuple, default=(0.5, 0.5, 0.5))

    opt = parser.parse_args()

    device = torch.device('cuda:{}'.format(opt.gpu_num))

    # Random Seeds
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    # Save Directory
    save_dir = './results/{}'.format(opt.dataset_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Model
    generator = StyledGenerator().to(device)
    generator.load_state_dict(torch.load(opt.model_path)['g_running'])
    # generator.load_state_dict(torch.load('{}'.format(opt.model_path), map_location=device))
    generator.eval()

    # Mean Styles
    mean_style = get_mean_style(generator, device, style_mean_num=opt.style_mean_num)

    # Parameters
    step = int(math.log(opt.img_size, 2)) - 2

    # PCA
    latent = torch.randn(opt.batch_size, 512).to(device)
    styles = generator.get_style(latent)

    # PCA
    styles = styles.detach().cpu()  # (B, D) = (16384, 512)
    transformer = PCA(n_components=opt.n_components, svd_solver='full')
    transformer.fit(X=styles)

    components = transformer.components_  # (P, D) = (100, 512)
    variances = transformer.explained_variance_  # (P, 1) = (100, 1)

    # Generate
    random_style = styles[70, :]
    examples = [random_style + (i-5) * 0.4 * components[0, :] for i in range(11)]
    examples = [torch.unsqueeze(examples[i], dim=0) for i in range(len(examples))]
    examples = torch.cat(examples, dim=0).type(torch.FloatTensor).to(device)
    start = torch.unsqueeze(random_style, dim=0).to(device)

    start_img = generator.forward_from_style(start, step=step, alpha=opt.alpha, mean_style=mean_style, style_weight=opt.style_weight)
    example_imgs = generator.forward_from_style(examples, step=step, alpha=opt.alpha, mean_style=mean_style, style_weight=opt.style_weight)

    # Visualize
    mean, std = opt.mean, opt.std
    transform = transforms.Compose([
        transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
    ])
    # img = transform(start_img[0])
    # img = img.detach().cpu().numpy().transpose(1, 2, 0)
    # img = np.clip(img, 0., 1.)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    fig = plt.figure(figsize=(15, 10))
    rows, cols = 3, 4
    # fig.add_subplot(rows, cols, 1)
    # plt.imshow(img)
    # plt.title('start')

    for i in range(11):
        img = transform(example_imgs[i])
        img = img.detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0., 1.)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(img)
        plt.title('{}'.format(i-5))

    plt.tight_layout()
    plt.show()



