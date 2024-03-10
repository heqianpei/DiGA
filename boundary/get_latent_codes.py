import os
import tempfile
import numpy as np
from argparse import Namespace
from pathlib import Path
import torch
from torchvision import transforms
import PIL.Image
import scipy
import scipy.ndimage
import imageio
from models.psp import pSp
import argparse


def get_latents(net, x):
    codes = net.encoder(x)
    # if net.opts.start_from_latent_avg:
    #     if codes.ndim == 2:
    #         codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
    #     else:
    #         codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)

    # latent_codes = np.random.randn(1, net.latent_space_dim)
    # latent_codes = latent_codes.reshape(-1, net.latent_space_dim)
    # norm = np.linalg.norm(latent_codes, axis=1, keepdims=True)
    # latent_codes = latent_codes / norm * np.sqrt(net.latent_space_dim)
    return codes

def get_latent_code(image,net):
    #out_path = Path(tempfile.mkdtemp()) / "out.png"
    resize_dims = (256, 256)
    input_path = str(image)
    # for replicate, webcam input might be rgba, convert to rgb first
    input = imageio.imread(input_path)
    if input.shape[-1] == 4:
        rgba_image = PIL.Image.open(input_path)
        rgb_image = rgba_image.convert("RGB")
        input_path = "rgb_input.png"
        imageio.imwrite(input_path, rgb_image)

    # align and crop image
    input_image = PIL.Image.open(input_path)
    input_image.resize(resize_dims)
    img_transforms = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    transformed_image = img_transforms(input_image)
    x = transformed_image.unsqueeze(0).cuda()
    latent_codes = get_latents(net, x)
    return latent_codes


if __name__ == "__main__":
    # device = "cuda"
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--data_path", type=str, default=None, help="")
    parser.add_argument("--model_path", type=str, default=None, help="")
    parser.add_argument("--result_path", type=str, default=None, help="")

    args = parser.parse_args()

    path = args.data_path
    result_path =  args.result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)


    file_names = os.listdir(path)
    model_path = args.model_path
    ckpt = torch.load(model_path, map_location="cpu")
    opts = ckpt["opts"]
    opts["is_train"] = False
    opts["checkpoint_path"] = model_path
    opts = Namespace(**opts)
    net = pSp(opts)

    net.eval()
    net.cuda()
    index = 0
    for image in file_names:
        index += 1
        image_path = path+image
        result = get_latent_code(image_path,net)
        last_dot_index = image.rfind('.')
        filename = image[:last_dot_index]
        torch.save(result,result_path+filename+".pth")
        print(index)
        # print(result)