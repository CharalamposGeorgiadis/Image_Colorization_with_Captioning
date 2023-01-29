import torch
from colorizer_model import Generator
import os
import skimage.io as io
from skimage.color import rgb2lab
import PIL.Image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def rgb_to_lab(rgb):
    """ Transforms a PIL RGB image into a Lab tensor """
    img = rgb2lab(rgb).astype('float32')
    lab = (img[..., 0:1] / 50.) - 1.
    return lab


def main():
    checkpoint_dir = 'checkpoints_colorizer/'
    generator = Generator().to(device)
    cpkt = torch.load(os.path.join(checkpoint_dir, 'cp-107.pth'), map_location=device)
    generator.load_state_dict(cpkt['G_state_dict'])
    generator.eval()

    for image in os.listdir("data/test2014/"):
        gray_image = io.imread("data/test2014/" + image, as_gray=True) * 255
        rgb_image = io.imread("data/test2014/" + image)

        pil_image = PIL.Image.fromarray(gray_image)
        rgb_image = PIL.Image.fromarray(rgb_image)

        gray_image_3 = PIL.Image.new('RGB', pil_image.size)
        gray_image_3.paste(pil_image)

        rgb_image = transforms.Resize((256, 256))(rgb_image)
        gray_image_3 = transforms.Resize((256, 256))(gray_image_3)
        lab_image = transforms.ToTensor()(rgb_to_lab(gray_image_3))
        lab_image = lab_image.unsqueeze(0).to(device)

        with torch.no_grad():
            generated_image = np.transpose(generator(lab_image).cpu().detach().numpy()[0], (1, 2, 0))
            rgb_image = np.array(rgb_image) / 255
            stack_image = np.hstack((rgb_image, generated_image))
            fig = plt.figure()
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(stack_image)
            plt.show()


if __name__ == '__main__':
    main()
