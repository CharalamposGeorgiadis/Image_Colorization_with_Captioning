import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage.color import rgb2lab


def rgb_to_lab(rgb):
    """ Transforms a PIL RGB image into a Lab tensor """
    img = rgb2lab(rgb).astype('float32')
    L = (img[..., 0:1] / 50.) - 1.
    ab = img[..., 1:] / 128
    return {'L': L, 'ab': ab}


class ImageDataset(Dataset):
    def __init__(self, path, im_size, crop_size, split):
        self.split = split
        self.paths = path
        self.total_imgs = os.listdir(path)

        if self.split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((im_size, im_size)),
                transforms.RandomHorizontalFlip(),
            ])

        elif self.split == 'test':
            self.transforms = transforms.Resize((im_size, im_size))

        self.preprocess = transforms.Compose([
            transforms.Resize(im_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, index):
        # Load RGB image
        img_loc = os.path.join(self.paths, self.total_imgs[index])
        img = Image.open(img_loc).convert('RGB')

        input_tensor = self.preprocess(img)

        # Augment the images and transform them to LAB
        img_input = self.transforms(img)
        img_lab = rgb_to_lab(img_input)

        # Return L, ab and embedding separately
        return {'L': transforms.ToTensor()(img_lab['L']), 'ab': transforms.ToTensor()(img_lab['ab'])}, input_tensor


def make_dataloaders(path, batch_size, im_size, crop_size, split, n_workers):
    dataset = ImageDataset(path, im_size, crop_size, split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
    return dataloader
