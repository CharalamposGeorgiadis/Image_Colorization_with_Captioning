import os
import re
import torch
import torch.optim as optim
from IPython.display import clear_output
from skimage.color import rgb2lab
from torch import nn
from tqdm import tqdm
from colorizer_dataloader import make_dataloaders
from colorizer_model import Generator, Discriminator, init_weights, DiscriminatorLoss

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def rgb_to_lab(rgb):
    """ Transforms a PIL RGB image into a Lab tensor """
    img = rgb2lab(rgb).astype('float32')
    L = (img[..., 0:1] / 50.) - 1.
    ab = img[..., 1:] / 128
    return {'L': L, 'ab': ab}


def main():
    image_size = 256
    crop_size = 256
    batch_size = 64
    n_epochs = 100

    checkpoint_dir = 'checkpoints_colorizer/'

    train_set = make_dataloaders(path='data/train2014', batch_size=batch_size, im_size=image_size,
                                 crop_size=crop_size, split='train', n_workers=0)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    opt_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # load the latest model if it finds checkpoint files in the checkpoint directory
    if os.listdir(checkpoint_dir):
        nums = [int(re.split('\-|\.', f)[1]) for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        cpkt = torch.load(os.path.join(checkpoint_dir, 'cp-' + str(max(nums)) + '.pth'), map_location=device)
        generator.load_state_dict(cpkt['G_state_dict'])
        discriminator.load_state_dict(cpkt['D_state_dict'])
        opt_G.load_state_dict(cpkt['optimizerG_state_dict'])
        opt_D.load_state_dict(cpkt['optimizerD_state_dict'])
        epoch = cpkt['epoch']
        loss_G = cpkt['loss_G']
        loss_D = cpkt['loss_D']
        generator.train()
        discriminator.train()
        initial_epoch = epoch + 1
        n_epochs += n_epochs  # Change that for fewer epochs
    else:
        generator = generator.apply(init_weights).train()
        discriminator = discriminator.apply(init_weights).train()
        initial_epoch = 0

    GANcriterion = DiscriminatorLoss(device)
    criterion = nn.L1Loss()
    lambda1 = 100.

    for epoch in range(initial_epoch, n_epochs):
        clear_output()
        print(f'Epoch ' + str(epoch))
        clear_output()
        running_loss_D = 0.0
        running_loss_G = 0.0
        for i, data in tqdm(enumerate(train_set)):
            L, ab = data[0]['L'].to(device), data[0]['ab'].to(device)
            fake_color = generator(L).to(device)
            real_image = torch.cat([L, ab], dim=1).to(device)

            fake_image = fake_color.to(device)
            rgb = data[1].to(device)

            # Train discriminator
            opt_D.zero_grad()

            # Train on real images
            real_preds = discriminator(real_image).to(device)
            loss_D_real = GANcriterion(real_preds, True).to(device)

            # Train on fake images
            fake_preds = discriminator(fake_image.detach()).to(device)
            loss_D_fake = GANcriterion(fake_preds, False).to(device)

            # Total discriminator loss
            loss_D = ((loss_D_fake + loss_D_real) * 0.5).to(device)
            loss_D.backward()
            opt_D.step()

            # Train generator
            opt_G.zero_grad()
            fake_preds = discriminator(fake_image).to(device)
            loss_G_GAN = GANcriterion(fake_preds, True).to(device)
            loss_G_L1 = (criterion(fake_color, rgb) * lambda1).to(device)

            # Total generator loss
            loss_G = (loss_G_GAN + loss_G_L1).to(device)
            loss_G.backward()
            opt_G.step()

            running_loss_D += loss_D.item()
            running_loss_G += loss_G.item()

        running_loss_D = running_loss_D / (i + 1)
        running_loss_G = running_loss_G / (i + 1)

        print('[%d, %5d] loss: %.3f %.3f' % (epoch, i + 1, running_loss_G, running_loss_D))

        checkpoint_path = os.path.join(checkpoint_dir, 'cp-{}.pth'.format(epoch))
        torch.save({'epoch': epoch,
                    'G_state_dict': generator.state_dict(),
                    'D_state_dict': discriminator.state_dict(),
                    'optimizerG_state_dict': opt_G.state_dict(),
                    'optimizerD_state_dict': opt_D.state_dict(),
                    'loss_G': loss_G,
                    'loss_D': loss_D
                    }, checkpoint_path)


if __name__ == '__main__':
    main()
