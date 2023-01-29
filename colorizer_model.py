import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.activation = nn.Sigmoid()
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm_1 = nn.BatchNorm2d(64)
        self.conv2d_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm_2 = nn.BatchNorm2d(128)
        self.conv2d_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm_3 = nn.BatchNorm2d(256)
        self.conv2d_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm_4 = nn.BatchNorm2d(512)
        self.conv2d_5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm_5 = nn.BatchNorm2d(512)
        self.conv2d_6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm_6 = nn.BatchNorm2d(512)
        self.conv2d_7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm_7 = nn.BatchNorm2d(512)
        self.conv2d_8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)

        self.conv2d_9 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1,
                                           bias=False)
        self.batchnorm_9 = nn.BatchNorm2d(512)
        self.conv2d_10 = nn.ConvTranspose2d(in_channels=512 * 2, out_channels=512, kernel_size=4, stride=2, padding=1,
                                            bias=False)
        self.batchnorm_10 = nn.BatchNorm2d(512)
        self.conv2d_11 = nn.ConvTranspose2d(in_channels=512 * 2, out_channels=512, kernel_size=4, stride=2, padding=1,
                                            bias=False)
        self.batchnorm_11 = nn.BatchNorm2d(512)
        self.conv2d_12 = nn.ConvTranspose2d(in_channels=512 * 2, out_channels=512, kernel_size=4, stride=2, padding=1,
                                            bias=False)
        self.batchnorm_12 = nn.BatchNorm2d(512)
        self.conv2d_13 = nn.ConvTranspose2d(in_channels=512 * 2, out_channels=256, kernel_size=4, stride=2, padding=1,
                                            bias=False)
        self.batchnorm_13 = nn.BatchNorm2d(256)
        self.conv2d_14 = nn.ConvTranspose2d(in_channels=256 * 2, out_channels=128, kernel_size=4, stride=2, padding=1,
                                            bias=False)
        self.batchnorm_14 = nn.BatchNorm2d(128)
        self.conv2d_15 = nn.ConvTranspose2d(in_channels=128 * 2, out_channels=64, kernel_size=4, stride=2, padding=1,
                                            bias=False)
        self.batchnorm_15 = nn.BatchNorm2d(64)
        self.conv2d_16 = nn.ConvTranspose2d(in_channels=64 * 2, out_channels=3, kernel_size=4,
                                            stride=2, padding=1, bias=True)

    def forward(self, encoder_input):
        # encoder
        encoder_output_1 = self.leakyrelu(self.conv2d_1(encoder_input))
        encoder_output_2 = self.leakyrelu(self.batchnorm_2(self.conv2d_2(encoder_output_1)))
        encoder_output_3 = self.leakyrelu(self.batchnorm_3(self.conv2d_3(encoder_output_2)))
        encoder_output_4 = self.leakyrelu(self.batchnorm_4(self.conv2d_4(encoder_output_3)))
        encoder_output_5 = self.leakyrelu(self.batchnorm_5(self.conv2d_5(encoder_output_4)))
        encoder_output_6 = self.leakyrelu(self.batchnorm_6(self.conv2d_6(encoder_output_5)))
        encoder_output_7 = self.leakyrelu(self.batchnorm_7(self.conv2d_7(encoder_output_6)))
        encoder_output = self.conv2d_8(encoder_output_7)
        # decoder
        decoder_output = self.batchnorm_9(self.conv2d_9(self.relu(encoder_output)))
        decoder_output = self.batchnorm_10(
            self.conv2d_10(self.relu(torch.cat([encoder_output_7, decoder_output], 1))))  # skip connection
        decoder_output = self.batchnorm_11(
            self.conv2d_11(self.relu(torch.cat([encoder_output_6, decoder_output], 1))))  # skip connection
        decoder_output = self.batchnorm_12(
            self.conv2d_12(self.relu(torch.cat([encoder_output_5, decoder_output], 1))))  # skip connection
        decoder_output = self.batchnorm_13(
            self.conv2d_13(self.relu(torch.cat([encoder_output_4, decoder_output], 1))))  # skip connection
        decoder_output = self.batchnorm_14(
            self.conv2d_14(self.relu(torch.cat([encoder_output_3, decoder_output], 1))))  # skip connection
        decoder_output = self.batchnorm_15(
            self.conv2d_15(self.relu(torch.cat([encoder_output_2, decoder_output], 1))))  # skip connection
        decoder_output = self.activation(
            self.conv2d_16(self.relu(torch.cat([encoder_output_1, decoder_output], 1))))  # skip connection
        return decoder_output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.leakyrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.conv2d_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv2d_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm_2 = nn.BatchNorm2d(128)
        self.conv2d_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm_3 = nn.BatchNorm2d(256)
        self.conv2d_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=False)
        self.batchnorm_4 = nn.BatchNorm2d(512)
        self.conv2d_5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=True)

    def forward(self, input):
        output = self.leakyrelu(self.conv2d_1(input))
        output = self.leakyrelu(self.batchnorm_2(self.conv2d_2(output)))
        output = self.leakyrelu(self.batchnorm_3(self.conv2d_3(output)))
        output = self.leakyrelu(self.batchnorm_4(self.conv2d_4(output)))
        output = self.sigmoid(self.conv2d_5(output))
        return output


@torch.no_grad()
def init_weights(m, gain=0.02):
    """Weight initialisation of the different layers of the Generator and Discriminator"""
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight.data, mean=0.0, std=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1., gain)
        nn.init.constant_(m.bias.data, 0.)


class DiscriminatorLoss(nn.Module):
    """For discriminator, the output is a 30x30 tensor
     if the image is real, it should return all ones 'real_labels'
     if the image is fake, it should return all zeros 'fake_labels'
     returns the MSE loss between the output of the discriminator and the label"""

    def __init__(self, device):
        super().__init__()
        self.register_buffer('real_labels', torch.ones([30, 30], requires_grad=False, device=device), False)
        self.register_buffer('fake_labels', torch.zeros([30, 30], requires_grad=False, device=device), False)
        # use MSE loss for the discriminator
        self.loss = nn.MSELoss()

    def forward(self, predictions, target_is_real):
        if target_is_real:
            target = self.real_labels
        else:
            target = self.fake_labels
        return self.loss(predictions, target.expand_as(predictions))
