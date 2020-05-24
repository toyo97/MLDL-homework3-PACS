import torch
import torch.nn as nn
from torch.autograd import Function
from torch.hub import load_state_dict_from_url

__all__ = ['DANN', 'dann']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class GradReverse(Function):
    # Forwards identity
    # Sends backward reversed gradients
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DANN(nn.Module):

    def __init__(self, num_classes=1000):
        super(DANN, self).__init__()
        # same as AlexNet features
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # same as Alexnet classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        # new branch for domain classification
        self.discriminator = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),  # 2 output neurons
        )

    def forward(self, x, alpha=None):
        """
        Compute the output of the network to the input x
        :param x: input data
        :param alpha: parameter for discriminator loss contribution.
            If None, x is fed to classifier branch
        :return: output data
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if alpha is None:
            x = self.classifier(x)
        else:
            x = GradReverse.apply(x, alpha)
            x = self.discriminator(x)

        return x


def dann(pretrained=False, progress=True, num_classes=1000):
    """
    Custom DANN implementation with AlexNet architecture
    :param pretrained: flag to decide whether to load pretrained weights
    :param progress: show the download progress bar
    :return: model
    """
    model = DANN()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
        # copy parameters in fc6 to discriminator branch
        model.discriminator[1].weight.data = model.classifier[1].weight.data.clone().detach().requires_grad_(True)
        model.discriminator[1].bias.data = model.classifier[1].bias.data.clone().detach().requires_grad_(True)
        # copy parameters in fc7 to discriminator branch
        model.discriminator[4].weight.data = model.classifier[4].weight.data.clone().detach().requires_grad_(True)
        model.discriminator[4].bias.data = model.classifier[4].bias.data.clone().detach().requires_grad_(True)

        if num_classes != 1000:
            model.classifier[6] = nn.Linear(4096, num_classes)

    return model