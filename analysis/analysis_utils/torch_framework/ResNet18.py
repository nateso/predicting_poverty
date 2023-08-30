import torch
import torchvision
import torch.nn as nn


def init_resnet(input_channels, ms=True, random_weights=True, random_seed=None):
    if random_seed is not None:
        torch.manual_seed(random_seed)
    if random_weights:
        ResNet18 = torchvision.models.resnet18(weights=None)
        # adapt the first and the last layer to accommodate the different shape of the input and output
        ResNet18.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        ResNet18.fc = nn.Linear(512, 1, bias=True)

        ResNet18.fc.weight = torch.nn.init.kaiming_uniform_(ResNet18.fc.weight)
        ResNet18.conv1.weight = torch.nn.init.kaiming_uniform_(ResNet18.conv1.weight)

    if random_weights == False:
        # initialise the pretrained ResNet18 architecture using the default weights (i.e. pretrained weights)
        ResNet18 = torchvision.models.resnet18(weights='DEFAULT')

        # extract the weights of the first layer:
        layer1_pretrained_weights = ResNet18.conv1.weight
        mean_weights = torch.mean(layer1_pretrained_weights, axis=1).unsqueeze(1)

        # adapt the first and the last layer to accomodate the different shape of the input and output
        ResNet18.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        ResNet18.fc = nn.Linear(512, 1, bias=True)  # just adapt the output to 1, since we have a regression task

        # initialise the weights of the last layer randomly
        # set a seed to make results reproducible

        ResNet18.fc.weight = torch.nn.init.kaiming_uniform_(ResNet18.fc.weight)
        if ms:
            ResNet18.conv1.weight = torch.nn.Parameter(
                init_weights_scaled_l1(layer1_pretrained_weights, input_channels))
        else:
            ResNet18.conv1.weight = torch.nn.init.kaiming_uniform_(ResNet18.conv1.weight)

    return ResNet18


# concatenate the weights for the new input channel
def init_weights_scaled_l1(pretrained_weights, n_channels):
    mean_weights = torch.mean(pretrained_weights, axis=1).unsqueeze(1)
    mean_weights = torch.cat([mean_weights] * (n_channels - 3), dim=1)
    weights = torch.cat([pretrained_weights, mean_weights], dim=1)
    weights = weights * 3 / n_channels
    return weights
