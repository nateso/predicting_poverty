import torch
import torchvision
import torch.nn as nn


def init_weights_kaiming(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


# concatenate the weights for the new input channel
def init_weights_scaled_l1(pretrained_weights, n_channels):
    mean_weights = torch.mean(pretrained_weights, axis=1).unsqueeze(1)
    mean_weights = torch.cat([mean_weights] * (n_channels - 3), dim=1)
    weights = torch.cat([pretrained_weights, mean_weights], dim=1)
    weights = weights * 3 / n_channels
    return weights


class ResNet18(nn.Module):
    def __init__(self,
                 input_channels,
                 pretrained_weights=False,
                 scaled_weight_init=False,
                 random_seed=None):
        super(ResNet18, self).__init__()  # initialise the parent class

        self.input_channels = input_channels
        self.pretrained_weights = pretrained_weights
        self.scaled_weight_init = scaled_weight_init
        self.random_seed = random_seed
        self.model = None
        self.layer1_pretrained_weights = None

        # initialise the model
        self.initialise()

    def initialise(self):
        # initialise the ResNet18 architecture
        if self.pretrained_weights:
            self.model = torchvision.models.resnet18(weights='DEFAULT')

            if self.scaled_weight_init:
                # extract the weights of the first layer: (only if input is multispectral image)
                self.layer1_pretrained_weights = self.model.conv1.weight

        if not self.pretrained_weights:
            self.model = torchvision.models.resnet18(weights=None)

        # adapt the first and the last layer to accommodate the different shape of the input and output
        self.adapt_input_output()

        # initialise weights
        self.init_weights(random_seed=self.random_seed)

    def adapt_input_output(self):
        self.model.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                     bias=False)
        self.model.fc = nn.Linear(512, 1, bias=True)

    def init_weights(self, random_seed=None):
        if random_seed is not None:
            torch.manual_seed(random_seed)

        if not self.pretrained_weights:
            # initialise all weights randomly
            self.model.apply(init_weights_kaiming)

        if self.pretrained_weights:
            # initialise the weights of the last layer randomly
            self.model.fc.weight = torch.nn.init.kaiming_normal_(self.model.fc.weight, nonlinearity='relu')

            # initialise the weights of the first layer depending on whether the input is multispectral image or not
            if self.scaled_weight_init:
                # if input is multispectral image, initialise the weights of the first layer using scaled initialisation
                self.model.conv1.weight = torch.nn.Parameter(
                    init_weights_scaled_l1(self.layer1_pretrained_weights, self.input_channels))
            # else initialise the weights of the first layer using kaiming initialisation
            else:
                self.model.conv1.weight = torch.nn.init.kaiming_normal_(self.model.conv1.weight, nonlinearity='relu')

#
# def init_resnet(input_channels, ms=True, random_weights=True, random_seed=None):
#     if random_seed is not None:
#         torch.manual_seed(random_seed)
#     if random_weights:
#         ResNet18 = torchvision.models.resnet18(weights=None)
#         # adapt the first and the last layer to accommodate the different shape of the input and output
#         ResNet18.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         ResNet18.fc = nn.Linear(512, 1, bias=True)
#
#         ResNet18.fc.weight = torch.nn.init.kaiming_uniform_(ResNet18.fc.weight)
#         ResNet18.conv1.weight = torch.nn.init.kaiming_uniform_(ResNet18.conv1.weight)
#
#     if random_weights == False:
#         # initialise the pretrained ResNet18 architecture using the default weights (i.e. pretrained weights)
#         ResNet18 = torchvision.models.resnet18(weights='DEFAULT')
#
#         # extract the weights of the first layer:
#         layer1_pretrained_weights = ResNet18.conv1.weight
#         mean_weights = torch.mean(layer1_pretrained_weights, axis=1).unsqueeze(1)
#
#         # adapt the first and the last layer to accomodate the different shape of the input and output
#         ResNet18.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         ResNet18.fc = nn.Linear(512, 1, bias=True)  # just adapt the output to 1, since we have a regression task
#
#         # initialise the weights of the last layer randomly
#         # set a seed to make results reproducible
#
#         ResNet18.fc.weight = torch.nn.init.kaiming_uniform_(ResNet18.fc.weight)
#         if ms:
#             ResNet18.conv1.weight = torch.nn.Parameter(
#                 init_weights_scaled_l1(layer1_pretrained_weights, input_channels))
#         else:
#             ResNet18.conv1.weight = torch.nn.init.kaiming_uniform_(ResNet18.conv1.weight)
#
#     return ResNet18
