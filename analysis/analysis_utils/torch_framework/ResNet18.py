import torch
import torchvision
import torch.nn as nn
import copy


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
                 use_pretrained_weights=False,
                 scaled_weight_init=False,
                 random_seed=None):
        super(ResNet18, self).__init__()  # initialise the parent class

        self.input_channels = input_channels
        self.use_pretrained_weights = use_pretrained_weights
        self.scaled_weight_init = scaled_weight_init
        self.random_seed = random_seed
        self.model = None
        self.pretrained_weights = None
        self.layer1_pretrained_weights = None

        # initialise the model
        self.initialise()

    def initialise(self):
        # initialise the ResNet18 architecture
        if self.use_pretrained_weights:
            self.model = torchvision.models.resnet18(weights='DEFAULT')

            # extract the weights of the first layer: (only if you want scaled weight init in the first conv)
            self.layer1_pretrained_weights = self.model.conv1.weight

        if not self.use_pretrained_weights:
            self.model = torchvision.models.resnet18(weights=None)

        # adapt the first and the last layer to accommodate the different shape of the input and output
        self.adapt_input_output()

        # initialise weights
        self.init_weights(random_seed=self.random_seed)

        # store the pretrained weights for weight resetting
        if self.use_pretrained_weights:
            self.pretrained_weights = copy.deepcopy(self.model.state_dict())

    def adapt_input_output(self):
        self.model.conv1 = nn.Conv2d(in_channels=self.input_channels,
                                     out_channels=64,
                                     kernel_size=(7, 7),
                                     stride=(2, 2),
                                     padding=(3, 3),
                                     bias=False)

        self.model.fc = nn.Linear(in_features=512,
                                  out_features=1,
                                  bias=True)

    def init_weights(self, random_seed=None):
        if random_seed is not None:
            torch.manual_seed(random_seed)

        if not self.use_pretrained_weights:
            # initialise all weights randomly
            self.model.apply(init_weights_kaiming)

        if self.use_pretrained_weights:

            # initialise the weights of the last layer randomly
            self.model.fc.weight = torch.nn.init.kaiming_normal_(self.model.fc.weight, nonlinearity='relu')

            # initialise the weights of the first layer depending on whether the input is multispectral image or not
            if self.scaled_weight_init:
                # if input is multispectral image, initialise the weights of the first layer using scaled initialisation
                self.model.conv1.weight = torch.nn.Parameter(
                    init_weights_scaled_l1(self.layer1_pretrained_weights, self.input_channels)
                )
            # else initialise the weights of the first layer using kaiming initialisation
            else:
                self.model.conv1.weight = torch.nn.init.kaiming_normal_(self.model.conv1.weight, nonlinearity='relu')

    def reset_weights(self, random_seed=None):
        if random_seed is not None:
            torch.manual_seed(random_seed)

        if self.use_pretrained_weights:
            self.model.load_state_dict(self.pretrained_weights)
            self.model.fc.weight = torch.nn.init.kaiming_normal_(self.model.fc.weight, nonlinearity='relu')
            if not self.scaled_weight_init:
                self.model.conv1.weight = torch.nn.init.kaiming_normal_(self.model.conv1.weight, nonlinearity='relu')

        else:
            self.model.apply(init_weights_kaiming)
