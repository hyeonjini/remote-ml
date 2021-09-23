import torch
import torchvision
class VGG11(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGG11, self).__init__()

        self.vgg11 = torchvision.models.vgg11(pretrained=True)

        self.features_conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.features_conv = self.vgg11.features[1:20]

        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.classifier = self.vgg11.classifier
        self.classifier[6] = torch.nn.Linear(in_features=4096, out_features=num_classes, bias=True)

        self.gradients = None
    
    def forward(self, x):
        x = self.features_conv1(x)
        x = self.features_conv(x)

        if self.train and x.requires_grad:
            x.register_hook(self.activations_hook)
        
        x = self.max_pool(x)
        x = self.avg_pool(x)
        
        x = x.view(x.size(0), -1)
        x= self.classifier(x)

        return x

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        x = self.features_conv1(x)
        return self.features_conv(x)
    
class BaseCNN(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(BaseCNN, self).__init__()
        
        self.relu = torch.nn.ReLU()

        self.conv_1 = torch.nn.Conv2d(in_channels, 32, kernel_size = (3,3), stride=(1,1), padding=(1,1))
        self.bn_1 = torch.nn.BatchNorm2d(32)

        self.conv_2 = torch.nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn_2 = torch.nn.BatchNorm2d(64)

        self.classifier = torch.nn.Linear(in_features=64*32*32, out_features=num_classes, bias=True)

        self.gradients = None

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.bn_1(x)

        x = self.conv_2(x)
        x = self.relu(x)
        x = self.bn_2(x)
        
        if self.train and x.requires_grad:
            x.register_hook(self.activations_hook)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x
    
    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.bn_1(x)

        x = self.conv_2(x)
        x = self.relu(x)
        x = self.bn_2(x)

        return x
