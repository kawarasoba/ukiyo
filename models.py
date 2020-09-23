import torch.nn as nn
import torchvision.models as torch_models
import pretrainedmodels
from efficientnet_pytorch import EfficientNet

def vgg11_bn(num_classes=1000,pretrained=False):
    model = torch_models.vgg11_bn(pretrained=pretrained)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model

def vgg19_bn(num_classes=1000,pretrained=False):
    model = torch_models.vgg19_bn(pretrained=pretrained)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model

def resnet50(num_classes=1000,pretrained=False):
    model = torch_models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def se_resnet50(num_classes=1000,pretrained=None):
    model = pretrainedmodels.se_resnet50(num_classes=num_classes,pretrained=pretrained)
    return model

def se_resnext50_32x4d(num_classes=1000,pretrained=None):
    model = pretrainedmodels.se_resnext50_32x4d(num_classes=num_classes,pretrained=pretrained)
    return model

def se_resnet152(num_classes=1000,pretrained=None):
    model = pretrainedmodels.se_resnet152(num_classes=num_classes,pretrained=pretrained)
    return model

def densenet161(num_classes=1000,pretrained=False):
    model = torch_models.densenet161(pretrained=pretrained,num_classes=num_classes)
    return model

def dualpathnet98(num_classes=1000,pretrained=None):
    model = pretrainedmodels.dpn98(num_classes=num_classes,pretrained=pretrained)
    return model

def efficientnet_b1(num_classes=1000,pretrained=False):
    if pretrained:  model = EfficientNet.from_pretrained('efficientnet-b1')
    else:           model = EfficientNet.from_name('efficientnet-b1')
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    return model

def efficientnet_b3(num_classes=1000,pretrained=False):
    if pretrained:  model = EfficientNet.from_pretrained('efficientnet-b3')
    else:           model = EfficientNet.from_name('efficientnet-b3')
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    return model

def senet154(num_classes=1000,pretrained=None):
    model = pretrainedmodels.senet154(num_classes=num_classes,pretrained=pretrained)
    return model

def inceptionv3(num_classes=1000,pretrained=None):
    model = torch_models.inception_v3(pretrained=pretrained,num_classes=num_classes)
    #model.AuxLogits = InceptionAux(model.AuxLogits.in_features, num_classes)
    #model.fc = nn.Linear(model.fc.in_features, num_classes)
    # model = pretrainedmodels.inceptionv3(num_classes=num_classes,pretrained=pretrained)
    return model

def pnasnet5large(num_classes=1000,pretrained=None):
    model = pretrainedmodels.pnasnet5large(num_classes=num_classes,pretrained=pretrained)
    return model

def get_model(model_name='', num_classes=1000, pretrained=False):
    if model_name=='vgg11_bn':
        return vgg11_bn(num_classes, pretrained)
    elif model_name=='vgg19_bn':
        return vgg19_bn(num_classes, pretrained)
    elif model_name=='resnet50':
        return resnet50(num_classes, pretrained)
    elif model_name=='se_resnet50':
        if not pretrained: pretrained = None
        return se_resnet50(num_classes, pretrained)
    elif model_name=='se_resnext50':
        if not pretrained: pretrained = None
        return se_resnext50_32x4d(num_classes, pretrained)
    elif model_name=='se_resnet152':
        if not pretrained: pretrained = None
        return se_resnet152(num_classes, pretrained)    
    elif model_name=='densenet161':
        return densenet161(num_classes, pretrained)
    elif model_name=='dualpathnet98':
        if not pretrained: pretrained = None
        return dualpathnet98(num_classes, pretrained)
    elif model_name=='efficientnet_b1':
        return efficientnet_b1(num_classes, pretrained)
    elif model_name=='efficientnet_b3':
        return efficientnet_b3(num_classes, pretrained)
    elif model_name=='senet154':
        if not pretrained: pretrained = None
        return senet154(num_classes,pretrained)
    elif model_name=='inceptionv3':
        if not pretrained: pretrained = None
        return inceptionv3(num_classes,pretrained)
    elif model_name=='pnasnet5large':
        return pnasnet5large(num_classes,pretrained)
