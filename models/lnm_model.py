import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import resnet3d


class LNMModel(nn.Module):

    def __init__(self, model_name='resnet', num_classes=2, loc=False):
        super().__init__()

        # loc
        self.loc = loc
        timm_num_classes = 0 if self.loc else num_classes

        # resnet50
        if model_name == 'resnet':
            model_name = 'resnet50'
            file = 'models/pretrain/resnet50.safetensors'
            self.model = timm.create_model(model_name=model_name, pretrained=True, num_classes=timm_num_classes, in_chans=1,
                                           pretrained_cfg_overlay=dict(file=file))
        elif model_name == 'vgg':
            model_name = 'vgg16.tv_in1k'
            self.model = timm.create_model(model_name=model_name, pretrained=True, num_classes=timm_num_classes, in_chans=1)
        # convnextv2
        else:
            model_name = 'convnextv2_large.fcmae_ft_in22k_in1k'
            file = 'models/pretrain/convnextv2_large.safetensors'
            self.model = timm.create_model(model_name=model_name, pretrained=True, num_classes=timm_num_classes, in_chans=1,
                                           pretrained_cfg_overlay=dict(file=file))

        # LNM localization
        self.localizer = nn.Sequential(
            nn.Linear(self.model.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4)  # [x_min, y_min, x_max, y_max]
        )

    def forward(self, x):
        if x.dim() == 5:  # Patient-level input
            x = x[0]

        if self.loc:
            x = self.model.forward(x)
            x = self.localizer(x)
        else:
            x = self.model.forward_features(x)  # [1, 1536, 7, 7]
            x = self.model.forward_head(x)  # [1, 3]

        return x


class LNMDualModel(nn.Module):

    def __init__(self, model_name='resnet', num_classes=2):
        super().__init__()

        # resnet50
        if model_name == 'resnet':
            model_name = 'resnet50'
        # convnextv2
        else:
            model_name = 'convnextv2_large.fcmae_ft_in22k_in1k'

        # Load custom pre-trained weights
        self.model1 = timm.create_model(model_name=model_name, pretrained=False, num_classes=0, in_chans=1)
        self.model1.load_state_dict(torch.load('run/ckpt/model_best.pth'), strict=False)

        self.model2 = timm.create_model(model_name=model_name, pretrained=False, num_classes=0, in_chans=1)
        self.model2.load_state_dict(torch.load('run/ckpt/model_best.pth'), strict=False)

        # Classification layer after concatenation
        self.classifier = nn.Sequential(
            nn.Linear(self.model1.num_features*2, 512),  # After concatenation, feature dimensions are 1024 (512 + 512)
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if x.dim() == 6:  # Patient-level input
            x = x[0]

        # Method 1
        x1 = self.model1.forward(x[:, 0, :, :, :])  # CRC
        x2 = self.model2.forward(x[:, 1, :, :, :])  # LNM

        combined_features = torch.cat((x1, x2), dim=1)  # Concatenated features with feature_dim * 2 dimensions
        output = self.classifier(combined_features)

        return output


class LNM3DModel(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        self.model_3d = resnet3d.generate_model(model_type='resnet', model_depth=50, resnet_shortcut='B', num_classes=num_classes)
        self.model_3d.load_state_dict(torch.load('models/pretrain/resnet_50_23dataset.pth'), strict=False)

    def forward(self, x):
        x = self.model_3d.forward(x)

        return x
