import timm
import torch
import torch.nn as nn

from models.resnet3d import generate_model


class AdvancedClassificationBranch(nn.Module):
    def __init__(self, in_channels=1536, out_channels=2):
        super(AdvancedClassificationBranch, self).__init__()
        # First convolutional layer to reduce channels and add non-linearity
        self.conv1 = nn.Conv2d(in_channels, 1024, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU(inplace=True)

        # Second convolutional layer for further feature processing
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU(inplace=True)

        # Final classification convolutional layer
        self.conv3 = nn.Conv2d(512, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        # Note: Global average pooling is not used here, spatial dimensions remain 7x7
        return x


class CRCModel(nn.Module):

    def __init__(self, model_name='convnextv2_large.fcmae_ft_in22k_in1k', num_classes=2, loc=False, align=False, test_grad_cam=False):
        super().__init__()

        self.loc = loc  # location
        self.align = align  # alignment
        self.test_grad_cam = test_grad_cam
        timm_num_classes = 0 if self.loc else num_classes

        # resnet50
        if model_name == 'resnet':
            model_name = 'resnet50'
            file = 'models/pretrain/resnet50.safetensors'
            self.model = timm.create_model(model_name=model_name, pretrained=True, num_classes=timm_num_classes, in_chans=1,
                                           pretrained_cfg_overlay=dict(file=file))
        # convnextv2
        else:
            file = 'models/pretrain/convnextv2_large.safetensors'
            self.model = timm.create_model(model_name=model_name, pretrained=True, num_classes=timm_num_classes, in_chans=1,
                                           pretrained_cfg_overlay=dict(file=file))
        # resnet3d
        self.model_3d = generate_model(model_type='resnet', model_depth=50, resnet_shortcut='B', num_classes=num_classes)
        self.model_3d.load_state_dict(torch.load('models/pretrain/resnet_50_23dataset.pth'), strict=False)

        # Classification branch
        self.classifier = nn.Linear(self.model.num_features, num_classes)
        # # Patch classification branch, 1536-dimensional feature vector as a Patch
        # self.patch_cls = AdvancedClassificationBranch()

        # Localization branch
        self.localizer = nn.Sequential(
            nn.Linear(self.model.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4)  # [x_min, y_min, x_max, y_max]
        )

    def forward(self, x1, x2=None, mode=None):
        # x1: image (B, C, H, W) or (1, D, C, H, W)
        # x2: volume (B, 1) or (B, C, H, W, 2*slice+1) / or (1, D, C, H, W, 2*slice+1)
        # New requirements added in the volume dimension
        if self.test_grad_cam:
            if self.loc:
                out1 = self.model.forward(x1)
                out1_cls = self.classifier(out1)  # cls
                out1_loc = self.localizer(out1)  # loc
                return out1_cls, out1_loc
            else:
                out1 = self.model.forward_features(x1)  # features
                out1_cls = self.model.forward_head(out1)  # [1, 2]
                return out1_cls
        else:
            x2_dim = x2.dim()
            if x1.dim() == 5:
                x1, x2 = x1[0], x2[0]
            # x1 torch.Size([128, 1, 224, 224]), x2 torch.Size([128, 1, 224, 224, 3])
            # image
            if self.loc:
                out1 = self.model.forward(x1)  # [1, 1536]
                out1_cls = self.classifier(out1)  # [1, 2]
            else:
                out1 = self.model.forward_features(x1)  # [1, 1536, 7, 7]
                out1_cls = self.model.forward_head(out1)  # [1, 2]

            if x2_dim == 2:
                if self.loc:
                    out1_loc = self.localizer(out1)  # loc
                    return out1_cls, out1_loc
                else:
                    return out1_cls

            # volume
            if self.align:
                out2_cls = self.model_3d.forward(x2[:, :, :, :, :3])
                x2_A, x2_P = x2[:, :, :, :, 3], x2[:, :, :, :, 4]
                out2_A = self.model.forward(x2_A)
                out2_P = self.model.forward(x2_P)
            else:
                out2_cls = self.model_3d.forward(x2)

            out_cls = (out1_cls + out2_cls) / 2
            if self.loc:
                out1_loc = self.localizer(out1)  # loc
                if self.align:
                    return out_cls, out1_loc, out1, out2_A, out2_P
                else:
                    return out_cls, out1_loc
            else:
                if self.align:
                    return out_cls, out1, out2_A, out2_P
                else:
                    return out_cls
