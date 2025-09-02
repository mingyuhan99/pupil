import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import distance as dist


class AdaptiveEyeNet6Layers(nn.Module):
    def __init__(self, num_landmarks=8):
        super().__init__()
        layers_config = [
            (64, 7, 2),
            (128, 5, 2),
            (192, 3, 2),
            (256, 3, 2),
            (320, 3, 2),
            (384, 3, 2),
        ]
        self.layers = nn.ModuleList()
        in_channels = 1
        for out_channels, kernel_size, stride in layers_config:
            self.layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                )
            )
            self.layers.append(nn.InstanceNorm2d(out_channels))
            self.layers.append(nn.LeakyReLU(0.1))
            in_channels = out_channels
        with torch.no_grad():
            x = torch.zeros(1, 1, 192, 192)
            for layer in self.layers:
                x = layer(x)
        self.conv_out_size = x.view(1, -1).size(1)
        self.fc1 = nn.Linear(self.conv_out_size, 1024)
        self.fc2 = nn.Linear(1024, num_landmarks * 2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = self.fc2(x)
        return x


def calculate_ear(landmarks):
    p1, p4, p2, p6, p3, p5 = (
        landmarks[0],
        landmarks[4],
        landmarks[1],
        landmarks[7],
        landmarks[3],
        landmarks[5],
    )
    ver1, ver2, hor = (
        dist.euclidean(p2, p6),
        dist.euclidean(p3, p5),
        dist.euclidean(p1, p4),
    )
    return (ver1 + ver2) / (2.0 * hor) if hor > 0 else 0


def apply_camera_rotation(frame, position):
    """카메라 위치별 회전 적용"""
    if position == "left_below":
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif position == "left_side":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif position == "right_side":
        temp_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return cv2.flip(temp_frame, 1)
    elif position == "right_below":
        temp_frame = cv2.rotate(frame, cv2.ROTATE_180)
        return cv2.flip(temp_frame, 1)
    else:
        return frame
