import torch
import torch.nn as nn
import torch.nn.functional as F

# Wave-U-Net 모델 정의
class WaveUNet(nn.Module):
    def __init__(self, num_channels=4, num_layers=5, base_filter_size=24, kernel_size=15):
        super(WaveUNet, self).__init__()
        
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.base_filter_size = base_filter_size
        
        # 인코더 레이어 정의
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(num_layers):
            in_channels = num_channels if i == 0 else base_filter_size * (2**(i-1))
            out_channels = base_filter_size * (2**i)
            self.encoders.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=2, padding=(kernel_size // 2))
            )

        self.bottle_neck = nn.Conv1d(base_filter_size*2**(num_layers-1), base_filter_size*2**(num_layers-1), kernel_size, stride=1, padding=(kernel_size // 2))

        # 디코더 레이어 정의
        for i in range(num_layers-1, -1, -1):
            in_channels = base_filter_size * (2**i) * 2 if i != num_layers-1 else base_filter_size * (2**i)
            out_channels = base_filter_size * (2**(i-1)) if i > 0 else num_channels
            self.decoders.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size // 2))
            )
            

        self.final_conv = nn.Conv1d(num_channels, num_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        
        # 인코더 단계
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            if i != len(self.encoders) - 1:  # 마지막 레이어는 스킵하지 않음
                skips.append(x)
                x = F.leaky_relu(x)

        x = self.bottle_neck(x)

        # 디코더 단계
        for i, decoder in enumerate(self.decoders):
            if i != 0:
                x = torch.cat([x, skips[-(i)]], dim=1)  # 스킵 연결
            x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
            x = decoder(x)

            if i != len(self.decoders) - 1:
                x = F.leaky_relu(x)

        # 출력 레이어
        output = F.tanh(self.final_conv(x))
        
        return output