"""
Implementation of Brady Zhou's cross view transformer
"""

import torch.nn as nn
from einops import rearrange
from opencood.models.sub_modules.fax_modules import FAXModule
from opencood.models.backbones.resnet_ms import ResnetEncoder
from opencood.models.sub_modules.naive_decoder import NaiveDecoder


class FaxFusedTransformer(nn.Module):
    def __init__(self, config):
        super(FaxFusedTransformer, self).__init__()
        # encoder params
        self.encoder = ResnetEncoder(config['encoder'])

        # cvm params
        cvm_params = config['fax']
        cvm_params['backbone_output_shape'] = self.encoder.output_shapes
        self.fax = FAXModule(cvm_params)

        # decoder params
        decoder_params = config['decoder']

        self.decoder = NaiveDecoder(decoder_params)

        self.cls_head = nn.Conv2d(256, config['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(256, 7 * config['anchor_number'],
                                  kernel_size=1)
        self.return_features = False

    def set_return_features(self):
        self.return_features = True

    def forward(self, batch_dict):
        print("Original camera shape:", batch_dict['camera'].shape)

        # **如果 `camera` 只有 (B, L)，扩展成 (B, L, H, W, C)**
        if batch_dict['camera'].dim() == 2:  # 如果 shape 是 (B, 4)
            M, H, W, C = 2, 512, 512, 3  # **确保 H, W, C 是你数据的真实大小**
            batch_dict['camera'] = batch_dict['camera'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, L, 1, 1, 1, 1)
            batch_dict['camera'] = batch_dict['camera'].expand(-1, -1, M, H, W, C)  # (B, L, H, W, C)

        print("Camera shape AFTER HWC processing:", batch_dict['camera'].shape)

        # batch_dict['camera'] = batch_dict['camera'].unsqueeze(1)

        if batch_dict['camera'].dim() > 6:
            batch_dict['camera'] = batch_dict['camera'].squeeze(-1)

        batch_dict['intrinsic'] = batch_dict['intrinsic'].unsqueeze(1)
        batch_dict['extrinsic'] = batch_dict['extrinsic'].unsqueeze(1)
        x = batch_dict['camera']
        print("Camera shape AFTER 6D processing:", x.shape)





        if x.dim() != 6:
            raise ValueError(f"Expected 6D tensor, but got shape {x.shape}")

        b, l, m, _, _, _ = x.shape
        # (3,1,4,512,512,3)
        # (2,1,4,512,512,3)
        x = self.encoder(x)
        batch_dict.update({'features': x})
        # (B, 1, 128, 32, 32)
        x = self.fax(batch_dict)

        # dynamic head
        # (B, 1, 32, 128, 128)
        x = self.decoder(x)

        # (BL, 32, 128, 128)
        x = rearrange(x, 'b l c h w -> (b l) c h w')
        if self.return_features:
            return x

        psm = self.cls_head(x)
        rm = self.reg_head(x)
        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict


if __name__ == '__main__':
    import os
    import torch
    from opencood.hypes_yaml.yaml_utils import load_yaml

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    test_data = torch.rand(1, 2, 4, 512, 512, 3)
    test_data = test_data.cuda()

    extrinsic = torch.rand(1, 2, 4, 4, 4)
    intrinsic = torch.rand(1, 2, 4, 3, 3)

    extrinsic = extrinsic.cuda()
    intrinsic = intrinsic.cuda()

    params = load_yaml('../hypes_yaml/opcamera/fax.yaml')

    model = FaxFusedTransformer(params['model']['args'])
    model = model.cuda()
    while True:
        output = model({'inputs': test_data,
                        'extrinsic': extrinsic,
                        'intrinsic': intrinsic})
        print('test_passed')
