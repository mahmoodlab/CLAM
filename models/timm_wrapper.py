import torch
import timm

# class TimmCNNEncoder(torch.nn.Module):
#     def __init__(self, model_name: str = 'resnet50.tv_in1k', 
#                  kwargs: dict = {'features_only': True, 'out_indices': (3,), 'pretrained': True, 'num_classes': 0}, 
#                  pool: bool = True):
#         super().__init__()
#         assert kwargs.get('pretrained', False), 'only pretrained models are supported'
#         self.model = timm.create_model(model_name, **kwargs)
#         self.model_name = model_name
#         if pool:
#             self.pool = torch.nn.AdaptiveAvgPool2d(1)
#         else:
#             self.pool = None
class TimmCNNEncoder(torch.nn.Module):
    def __init__(self,
                 model_name: str = 'resnet50.tv_in1k',
                 kwargs: dict = {'features_only': True,
                                'out_indices': (3,),
                                'pretrained': True,
                                'num_classes': 0},
                 pool: bool = True):
        super().__init__()

        # 1. 备份并强制“无权重创建网络”
        pretrained = kwargs.get('pretrained', False)
        kwargs['pretrained'] = False          # 不再去 HF Hub
        self.model = timm.create_model(model_name, **kwargs)
        self.model_name = model_name

        # 2. 再手工载入本地权重
        if pretrained:
            # 根据你实际路径填写
            local_ckpt = '/home/cuiping/CLAM/resnet50_tv_in1k/pytorch_model.bin'
            state_dict = torch.load(local_ckpt, map_location='cpu')
            # 去掉 timm 在 keys 里加的 “model.” 前缀（如有）
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None
    def forward(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out