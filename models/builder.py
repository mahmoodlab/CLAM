import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
from torchvision import models, transforms

def has_CONCH():
    HAS_CONCH = False
    HF_AUTH_TOKEN = ''
    # check if conch is installed and HF token is available
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        # check if HF_AUTH_TOKEN is set (optional, can be empty for public models)
        HF_AUTH_TOKEN = os.environ.get('HF_AUTH_TOKEN', '')
        HAS_CONCH = True
    except Exception as e:
        print(e)
        print('CONCH not installed')
    return HAS_CONCH, HF_AUTH_TOKEN

def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = ''
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH
        
def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    img_transforms = None
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()
    elif model_name == 'uni_v1':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == 'conch_v1':
        HAS_CONCH, HF_AUTH_TOKEN = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained(
            'conch_ViT-B-16',
            "hf_hub:MahmoodLab/conch",
            hf_auth_token=HF_AUTH_TOKEN
        )
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    elif model_name == 'gigapath':
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        img_transforms = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    elif model_name == 'conch_v1_5':
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError("Please install huggingface transformers (e.g. 'pip install transformers') to use CONCH v1.5")
        titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        model, _ = titan.return_conch()
        assert target_img_size == 448, 'TITAN is used with 448x448 CONCH v1.5 features'
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    
    print(model)
    if not img_transforms:
        constants = MODEL2CONSTANTS[model_name]
        img_transforms = get_eval_transforms(mean=constants['mean'],
                                            std=constants['std'],
                                            target_img_size = target_img_size)

    return model, img_transforms