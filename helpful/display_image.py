from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from torchvision import transforms
import torchstain
import cv2

# Load the image
# path_img_56x = Path('/mnt/EncryptedDisk1/KidneyData/Studies/KidneyChallenge2024/train/56Nx/12_170/img/56Nx_12_170_713_17408_24576_img.jpg')
# path_mask_str_56x = str(path_img_56x).replace('img', 'mask')
# path_mask_56x = Path(path_mask_str_56x)
#
# path_img_DN = Path('/mnt/EncryptedDisk1/KidneyData/Studies/KidneyChallenge2024/train/DN/11_357/img/DN_11_357_74_8192_3072_img.jpg')
# path_mask_str_DN = str(path_img_DN).replace('img', 'mask')
# path_mask_DN = Path(path_mask_str_DN)
#
# path_img_NEP25 = Path('/mnt/EncryptedDisk1/KidneyData/Studies/KidneyChallenge2024/train/NEP25/08_368_03/img/08_368_03_392_14336_18432_img.jpg')
# path_mask_str_NEP25 = str(path_img_NEP25).replace('img', 'mask')
# path_mask_NEP25 = Path(path_mask_str_NEP25)
#
# path_img_normal = Path('/mnt/EncryptedDisk1/KidneyData/Studies/KidneyChallenge2024/train/normal/normal_F4/img/normal_F4_448_8192_22528_img.jpg')
# path_mask_str_normal = str(path_img_normal).replace('img', 'mask')
# path_mask_normal = Path(path_mask_str_normal)

path_img_56x = '/mnt/EncryptedDisk1/KidneyData/Studies/KidneyChallenge2024/train/56Nx/12_170/img/56Nx_12_170_713_17408_24576_img.jpg'
path_img_DN = '/mnt/EncryptedDisk1/KidneyData/Studies/KidneyChallenge2024/train/DN/11_357/img/DN_11_357_74_8192_3072_img.jpg'
path_img_NEP25 = '/mnt/EncryptedDisk1/KidneyData/Studies/KidneyChallenge2024/train/NEP25/08_368_03/img/08_368_03_392_14336_18432_img.jpg'
path_img_normal = '/mnt/EncryptedDisk1/KidneyData/Studies/KidneyChallenge2024/train/normal/normal_F4/img/normal_F4_448_8192_22528_img.jpg'


# Load the images
img_56x = mpimg.imread(path_img_56x)
# mask_56x = mpimg.imread(path_mask_56x)

img_DN = mpimg.imread(path_img_DN)
# mask_DN = mpimg.imread(path_mask_DN)

img_NEP25 = mpimg.imread(path_img_NEP25)
# mask_NEP25 = mpimg.imread(path_mask_NEP25)

img_normal = mpimg.imread(path_img_normal)
# mask_normal = mpimg.imread(path_mask_normal)

target = cv2.cvtColor(cv2.imread(path_img_56x), cv2.COLOR_BGR2RGB)

normalize = []

for img_path in [path_img_56x,path_img_DN, path_img_NEP25, path_img_normal]:
    to_transform = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])

    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    normalizer.fit(T(target))

    t_to_transform = T(to_transform)
    norm, H, E = normalizer.normalize(I=t_to_transform, stains=True)
    normalize.append(norm)

fig, axes = plt.subplots(4, 2, figsize=(12, 24))

# 56x images
axes[0, 0].imshow(img_56x)
axes[0, 0].set_title('56x Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(normalize[0])
axes[0, 1].set_title('56x Mask')
axes[0, 1].axis('off')

# DN images
axes[1, 0].imshow(img_DN)
axes[1, 0].set_title('DN Image')
axes[1, 0].axis('off')

axes[1, 1].imshow(normalize[1])
axes[1, 1].set_title('DN Mask')
axes[1, 1].axis('off')

# NEP25 images
axes[2, 0].imshow(img_NEP25)
axes[2, 0].set_title('NEP25 Image')
axes[2, 0].axis('off')

axes[2, 1].imshow(normalize[2])
axes[2, 1].set_title('NEP25 Mask')
axes[2, 1].axis('off')

# Normal images
axes[3, 0].imshow(img_normal)
axes[3, 0].set_title('Normal Image')
axes[3, 0].axis('off')

axes[3, 1].imshow(normalize[3])
axes[3, 1].set_title('Normal Mask')
axes[3, 1].axis('off')

plt.tight_layout()
plt.show()
