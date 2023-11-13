import imgaug.augmenters as iaa
from torchvision import transforms
from fnnaug.transform.base import ToTensor, PadSquare, RelativeLabels, AbsoluteLabels, ImgAug


class DefaultAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Sharpen((0.0, 0.1)),  # 锐化
            iaa.Affine(rotate=(-0, 0), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),  # 平移、旋转、缩放、裁切为梯形
            iaa.AddToBrightness((-60, 40)),  # 亮度调整
            iaa.AddToHue((-10, 10)),  # 随机调整 Hue（色调、色相）。HSV分别代表色调、饱和度、亮度
            iaa.Fliplr(0.5),  # 翻转概率
        ])


class StrongAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-20, 20)),
            iaa.Fliplr(0.5),
        ])


AUGMENTATION_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(),
    DefaultAug(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])
