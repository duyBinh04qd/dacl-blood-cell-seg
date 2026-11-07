from .swin_unet import SWIN_TINY_URL, SwinUnet
from .unext import UNext


# Two UNeXt channel configurations:
#  "dacl"     : (32, 64, 128, 160, 256) - as STATED in the DACL paper -> 1.58M params (measured)
#  "original" : (16, 32, 128, 160, 256) - as in the original UNeXt paper -> 1.47M params
# The DACL paper reports "1.47M" but states the "dacl" channels -> the two are inconsistent;
# 1.47M actually matches the "original" configuration.
UNEXT_DIMS = {"dacl": (32, 64, 128, 160, 256), "original": (16, 32, 128, 160, 256)}


def build_model(name: str, num_classes: int = 2, img_size: int = 224, pretrained: str = "",
                unext_variant: str = "dacl"):
    """Build a model by name.

    name: "unext" or "swin_unet"
    pretrained: swin_unet only. "" = none; "imagenet" = download from Microsoft's URL;
                or a path to a .pth file (e.g. uploaded as a Kaggle Dataset).
    unext_variant: "dacl" or "original" (see UNEXT_DIMS).
    """
    name = name.lower()
    if name == "unext":
        # UNeXt is trained from scratch, no pretrained weights (as in the paper).
        return UNext(num_classes=num_classes, dims=UNEXT_DIMS[unext_variant])
    if name == "swin_unet":
        model = SwinUnet(img_size=img_size, num_classes=num_classes)
        if pretrained:
            model.load_pretrained(SWIN_TINY_URL if pretrained == "imagenet" else pretrained)
        return model
    raise ValueError(f"Unsupported model: {name}")
