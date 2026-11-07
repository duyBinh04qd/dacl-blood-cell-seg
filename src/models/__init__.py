from .swin_unet import SWIN_TINY_URL, SwinUnet
from .unext import UNext


# Hai cấu hình số kênh của UNeXt:
#  "dacl"     : (32, 64, 128, 160, 256) - như bài báo DACL KHAI BÁO  -> 1.58M tham số (đo thực tế)
#  "original" : (16, 32, 128, 160, 256) - như bài báo UNeXt gốc      -> 1.47M tham số
# Bài báo DACL ghi "1.47M" nhưng lại khai báo số kênh của cấu hình "dacl" -> hai thông tin
# mâu thuẫn; con số 1.47M thực ra khớp với cấu hình "original".
UNEXT_DIMS = {"dacl": (32, 64, 128, 160, 256), "original": (16, 32, 128, 160, 256)}


def build_model(name: str, num_classes: int = 2, img_size: int = 224, pretrained: str = "",
                unext_variant: str = "dacl"):
    """Tạo mô hình theo tên.

    name: "unext" hoặc "swin_unet"
    pretrained: chỉ dùng cho swin_unet. "" = không nạp; "imagenet" = tải từ URL của Microsoft;
                hoặc đường dẫn tới file .pth (ví dụ đã upload thành Kaggle Dataset).
    unext_variant: "dacl" hoặc "original" (xem UNEXT_DIMS).
    """
    name = name.lower()
    if name == "unext":
        # UNeXt huấn luyện từ đầu, không dùng trọng số tiền huấn luyện (giống bài báo).
        return UNext(num_classes=num_classes, dims=UNEXT_DIMS[unext_variant])
    if name == "swin_unet":
        model = SwinUnet(img_size=img_size, num_classes=num_classes)
        if pretrained:
            model.load_pretrained(SWIN_TINY_URL if pretrained == "imagenet" else pretrained)
        return model
    raise ValueError(f"Mô hình không hỗ trợ: {name}")
