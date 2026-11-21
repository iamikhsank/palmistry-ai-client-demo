"""Preview the V1 center crop without changing model preprocessing."""
from io import BytesIO
from PIL import Image, ImageOps


def center_crop_preview(image_bytes):
    """Return the exact center-square region and fraction of source area removed."""
    with Image.open(BytesIO(image_bytes)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        side = min(width, height)
        x = (width - side) // 2
        y = (height - side) // 2
        return image.crop((x, y, x + side, y + side)), 1.0 - side**2 / (width * height)
