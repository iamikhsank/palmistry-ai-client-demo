import io
import unittest

from PIL import Image
from modules.crop_preview import center_crop_preview


class CropPreviewTests(unittest.TestCase):
    def encoded(self, size):
        image = Image.new("RGB", size, "white")
        data = io.BytesIO()
        image.save(data, format="PNG")
        return data.getvalue()

    def test_square_no_loss(self):
        crop, removed = center_crop_preview(self.encoded((100, 100)))
        self.assertEqual(crop.size, (100, 100))
        self.assertEqual(removed, 0)

    def test_portrait_crop(self):
        crop, removed = center_crop_preview(self.encoded((100, 200)))
        self.assertEqual(crop.size, (100, 100))
        self.assertAlmostEqual(removed, 0.5)

    def test_landscape_odd_dimensions(self):
        crop, removed = center_crop_preview(self.encoded((201, 100)))
        self.assertEqual(crop.size, (100, 100))
        self.assertAlmostEqual(removed, 1 - 10000 / 20100)


if __name__ == "__main__":
    unittest.main()
