import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

from torch.utils.data import Dataset
from PIL import Image

import torchvision.transforms.functional as F

# ----------------------------
# Dataset
# ----------------------------


class SupermarketSemSeg(Dataset):
    """
    Expects a folder layout like:
        root/
          images/
            xxx.jpg|png
          masks/
            xxx_mask.png   # categorical mask (uint8 or uint16), same stem

    If your masks are 16-bit (raw COCO IDs), set mask_mode="uint16".
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        mask_suffix: str = "_mask",
        mask_mode: str = "uint8",  # or "uint16"
        include_filenames: bool = False,
        transform=None
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mask_suffix = mask_suffix
        self.mask_mode = mask_mode
        self.include_filenames = include_filenames
        self.transform = transform

        assert self.images_dir.exists(), f"Missing {self.images_dir}"
        assert self.masks_dir.exists(), f"Missing {self.masks_dir}"

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        self.items: List[Tuple[Path, Path]] = []

        # Pair images with masks by stem
        mask_ext = ".png"  # images have png format
        for img_path in sorted(self.images_dir.iterdir()):
            if img_path.suffix.lower() not in exts:
                continue
            stem = img_path.stem
            mask_name = f"{stem}{self.mask_suffix}{mask_ext}"
            mask_path = self.masks_dir / mask_name
            if mask_path.exists():
                self.items.append((img_path, mask_path))
            else:
                # If you want to enforce strict pairing, raise an error here
                # raise FileNotFoundError(f"Mask not found for {img_path}")
                pass

        if not self.items:
            raise RuntimeError(
                "No image/mask pairs found. Check paths and naming.")

        # Optional: load mapping (contiguous labels) if present
        self.label_mapping: Optional[List[Dict]] = None
        mapping_path = self.masks_dir.parent / "label_mapping.json"
        if mapping_path.exists():
            import json
            with open(mapping_path, "r") as f:
                self.label_mapping = json.load(f)

    def __len__(self):
        return len(self.items)

    def _read_image(self, p: Path) -> Image.Image:
        # Opens an image as an Image object (in RGB format)
        img = Image.open(p).convert("RGB")
        return img

    def _read_mask(self, p: Path) -> Image.Image:
        # If category_id masks are 16-bit PNG, we preserve bit depth.
        if self.mask_mode == "uint16":
            # PIL loads 16-bit PNG as mode "I;16". It'll be cast later.
            m = Image.open(p)
            if m.mode != "I;16":
                m = m.convert("I;16")
        else:
            # 8-bit categorical mask
            # Our masks are stored in this format!
            m = Image.open(p).convert("L")  # uint8
        return m

    def __getitem__(self, idx: int):
        img_path, mask_path = self.items[idx]

        img = self._read_image(img_path)
        mask = self._read_mask(mask_path)
        if self.transform:
            img_t = self.transform(img)
        else:
            # Transform image to tensor and mask to uint8 mask:
            img_t = F.to_tensor(img)
        mask_t = F.pil_to_tensor(mask).squeeze(0)

        if self.include_filenames:
            return {
                "image": img_t,         # float32 [C,H,W]
                "mask": mask_t,         # long [H,W], class indices
                "image_path": str(img_path),
                "mask_path": str(mask_path),
            }
        return img_t, mask_t
