import os
import PIL
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tifffile
import matplotlib.pyplot as plt
import torchvision.transforms as T

class SegDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=None, img_size=(352, 352)):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform
        self.img_size = img_size

        self.image_dir = os.path.join(root_dir, phase, 'images')
        self.mask_dir = os.path.join(root_dir, phase, 'masks')


        self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png', '.tif', '.bmp'))])


        self.masks = []
        for img_name in self.images.copy():

            base_name = os.path.splitext(img_name)[0]

            possible_mask_names = [
                f"{base_name}_anno.bmp",
                f"{base_name}.png",
                f"{base_name}_mask.png",
                f"{base_name}_anno.png",
                f"{base_name}_mask.tif",
                f"{base_name}.tif",
                f"{base_name}_mask.bmp",
            ]
            mask_name = None
            for candidate in possible_mask_names:
                if os.path.exists(os.path.join(self.mask_dir, candidate)):
                    mask_name = candidate
                    break
            if mask_name:
                self.masks.append(mask_name)
            else:
                print(f"No mask found for image {img_name}, skipping...")
                self.images.remove(img_name)

        assert len(self.images) == len(self.masks), \
            f"Number of images ({len(self.images)}) and masks ({len(self.masks)}) don't match"

        print(f"Found {len(self.images)} images in {phase} set")
        # print(f"Sample images: {self.images[:5]}")
        # print(f"Sample masks: {self.masks[:5]}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])


        try:
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
        except (PIL.UnidentifiedImageError, ValueError) as e:
            try:
                image = tifffile.imread(img_path)
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)
                elif image.shape[-1] != 3:
                    image = image[..., :3]
                image = Image.fromarray(image).convert('RGB')
                mask = tifffile.imread(mask_path)
                if mask.ndim > 2:
                    mask = mask[..., 0]
                mask = Image.fromarray(mask).convert('L')
            except Exception as e2:
                print(f"Error loading {img_path} or {mask_path}: {e}, {e2}")
                return self.__getitem__((idx + 1) % len(self))

        image = np.array(image)
        mask = np.array(mask)


        if mask.max() > 1:
            mask = (mask == 255).astype(np.float32)
        else:
            mask = mask.astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            if isinstance(mask, torch.Tensor) and len(mask.shape) == 2:
                mask = mask.unsqueeze(0)

        return {
            'image': image,
            'mask': mask,
            'image_path': img_path
        }

def get_transforms(phase, img_size=(352, 352)):
    if phase == 'train':
        return A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
            ], p=0.3),
            A.OneOf([A.GaussNoise(p=0.5), A.RandomBrightnessContrast(p=0.5), A.ColorJitter(p=0.5)], p=0.3),

            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ], p=1.0)
    else:
        return A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),

            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ], p=1.0)

def get_test_dataloaders(data_dir, batch_size=1, num_workers=0, img_size=(256, 256)):
    test_dataset = SegDataset(root_dir=data_dir, phase='test', transform=get_transforms('test', img_size),
                              img_size=img_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             drop_last=True)
    return test_loader

def get_dataloaders(data_dir, batch_size=1, num_workers=0, img_size=(256, 256)):
    train_dataset = SegDataset(root_dir=data_dir, phase='train', transform=get_transforms('train', img_size),
                               img_size=img_size)
    val_dataset = SegDataset(root_dir=data_dir, phase='val', transform=get_transforms('val', img_size),
                             img_size=img_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    return train_loader, val_loader

def visualize_batch(batch, num_samples=4):
    images = batch['image'][:num_samples]
    masks = batch['mask'][:num_samples]


    images = (images * 255).byte()

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    for i in range(num_samples):
        axes[i, 0].imshow(T.ToPILImage()(images[i]))
        axes[i, 0].set_title(f"Image {i + 1}")
        axes[i, 0].axis('off')
        axes[i, 1].imshow(masks[i].squeeze(), cmap='gray')
        axes[i, 1].set_title(f"Mask {i + 1}")
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_dir = ""
    batch_size = 4
    img_size = (256, 256)

    train_loader, val_loader = get_dataloaders(data_dir, batch_size, img_size=img_size)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    for batch in train_loader:
        images = batch['image']
        masks = batch['mask']
        print(f"Batch image shape: {images.shape}")
        print(f"Batch mask shape: {masks.shape}")
        print(f"Image data range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Mask data range: [{masks.min():.3f}, {masks.max():.3f}]")
        visualize_batch(batch)
        break