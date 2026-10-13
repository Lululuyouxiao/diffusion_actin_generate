import os
import random
from torch.utils import data
from torchvision import transforms as T
from PIL import Image, ImageEnhance
import config

params = config.parse_args()


class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=256, mode='train', augmentation_prob=0.4):
        self.mode = mode
        self.root = root
        self.GT_paths = root[:-1]
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        if self.mode == 'train' or self.mode == 'valid':
            self.GT_paths_test = list(map(lambda k: os.path.join(self.GT_paths, k), os.listdir(self.GT_paths)))
        else:
            self.GT_paths_test = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.image_size = image_size
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        GT_path = self.GT_paths_test[index]
        image = Image.open(image_path)
        GT = Image.open(GT_path)
        if params.RGB is True:
            image = image.convert('RGB')
        # aspect_ratio = image.size[1] / image.size[0]
        Transform = []
        # ResizeRange = random.randint(300,320)
        #Transform.append(T.Resize((self.image_size, self.image_size)))
        p_transform = random.random()
        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            # Transform.append(T.RandomCrop(self.image_size))
            # Transform = T.Compose(Transform)
            # image = Transform(image)
            # GT = Transform(GT)
            Transform = []
            if params.LV2_augmentation is True:
                # contract
                enh_con = ImageEnhance.Contrast(image)
                con_factor = random.random()
                image = enh_con.enhance(con_factor)
                # brightness
                enh_bri = ImageEnhance.Brightness(image)
                bri_factor = random.random()
                image = enh_bri.enhance(bri_factor)

            rotate = random.random()
            if rotate < 0.25:
                image = image.transpose(Image.Transpose.ROTATE_90)
                GT = GT.transpose(Image.Transpose.ROTATE_90)
            elif 0.5 > rotate > 0.25:
                image = image.transpose(Image.Transpose.ROTATE_180)
                GT = GT.transpose(Image.Transpose.ROTATE_180)
            elif 0.75 > rotate > 0.5:
                image = image.transpose(Image.Transpose.ROTATE_270)
                GT = GT.transpose(Image.Transpose.ROTATE_270)
            # Transform = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.0)
            # GT = Transform(GT)
            flip_left_right = random.random()
            if flip_left_right < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                GT = GT.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            flip_up_down = random.random()
            if flip_up_down < 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                GT = GT.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        image = Transform(image)
        GT = Transform(GT)
        if params.RGB is True:
            Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            image = Norm_(image)
        return image, GT

    def __len__(self):
        return len(self.image_paths)


def get_loader(image_path, image_size, batch_size, shuffle=True, num_workers=0, mode='train', augmentation_prob=0.4):
    dataset = ImageFolder(root=image_path, image_size=image_size, mode=mode, augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  drop_last=True, pin_memory=False)
    return data_loader
