from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder



class DataClass(Dataset):
    def __init__(self, path, mode='train'):
        super().__init__()
        self.mode = mode
        self.path = path
        self.files = list((Path(self.path + '/' + self.mode).rglob('*.*')))

        self.labels = [Path(path).parent.name for path in self.files]
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.4, 1.0)),
            transforms.RandomApply([transforms.RandomAffine(10, translate=None, scale=None, shear=None)], p=0.5),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.transform_val = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert('RGB')
        if self.mode == 'train':
            image = self.transform(image)

        elif self.mode == 'val':
            image = self.transform_val(image)
        else:
            print('wrong mode: available modes - train, val') 

        label = self.labels[idx]
        return image, label

    def __len__(self):
        return len(self.files)

