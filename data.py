import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ForestDataset(Dataset):

    def __init__(self, dataset_path, image_size):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.paths = [os.path.join(dataset_path, name) for name in os.listdir(dataset_path)]

    def __getitem__(self, item):
        image = Image.open(self.paths[item])
        data = self.transform(image)
        return data, 0

    def __len__(self):
        return len(self.paths)


class LossWriter:
    def __init__(self, save_path):
        self.save_path = save_path

    def add(self, loss, i):
        with open(self.save_path, mode="a") as f:
            term = str(i) + " " + str(loss) + "\n"
            f.write(term)
            f.close()
