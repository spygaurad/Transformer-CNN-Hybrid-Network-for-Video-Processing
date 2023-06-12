from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms



class CustomDataset(Dataset):

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data.iloc[index, 0]
        print(image_path)
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image)
        return image_tensor



class CustomDataLoader:

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def get_data(self):
        train_csv = "Datasets/image2image/train.csv"
        val_csv = "Datasets/image2image/val.csv"
        test_csv = "Datasets/image2image/test.csv"

        train_dataset = CustomDataset(train_csv)
        val_dataset = CustomDataset(val_csv)
        test_dataset = CustomDataset(test_csv)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader, val_loader, test_loader

    
# train_data, val_data, test_data = CustomDataLoader(4).get_data()