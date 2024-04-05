import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = read_image(img_path)
        # 假设标签与图像文件名相关，例如图像文件名为 "cat_01.jpg"，标签为 "cat"
        label = img_name.split('_')[0]
        
        if self.transform:
            image = self.transform(image)

        return image, label

# 使用自定义数据集创建 DataLoader
data_dir = "path/to/your/dataset"
transform = None  # 可以定义自己的图像转换
dataset = CustomDataset(data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 遍历 DataLoader 获取数据
for images, labels in data_loader:
    # 在这里进行模型训练等操作
    pass
