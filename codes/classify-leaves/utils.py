import torch as pt
import pandas as pd
import cv2
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from lib.utils import *


# 对于训练集，getitem返回(image, label)
# 对于测试集，getitem返回image
class Leaves(Dataset):
    def __init__(self, train, df: pd.DataFrame, transform: torchvision.transforms):
        self.train = train
        self.transform = transform
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = cv2.imread(str(data_path / "classify-leaves" / self.df.iat[idx, 0]))
        img = self.transform(img)
        if self.train:
            return img, self.df.iat[idx, 1]
        else:
            return img


normalize = transforms.Normalize([0.7590, 0.7780, 0.7579], [0.2560, 0.2274, 0.2401])
train_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])


# 返回LabelEncoder，train dataset，cv dataset，test dataset，test image filename
# train dataset 返回的是 transformed cv2 image, encoded label
# test dataset 返回的是 transformed cv2 image
def load_leaves(cv_frac=.3):
    __le = LabelEncoder()
    train_df = pd.read_csv(data_path / 'classify-leaves/train.csv')
    test_df = pd.read_csv(data_path / 'classify-leaves/test.csv')
    encoded = __le.fit_transform(train_df['label'])
    train_df['label'] = encoded

    cv_df = train_df.sample(frac=cv_frac)
    train_df.drop(cv_df.index, inplace=True)
    return __le, \
           Leaves(True, train_df, train_transform), \
           Leaves(True, cv_df, test_transform), \
           Leaves(False, test_df, test_transform), \
            test_df['image']


def plot_samples(sample_size, dataset: Dataset):
    import matplotlib.pyplot as plt
    sample_loader = DataLoader(dataset, sample_size ** 2, True)
    fig, ax_array = plt.subplots(nrows=sample_size, ncols=sample_size, sharey=True, sharex=True, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.4)
    imgs, labels = next(iter(sample_loader))
    for r in range(sample_size):
        for c in range(sample_size):
            ax_array[r, c].imshow(imgs[sample_size * r + c].permute(1, 2, 0).flip(2))
            ax_array[r, c].text(.5, -.25, str(labels[sample_size * r + c].item()), horizontalalignment='center',
                                transform=ax_array[r, c].transAxes)
            plt.xticks(None)  # 隐藏坐标轴
            plt.yticks(None)  # 隐藏坐标轴


def get_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=8, num_workers=num_workers)
    data_mean = pt.zeros(3)
    data_std = pt.zeros(3)
    for imgs, _ in tqdm(loader, desc="mean"):
        data_mean += imgs.mean((0, 2, 3))
    data_mean /= len(loader)
    # 为了方便，再加上样本数很大，使用了总体标准差
    for imgs, _ in tqdm(loader, desc="var"):
        data_std += imgs.var((0, 2, 3), unbiased=False)
    data_std /= len(loader)
    return data_mean, pt.sqrt(data_std)
