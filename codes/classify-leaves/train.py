from utils import *
from torchvision import transforms
from torch import optim
import torch.nn.functional as F
from model_def import *
import re
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler

def train(train_batch_size = 96,
          test_batch_size = 96,
          lr = 1,
          gamma = 0.9,
          weight_decay=0.001,
          epoch_num = 20,
          device='cuda'):
    lblenc, train_data, cv_data, _, _ = load_leaves()
    train_loader = DataLoader(train_data, train_batch_size, shuffle=True, num_workers=num_workers)
    cv_loader = DataLoader(cv_data, test_batch_size, shuffle=False, num_workers=num_workers)

    writer = SummaryWriter()

    # Start Training
    model = RareNet().to(device)
    # 提取预训练参数和fc参数
    pre_params = [param for name, param in model.named_parameters() if not re.match(r"^fc\..*", name)]
    fc_params = model.fc.parameters()
    # 构造分组的优化器
    updater = optim.SGD([
        {'params':pre_params, lr:lr/10},
        {'params':fc_params}
    ], lr = lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ExponentialLR(updater, gamma)

    # loss函数
    criterion = nn.CrossEntropyLoss()

    for epoch_i in range(epoch_num):
        print(f"Epoch {epoch_i} -----------------------------")
        tot_loss = 0
        correct_num = 0
        sample_num = 0
        model.train()
        pbar = tqdm(train_loader, desc="train") #训练进度条
        for batch_i, (imgs, lbls) in enumerate(pbar):
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            pred = model(imgs)
            loss = criterion(pred, lbls.long())
            updater.zero_grad()
            loss.backward()
            updater.step()

            tot_loss += loss.item()*len(lbls)
            sample_num += len(lbls)
            correct_num += (pred.argmax(dim=1)==lbls).sum().item()
            if loop_passed(3):
                avg_loss = tot_loss / sample_num
                accuracy = correct_num / sample_num
                pbar.set_postfix({'loss': avg_loss, "accuracy":accuracy}) #更新进度条
                writer.add_scalar(f"Loss/Train", avg_loss, epoch_i*len(train_loader)+batch_i)
                writer.add_scalar(f"Accuracy/Train", accuracy, epoch_i*len(train_loader)+batch_i)

        # 评价epoch
        accuracy = test(model, cv_loader, device)
        print(f'Epoch {epoch_i} has accuracy {accuracy}')
        writer.add_scalar("Accuracy/train", accuracy, epoch_i)
        pt.save(model.state_dict(), models_drive / "classify-leaves-resnet50.pth")


# 测试正确率
def test(model:nn.Module, data_loader:DataLoader, device='cuda', ):
    correct_num = 0
    with pt.no_grad():
        model.eval()
        for batch_i, (imgs, lbls) in enumerate(tqdm(data_loader, desc="cv")):
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            pred = model(imgs)
            pred = pred.argmax(dim=1)
            correct_num += (pred == lbls).sum().item()
        return correct_num / len(data_loader.dataset)