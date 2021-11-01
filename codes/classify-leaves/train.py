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
    #test_loader = DataLoader(test_data, test_batch_size, shuffle=False)

    writer = SummaryWriter()

    # Start Training
    model = Net().to(device)
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
        pbar = tqdm(train_loader, desc="train") #训练进度条
        for batch_i, (imgs, lbls) in enumerate(pbar):
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            pred = model(imgs)
            loss = criterion(pred, lbls.long())
            updater.zero_grad()
            loss.backward()
            updater.step()

            tot_loss += loss
            correct_num += (pred.argmax(dim=1)==lbls).sum()
            if loop_passed(3):
                avg_loss = tot_loss/(batch_i+1)
                accuracy = correct_num / (train_loader.batch_size*(batch_i+1))
                pbar.set_postfix({'loss': avg_loss, "accuracy":accuracy}) #更新进度条
                writer.add_scalar(f"Loss/Train", avg_loss, epoch_i*len(train_loader)+batch_i)
                writer.add_scalar(f"Accuracy/Train", accuracy, epoch_i*len(train_loader)+batch_i)

        print(f'Epoch {epoch_i} has loss {tot_loss/(batch_i+1)}')
        # 评价epoch
        correct_num = 0
        with pt.no_grad():
            for batch_i, (imgs, lbls) in enumerate(tqdm(cv_loader, desc="cv")):
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                pred = model(imgs)
                pred = pred.argmax(dim=1)
                correct_num += (pred == lbls).sum()
            tot_num = (batch_i+1)*cv_loader.batch_size
            accuracy = correct_num / tot_num
        print(f'Epoch {epoch_i} has accuracy {accuracy}')
        writer.add_scalar("Accuracy/train", accuracy, epoch_i)
        pt.save(model.state_dict(), models_drive / "classify-leaves.pth")
