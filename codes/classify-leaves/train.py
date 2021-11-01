from utils import *
from torchvision import transforms
from torch import optim
import torch.nn.functional as F
from model_def import *
import re
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train(train_batch_size = 96, test_batch_size = 96, lr = 1, epoch_num = 3, device='cuda'):
    lblenc, train_data, cv_data, test_data = load_leaves()
    train_loader = DataLoader(train_data, train_batch_size, shuffle=True)
    cv_loader = DataLoader(train_data, test_batch_size, shuffle=False)
    test_loader = DataLoader(test_data, test_batch_size, shuffle=False)

    writer = SummaryWriter()

    # Start Training
    model = Net().to(device)
    # 提取预训练参数和fc参数
    pre_params = [param for name, param in model.named_parameters() if not re.match(r"^fc\..*", name)]
    fc_params = model.fc.parameters()
    # 构造分组的优化器
    updater = optim.Adam([
        {'params':pre_params, lr:lr/10},
        {'params':fc_params}
    ], lr = lr, weight_decay=0.001)

    # updater = optim.Adam(filter(lambda x:x.requires_grad, model.parameters()), lr = lr, weight_decay=0.0001)
    # loss函数
    criterion = nn.CrossEntropyLoss()

    for epoch_i in range(epoch_num):
        print(f"Epoch {epoch_i} -----------------------------")
        tot_loss = 0
        correct_num = 0
        for batch_i, (imgs, lbls) in enumerate(tqdm(train_loader)):
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            pred = model(imgs)
            loss = criterion(pred, lbls.long())
            updater.zero_grad()
            loss.backward()
            updater.step()

            tot_loss += loss
            correct_num += (pred.argmax(dim=1)==lbls).sum()
            if loop_passed(5):
                avg_loss = tot_loss/(batch_i+1)
                accuracy = correct_num / (train_loader.batch_size*(batch_i+1))
                print(f'batch {batch_i} with loss {avg_loss}, accuracy {accuracy}')
                writer.add_scalar(f"BatchLoss/{epoch_i}", avg_loss)

        print(f'Epoch {epoch_i} has loss {tot_loss/(batch_i+1)}')
        writer.add_scalar("Loss/train", tot_loss/(batch_i+1))
        # 评价epoch
        correct_num = 0
        with pt.no_grad():
            for batch_i, (imgs, lbls) in enumerate(tqdm(cv_loader)):
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                pred = model(imgs)
                pred = pred.argmax(dim=1)
                correct_num += (pred == lbls).sum()
            tot_num = (batch_i+1)*cv_loader.batch_size
            accuracy = correct_num / tot_num
        print(f'Epoch {epoch_i} has accuracy {accuracy}')
        writer.add_scalar("Accuracy/train", accuracy)
        pt.save(model.state_dict(), models_path / "classify-leaves.pth")
