from utils import *
from lib.utils import *

def test(model, device='cuda'):
    lblenc, train_data, cv_data, test_data = load_leaves()
    test_loader = DataLoader(test_data, 1, shuffle=False, num_workers=num_workers)
    for imgs, lbls in tqdm(test_loader, desc="test"):
        imgs = imgs.to(device)
        lbls = lbls.to(device)
