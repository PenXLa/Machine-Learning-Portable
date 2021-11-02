import pandas as pd
from torch import nn
from utils import *
from lib.utils import *

def evaluate(model:nn.Module, device='cuda', eval_num:int=-1):
    submission = pd.DataFrame(columns=['image', 'label'])
    lblenc, _, _, test_data, filenames = load_leaves()
    test_loader = DataLoader(test_data, 96, shuffle=False, num_workers=num_workers)
    with pt.no_grad():
        model.eval()
        sample_id = 0
        for i, imgs in enumerate(tqdm(test_loader, desc="evaluate")):
            if eval_num != -1 and i >= eval_num: break
            imgs = imgs.to(device)
            pred = model(imgs).argmax(dim=1).to('cpu')
            for ans in pred:
                submission = submission.append({'image':filenames.iat[sample_id], 'label':lblenc.inverse_transform(ans).item()}, ignore_index=True)
                sample_id += 1
    submission.to_csv(mksure(data_drive / "classify-leaves/submission.csv"), index=None)