import pandas as pd

from utils import *
from lib.utils import *

def test(model, device='cuda'):
    submission = pd.DataFrame(columns=['image', 'label'])
    lblenc, _, _, test_data, filenames = load_leaves()
    test_loader = DataLoader(test_data, 1, shuffle=False, num_workers=num_workers)
    with pt.no_grad():
        for i, imgs in enumerate(tqdm(test_loader, desc="test")):
            imgs = imgs.to(device)
            pred = model(imgs).argmax(dim=1).to('cpu')
            submission = submission.append({'image':filenames.iat[i], 'label':lblenc.inverse_transform(pred).item()}, ignore_index=True)
    submission.to_csv(mksure(data_drive / "classify-leaves/submission.csv"), index=None)