import os, torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

class DrawDataset(torch.utils.data.Dataset):
    def __init__(self, training=True, transform=None, draw_type = "wave"):
        # self.draw_type = draw_type
        if training==True:
            self.path = f"drawings/{draw_type}/training"
        else:
            self.path = f"drawings/{draw_type}/testing"

        folders = list(os.walk(self.path))
        df_list = []
        for i, folder in enumerate(folders[0][1]):
            data = pd.DataFrame([folders[i+1][2]], index=['path']).T
            data['type'] = folder
            data['has_parkinson'] = i
            df_list.append(data)

        self.df = pd.concat(df_list, ignore_index=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = Image.open(f'{self.path}/{self.df["type"][idx]}/{self.df["path"][idx]}').convert("L")
        # x = Image.fromarray(self.x_data[idx].reshape(28, 28))
        x = x.resize((28,28))
        y = torch.tensor(self.df["has_parkinson"][idx])
        if self.transform:
            x = self.transform(x)
        x = np.array(x, dtype=np.float32)/255
        x = np.reshape(x,(1,28, 28))
        # x = (x - 0.5) * 2.0
        # x = transforms.ToTensor()(x
        x = torch.tensor(x, dtype=torch.float64)
        return x, y

