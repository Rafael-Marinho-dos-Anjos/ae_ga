
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from dataset.ds import train_ds, test_val_ds


train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)

test_ds, val_ds = random_split(test_val_ds, (0.5, 0.5))
val_dl = DataLoader(val_ds, batch_size=16)
test_dl = DataLoader(test_ds)
