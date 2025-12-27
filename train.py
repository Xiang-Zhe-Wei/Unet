import torch
from Unet import *
from data import * 
from torch.utils.data import DataLoader
from torch import optim
from torchvision.utils import save_image

device = "cuda" if torch.cuda.is_available() else "cpu"
data_path = "/Users/xiangzhewei/Documents/master/Unet/VOC2012_train_val"
model_path = "params/unet.pth"
save_path = "train_img"

if __name__ == "__main__":
    dataloader = DataLoader(MyDataSet(data_path), batch_size=2, shuffle=True)
    net = Unet().to(device)
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
        print("load sucessfully")
    else:
        print("load faild")

    opt = optim.Adam(net.parameters())
    loss_function = nn.BCELoss()
    
    epoch = 1
    while True:
        for i, (img, seg_img) in enumerate(dataloader):
            img, seg_img = img.to(device), seg_img.to(device)
            out_img = net(img)
            train_loss = loss_function(seg_img, out_img)
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i%5==0:
                print(f'{epoch}--{i}--training_loss ==> {train_loss}')
            if i%50==0:
                torch.save(net.state_dict(), model_path)

            # see the image
            _img     = img[0]
            _seg_img = seg_img[0]
            _out_img = out_img[0]
            combine_img = torch.stack([_img, _seg_img, _out_img], dim=0)
            save_image(combine_img, f"{save_path}/{i}.png")
        epoch+=1