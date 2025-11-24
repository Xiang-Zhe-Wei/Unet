from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import *
import os

trans_img = transforms.Compose([
    transforms.ToTensor()
])

class MyDataSet(Dataset):
    def __init__(self, path):
        self.path = path
        self.name  = os.listdir(os.path.join(path, "SegmentationClass"))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        seg_name = self.name[idx] # png -> jpg
        print(seg_name)
        seg_path = os.path.join(self.path, "SegmentationClass", seg_name)
        img_path = os.path.join(self.path, "JPEGImages", seg_name.replace("png", "jpg"))
        seg = mask_picture(seg_path)
        img = mask_picture(img_path)
        return trans_img(img), trans_img(seg)



if __name__ == "__main__":
    path = "/Users/xiangzhewei/Documents/master/Unet/VOC2012_train_val"
    m = MyDataSet(path)
    print(m[1][0].shape)

    # img = to_pil_image(m[1][0])
    img = to_pil_image(m[1][1])
    img.save("debug.png")
