from PIL import Image

def mask_picture(pathname, size=(256,256)):
    img = Image.open(pathname)
    maxsize = max(img.size)
    img_mask = Image.new('RGB', (maxsize,maxsize), (0,0,0))
    img_mask.paste(img, (0,0))
    img_mask = img_mask.resize(size)
    return img_mask