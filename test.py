import numpy
import PIL
import cv2
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torchvision.transforms as transforms
from model import *
import os

# dataset class for easy passing to the model
class StyleContentDataset(Dataset):
    def __init__(self, content_dir, style_dir):
        self.content_images = [os.path.join(content_dir, img) for img in os.listdir(content_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]
        
        self.style_images = []
        for subdir in os.listdir(style_dir):
            subdir_path = os.path.join(style_dir, subdir)
            if os.path.isdir(subdir_path):
                style_images_in_subdir = [os.path.join(subdir_path, img) for img in os.listdir(subdir_path) if img.endswith(('.jpg', '.png', '.jpeg'))]
                self.style_images.extend(style_images_in_subdir)
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()  # Converts to a tensor and scales to [0, 1]
        ])


    def __len__(self):
        return max(len(self.content_images), len(self.style_images))

    def __getitem__(self, idx):
        content_img_path = self.content_images[idx % len(self.content_images)]
        style_img_path = self.style_images[idx % len(self.style_images)]
        
        content_image = Image.open(content_img_path).convert("RGB")
        style_image = Image.open(style_img_path).convert("RGB")

        content_image = self.transform(content_image)
        style_image = self.transform(style_image)

        return content_image,style_image
    
def main():
    coco_path = "DataSets/unlabeled2017/"
    wikiart_path = "DataSets/wikiart/"

    test_set = StyleContentDataset(content_dir=coco_path,style_dir=wikiart_path)

    # retreive desired model
    styleTransferModel =  StyleTransferModel()
    styleTransferModel.load_state_dict(torch.load('SavedModels\cbam_AdaIN_SSIM=5_StyleScale=20(1).pth')["model_state_dict"])
    styleTransferModel.eval()

    loader = DataLoader(test_set,batch_size=1,shuffle=False)
    count =0 
    with torch.no_grad():
        for content, style in loader:
            
            generated_image = styleTransferModel(content,style)
            stylized_image = transforms.ToPILImage()(generated_image[0][0])
            content_im = transforms.ToPILImage()(style[0])
            style_im = transforms.ToPILImage()(content[0])
            # Display or save the image
            count += 1
            
            stylized_image.show()
            content_im.show()
            style_im.show()

            if count == 10: 
                break
    
    print("Done")




    return

main()