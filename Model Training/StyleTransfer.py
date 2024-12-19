# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torchvision.models as models
import os
from torch.utils.data import random_split
from PIL import Image
import time
import torchvision.transforms as transforms
from pytorch_msssim import ssim

# %% [markdown]
# # Sobel SSIM Loss Definition #

# %%
# Calculates the Gradient magnitude of the passed image and returns it, does so by performing a directional sobel convolution in the x and y direction and then using those as inputs for the gradient magnitude function
def sobel_filter(image):

    # setup the kernals for direction sobel
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # resize to cover all channels of the passed image
    channels = image.size(1)
    sobel_x = sobel_x.repeat(channels, 1, 1, 1).to(image.device)
    sobel_y = sobel_y.repeat(channels, 1, 1, 1).to(image.device)

    # apply sobel in x and y direction
    edges_x = F.conv2d(image, sobel_x, padding=1, groups=channels)
    edges_y = F.conv2d(image, sobel_y, padding=1, groups=channels)

    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-6)
    
    # normalize the gradient magnitude and return
    edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-6) 
    return edges



# %% [markdown]
# # Data Loader #

# %%
# Dataset class to gice to the dataloader to load our style and content images into the model
class StyleContentDataset(Dataset):
    def __init__(self, content_dir, style_dir):
        # Get the path to all image files in the content folder
        self.content_images = [os.path.join(content_dir, img) for img in os.listdir(content_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]
        

        # To my utter shagrin, the style images are in subfolders, so we have to go through all of them to get each one's path
        self.style_images = []
        for subdir in os.listdir(style_dir):
            subdir_path = os.path.join(style_dir, subdir)
            if os.path.isdir(subdir_path):
                style_images_in_subdir = [os.path.join(subdir_path, img) for img in os.listdir(subdir_path) if img.endswith(('.jpg', '.png', '.jpeg'))]
                self.style_images.extend(style_images_in_subdir)
        
        # transforms to perform on the image when being loaded into the model
        self.transform = transforms.Compose([

            transforms.Resize((512, 512)),
            # random crop prevents the model from overfitting
            transforms.RandomCrop((256,256)),
            transforms.ToTensor()
        ])

    # I'm not completely sure what the best practice is, because technically its two datasets of different sizes in one loader, but this shouldn't matter for now because we're using a very small subset 
    def __len__(self):
        return max(len(self.content_images), len(self.style_images))

    # Return the ith image pair
    def __getitem__(self, idx):
        # Get the corresponding image path
        content_img_path = self.content_images[idx % len(self.content_images)]
        style_img_path = self.style_images[idx % len(self.style_images)]
        
        # Retreive the image as a PIL
        content_image = Image.open(content_img_path).convert("RGB")
        style_image = Image.open(style_img_path).convert("RGB")

        # Transform the image to be returned
        content_image = self.transform(content_image)
        style_image = self.transform(style_image)

        return content_image,style_image
    
    




# %% [markdown]
# # Data Sets #

# %%
# Dataset paths, relative to this folder
coco_path = "DataSets/unlabeled2017/"
wikiart_path = "DataSets/wikiart/"

# Create the custom dataset
Dataset = StyleContentDataset(coco_path,wikiart_path)

# Split is pretty arbitrary, as we wont use nearly all of the images in either the test or train split
train_dataset ,test_dataset= random_split(Dataset,[0.8,0.2])

# Create the train and test dataloaders
train_loader = DataLoader(train_dataset,batch_size=1,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)

# %% [markdown]
# # AdaIN #

# %%
'''
The code for AdaIN was borrowed from this individual, who re-wrote the original code in python (was Lua)
https://github.com/naoto0804/pytorch-AdaIN/blob/master/function.py

Original: https://github.com/xunhuang1995/AdaIN-style/blob/master/lib/AdaptiveInstanceNormalization.lua

When I say "Original," I'm reffering to the code that the authors of the paper that this is based of wrote for their paper
'''


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat



def AdaIn(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


# %% [markdown]
# # Model Definitons #

# %% [markdown]
# # CBAM #

# %%
'''
    This code for CBAM was borrowed from this github:https://github.com/Jongchan/attention-module/blob/c06383c514ab0032d044cc6fcd8c8207ea222ea7/MODELS/cbam.py#L84

    It's the official implementation from the researchers who first proposed it
'''


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

# %% [markdown]
# # Encoder Decoder #

# %%

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # Get the pretrained vgg19 model
        vgg = models.vgg19(pretrained=True).features

        # Define layers to extract features at specific layers
        self.model = nn.Sequential(*[vgg[i] for i in range(len(vgg))])  
        
        # Freeze weights so we don't change anything we didn't mean to 
        for param in self.model.parameters():
            param.requires_grad = False

        # These need to be calculated seperatly, as for some loss calculation tasks we care about the higher level feature maps
        self.relu1_1 = self.model[:2]
        self.relu2_1 = self.model[2:7]
        self.relu3_1 = self.model[7:12]
        self.relu4_1 = self.model[12:21]
        self.relu5_1 = self.model[21:28]

    # Generate feature maps, X is the input image, returns relu1_1 - relu4_1 feature maps
    def forward(self, x):

        feat1_1 = self.relu1_1(x)
        feat2_1 = self.relu2_1(feat1_1)
        feat3_1 = self.relu3_1(feat2_1)
        feat4_1 = self.relu4_1(feat3_1)

        return feat1_1,feat2_1,feat3_1,feat4_1

# Custom decoder model, takes the processed feature map and reconstructs it back to the original image space as it goes along
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # Decoder mirrors the encoder, upsampling as it passes through each layer
        self.deconv_relu4_1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)  
        )
        self.deconv_relu3_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1) 
        )
        self.deconv_relu2_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  
        )
        self.deconv_relu1_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  
            nn.Sigmoid()  
        )

    def forward(self, feat4_1):
    
        #Reverse the feature maps through the decoder, output is the finalized image
    
        x = self.deconv_relu4_1(feat4_1)  
        x = self.deconv_relu3_1(x)        
        x = self.deconv_relu2_1(x)       
        x = self.deconv_relu1_1(x)        
        output = self.final(x)          
        return output




    
# Put the models together into a single model
class StyleTransferModel(nn.Module):
    def __init__(self):
        super(StyleTransferModel, self).__init__()

        self.encoder = Encoder().eval()
        self.cbam = CBAM(512)
        self.decoder = Decoder() 

        #alpha can be altered to adjust style application strength post training (0-1)
        self.alpha = 1.0


    # Process the content and style images
    def forward(self, content, style):

        # Pass content and style through encoder to get  feature maps
        content_feat1_1, content_feat2_1, content_feat3_1,content_feat4_1 = self.encoder(content)
        style_feat1_1, style_feat2_1, style_feat3_1, style_feat4_1 = self.encoder(style)

        # Place these in lists so we can access them easily later
        content_feats = [content_feat1_1, content_feat2_1, content_feat3_1,content_feat4_1]
        style_feats = [style_feat1_1, style_feat2_1, style_feat3_1, style_feat4_1]

        # Apply CBAM, add the out put back into the original for the skip connection
        attention4_1 = self.cbam(content_feat4_1)

        # Skip connection
        attention_boosted_4_1 = content_feat4_1+attention4_1

        # Perform adaptive instance normalization with to fuse style into content
        fused_feat4_1 = AdaIn(attention_boosted_4_1,style_feat4_1)

        # Just scales the degree of which the style is applied, if one it remains the same
        fused_feat4_1 = self.alpha * fused_feat4_1 + (1 - self.alpha) * content_feat4_1
    

        # decode the image
        generated_image = self.decoder(fused_feat4_1)
        
        return generated_image,style_feats,content_feats

    
# Custom Loss Class, will be used for both model cases, but the experiment with no ssim will set ssim lambda to 0

'''
The code for the style and content loss was borrowed from this individual, who re-wrote the original code in python (was Lua)
https://github.com/naoto0804/pytorch-AdaIN/blob/master/net.py

Original: https://github.com/xunhuang1995/AdaIN-style/blob/master/lib/ContentLossModule.lua

Specifically, the code for calc_content_loss and calc_style_loss were used
'''

class StyleTransferLoss(nn.Module):
    def __init__(self, vgg_encoder, lambda_c=1, lambda_s=4, lambda_ssim=7):

        super(StyleTransferLoss, self).__init__()
        self.vgg_encoder = vgg_encoder 
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.lambda_ssim = lambda_ssim
        self.mse_loss = nn.MSELoss()   

    def calc_content_loss(self, input, target):
        # MSE loss for content preservation
        assert input.size() == target.size(), "Content size mismatch!"
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        # Style loss based on mean and variance
        assert input.size() == target.size(), "Style size mismatch!"
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)

    # Actual loss calcualtion, takes the generated image, as well as the content image, and the feature maps for content and style
    def forward(self, generated_image, content,content_feats, style_feats):

        # encode the generated image for loss calculation
        gen_feats = self.vgg_encoder(generated_image)

        
        # calculate content loss (last feature map for structure preservation)
        content_loss = self.calc_content_loss(gen_feats[-1], content_feats[-1])
        
        # calculate style loss (mean and variance alignment for each layer)
        style_loss = 0
        for gen_feat, style_feat in zip(gen_feats, style_feats):
            style_loss += self.calc_style_loss(gen_feat, style_feat)

        # Sobel-SSIM edge loss
        sobel_gen = sobel_filter(generated_image)
        sobel_content = sobel_filter(content)

        edge_loss = 1 - ssim(sobel_gen, sobel_content, data_range=1.0, size_average=True)


        # Combine the losses with weights
        total_loss = (self.lambda_c * content_loss +
                      self.lambda_s * style_loss +
                      self.lambda_ssim * edge_loss)

        return total_loss,self.lambda_ssim*edge_loss,self.lambda_s*style_loss

# %% [markdown]
# # Train Model #

# %%

# trains the model for a set number of epcochs
def train_models(model,criterion,dataloader,optimizer,scheduler,device):

    epochs = 100

    model.train()
    count = 0


    # Run for X number of epochs
    for epoch in range(epochs):
        count = 0
        start_time = time.time()
        for content, style in dataloader:

            # Send images to whatever device you passed
            content = content.to(device)
            style = style.to(device)

            optimizer.zero_grad()
            
            # Get the generated image and the feature maps from the model
            generated_image,style_feat,content_feat = model(content,style)

            # Calculate loss
            loss,_,_ = criterion(generated_image,content,content_feat,style_feat)
          
            # Backpropagate and update weights
            loss.backward()
            optimizer.step()
            count +=1 

            # Used for visualizing where in the epoch the model is at
            print(count,end='\r')

            # Only look at the first X number of training examples, it should be random, as the training loader shuffles the data on initilization
            if count == 7000:
                break

        scheduler.step()


        # For my sanity, prints out the epoch and how long it took to run
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        print(f"Time: {(time.time()-start_time)/60} Minutes")


# %%
# get gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = StyleTransferModel()

#make sure we dont update the encoder
for param in model.encoder.model.parameters():
    assert param.requires_grad == False

criterion = StyleTransferLoss(model.encoder)
model.to(device)
criterion.to(device)

# Training options
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# send model to be trained
train_models(model,criterion,train_loader,optimizer,scheduler,device)

# %%
model.eval()


# Visual Evaluation of the model
with torch.no_grad(): 
    i = 0
    for content,style in test_loader:
        stylized_image = model(content.to(device),style.to(device))
        i +=1

        # Quick and dirty way to look at the ith inferenced image, not great but its nice for quickly evaluating model performance post training
        if (i <= 3):
            continue
        # display as a PIL
        stylized_image = transforms.ToPILImage()(stylized_image[0][0])
        content_im = transforms.ToPILImage()(style[0])
        style_im = transforms.ToPILImage()(content[0])

        
        stylized_image.show()
        content_im.show()
        style_im.show()
        
        break





# %%
#save the trained model 
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'newSSIMTEST.pth')




