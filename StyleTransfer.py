# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToTensor
import torchvision.models as models
import os
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from PIL import Image
import time
import torchvision.transforms as transforms

# %% [markdown]
# # Sobel SSIM Loss Definition #

# %%
def sobel_filter(image):

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    channels = image.size(1)  # Number of channels
    sobel_x = sobel_x.repeat(channels, 1, 1, 1).to(image.device)
    sobel_y = sobel_y.repeat(channels, 1, 1, 1).to(image.device)

    edges_x = F.conv2d(image, sobel_x, padding=1, groups=channels)
    edges_y = F.conv2d(image, sobel_y, padding=1, groups=channels)

    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-6)  # Avoid NaNs
    edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-6)  # Normalize to [0, 1]
    return edges



def ssim_loss(pred, target, C1=0.01 ** 2, C2=0.03 ** 2):

    mu_pred = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    mu_target = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    sigma_pred = F.avg_pool2d(pred ** 2, kernel_size=3, stride=1, padding=1) - mu_pred ** 2
    sigma_target = F.avg_pool2d(target ** 2, kernel_size=3, stride=1, padding=1) - mu_target ** 2
    sigma_pred_target = F.avg_pool2d(pred * target, kernel_size=3, stride=1, padding=1) - mu_pred * mu_target

    ssim_map = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / (
        (mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2)
    )
    return 1 - ssim_map.mean() 



# %% [markdown]
# # Data Loader #

# %%
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
            transforms.RandomCrop((256,256)),
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
    
    




# %% [markdown]
# # Data Sets #

# %%
coco_path = "DataSets/unlabeled2017/"
wikiart_path = "DataSets/wikiart/"


Dataset = StyleContentDataset(coco_path,wikiart_path)
train_dataset ,test_dataset= random_split(Dataset,[0.8,0.2])

train_loader = DataLoader(train_dataset,batch_size=1,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)
print(len(Dataset))



# %% [markdown]
# # AdaIN #

# %%
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

# %%
class FixedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(FixedConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        
        # Initialize the kernel to an identity-like transformation
        if in_channels == out_channels and kernel_size == 1:
            nn.init.eye_(self.conv.weight.view(out_channels, in_channels))  # Identity initialization for 1x1
        else:
            nn.init.xavier_uniform_(self.conv.weight)  # Minimal modification if sizes differ

        # Freeze weights to prevent updates
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)

# %% [markdown]
# # CBAM #

# %%
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

# %%
def visualize_adain_output(fused_feat, epoch, save_dir="visualizations"):
    """
    Visualizes and saves the AdaIN feature map for a given epoch.
    """
    os.makedirs(save_dir, exist_ok=True)  # Create directory to save visualizations

    # Take the first batch's fused features for visualization
    # Reduce channels for visualization (e.g., take the mean across channels)
    feature_map = fused_feat[0].mean(dim=0).cpu().detach().numpy()

    # Normalize the feature map to [0, 1] for visualization
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())

    # Plot and save
    plt.figure(figsize=(6, 6))
    plt.imshow(feature_map, cmap="viridis")
    plt.colorbar()
    plt.title(f"AdaIN Output - Epoch {epoch}")
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f"adain_epoch_{epoch}.png"))
    plt.close()
def visualize_adain_input(feat, epoch, save_dir="visualizations2"):
    """
    Visualizes and saves the AdaIN feature map for a given epoch.
    """
    os.makedirs(save_dir, exist_ok=True)  # Create directory to save visualizations

    # Take the first batch's fused features for visualization
    # Reduce channels for visualization (e.g., take the mean across channels)
    feature_map = feat[0].mean(dim=0).cpu().detach().numpy()

    # Normalize the feature map to [0, 1] for visualization
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())

    # Plot and save
    plt.figure(figsize=(6, 6))
    plt.imshow(feature_map, cmap="viridis")
    plt.colorbar()
    plt.title(f"AdaIN Output - Epoch {epoch}")
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f"adain_epoch_{epoch}.png"))
    plt.close()

# %% [markdown]
# # Encoder Decoder #

# %%

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        vgg = models.vgg19(pretrained=True).features

        # Define layers to extract features at specific layers
        self.model = nn.Sequential(*[vgg[i] for i in range(len(vgg))])  # Up to ReLU5_1
        
        #freeze weights
        for param in self.model.parameters():
            param.requires_grad = False

       # 1: 1, 2: 6, 3: 11, 4:20, 5: 29
        self.relu1_1 = self.model[:2]
        self.relu2_1 = self.model[2:7]
        self.relu3_1 = self.model[7:12]
        self.relu4_1 = self.model[12:21]
        self.relu5_1 = self.model[21:28]

    def forward(self, x):
        

        '''
        previously we were going to use relu3-5, however upon adaptation to teh AdaIN paper we will use 1-4
        # Extract features at each specific layer
        feat3_1 = self.model[:12](x)  # Feature map after relu3_1
        feat4_1 = self.model[12:21](feat3_1)  # Feature map after relu4_1
        feat5_1 = self.model[21:28](feat4_1)  # Feature map after relu5_1

        return feat3_1, feat4_1, feat5_1
        '''
        feat1_1 = self.relu1_1(x)
        feat2_1 = self.relu2_1(feat1_1)
        feat3_1 = self.relu3_1(feat2_1)
        feat4_1 = self.relu4_1(feat3_1)

        return feat1_1,feat2_1,feat3_1,feat4_1

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # Decoder mirrors the encoder, progressively upsampling
        self.deconv_relu4_1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # Refine
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)  # Upsample: 32 -> 64
        )
        self.deconv_relu3_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # Refine
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)  # Upsample: 64 -> 128
        )
        self.deconv_relu2_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # Refine
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # Upsample: 128 -> 256
        )
        self.deconv_relu1_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # Refine
            nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # Final RGB output
            nn.Sigmoid()  # Scale pixel values to [0, 1]
        )

    def forward(self, feat4_1):
        """
        Reverse the feature maps progressively through the decoder.
        """
        x = self.deconv_relu4_1(feat4_1)  # 32 -> 64
        x = self.deconv_relu3_1(x)       # 64 -> 128
        x = self.deconv_relu2_1(x)       # 128 -> 256
        x = self.deconv_relu1_1(x)       # Refinement
        output = self.final(x)           # Final RGB image
        return output




    
# put the models together into a single model
class StyleTransferModel(nn.Module):
    def __init__(self):
        super(StyleTransferModel, self).__init__()

        '''
            We need a lot of models to process the content image,
            each layer output of the encoded content will be passed through
            its own attention network, along with their corresponding style layer
        '''
        self.encoder = Encoder().eval()
        self.cbam = CBAM(512)
        self.decoder = Decoder() 
        self.alpha = 1.0

        #convolutional operations
        #self.conv1x1_3 = nn.Conv2d(256, 256, kernel_size=1)
        #self.conv1x1_45 = nn.Conv2d(512, 512, kernel_size=1)
        

        #self.conv3x3 = nn.Conv2d(256,256,kernel_size=3)  



    def forward(self, content, style):

        # Pass content and style through encoder to get  feature maps
        content_feat1_1, content_feat2_1, content_feat3_1,content_feat4_1 = self.encoder(content)
        style_feat1_1, style_feat2_1, style_feat3_1, style_feat4_1 = self.encoder(style)


        content_feats = [content_feat1_1, content_feat2_1, content_feat3_1,content_feat4_1]
        style_feats = [style_feat1_1, style_feat2_1, style_feat3_1, style_feat4_1 ]
        #print("Encoded")




        # Apply CBAM, add the out put back into the original for the skip connection
        spactial_attention4_1 = self.cbam(content_feat4_1)

        attention_boosted_4_1 = content_feat4_1+spactial_attention4_1





        # Perform adaptive instancenormalization with AdaIN to fuse style into content
        fused_feat4_1 = AdaIn(attention_boosted_4_1,style_feat4_1)

        # Just scales the degree of which the style is applied, if one it remains the same
        fused_feat4_1 = self.alpha * fused_feat4_1 + (1 - self.alpha) * content_feat4_1


    

        #print("Generating")
        generated_image = self.decoder(fused_feat4_1)
 
        
        return generated_image,style_feats,content_feats

    
# Custom Loss Class

class StyleTransferLoss(nn.Module):
    def __init__(self, vgg_encoder, lambda_c=1, lambda_s=10, lambda_ssim=1):

        super(StyleTransferLoss, self).__init__()
        self.vgg_encoder = vgg_encoder  # Pre-trained VGG encoder
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.lambda_ssim = lambda_ssim
        self.mse_loss = nn.MSELoss()    # Mean Squared Error loss

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

    def forward(self, generated_image, content,content_feats, style_feats):
        """
        Args:
            generated_image (torch.Tensor): The stylized image.
            content_feats (list of torch.Tensor): Features from the content image.
            style_feats (list of torch.Tensor): Features from the style image.

        Returns:
            total_loss: Weighted combination of content, style, and SSIM losses.
        """
        # Extract features from the generated image using VGG encoder
        gen_feats = self.vgg_encoder(generated_image)

        
        # Calculate content loss (last feature map for structure preservation)
        content_loss = self.calc_content_loss(gen_feats[-1], content_feats[-1])
        
        # Calculate style loss (mean and variance alignment for each layer)
        style_loss = 0
        for gen_feat, style_feat in zip(gen_feats, style_feats):
            style_loss += self.calc_style_loss(gen_feat, style_feat)

        # Sobel-SSIM edge loss
        sobel_gen = sobel_filter(generated_image)
        sobel_content = sobel_filter(content)
        edge_loss = ssim_loss(sobel_gen, sobel_content)
        
        # Combine the losses with weights
        total_loss = (self.lambda_c * content_loss +
                      self.lambda_s * style_loss +
                      self.lambda_ssim * edge_loss)

        return total_loss,self.lambda_ssim*edge_loss,self.lambda_s*style_loss

# %% [markdown]
# # Train Model #

# %%
def train_models(model,criterion,dataloader,optimizer,scheduler,device):

    epochs = 40

    model.train()
    count = 0

    for epoch in range(epochs):
        count = 0
        start_time = time.time()
        for content, style in dataloader:

            content = content.to(device)
            style = style.to(device)

            optimizer.zero_grad()
            
            generated_image,style_feat,content_feat = model(content,style)

            # Calculate loss
            loss,edge_l,style_l = criterion(generated_image,content,content_feat,style_feat)
          
            # Backpropagate and update weights
            loss.backward()
            optimizer.step()
            count +=1 
            print(count,end='\r')

            #only look at the first X number of training examples, it should be random, as the training loader shuffles the data on initilization
            if count == 7000:
                print(f'Edge{edge_l} Style{style_l}')
                break
            '''
            if count == 1:
                visualize_adain_output(fused_5_1,epoch+1)
                visualize_adain_input(content_feat[-1],epoch+1)
            '''
        scheduler.step()



        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        print(f"Time: {(time.time()-start_time)/60} Minutes")


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = StyleTransferModel()

for param in model.encoder.model.parameters():
    assert param.requires_grad == False

criterion = StyleTransferLoss(model.encoder)
model.to(device)
criterion.to(device)

# The decoder is what applies the arbitrary style and reconstructs the image
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

train_models(model,criterion,train_loader,optimizer,scheduler,device)

# %%
model.eval()
print(type(test_dataset.__getitem__(0)[0]))

'''
model = StyleTransferModel()
model.load_state_dict(torch.load("cbam_AdaIN_Vanillia1.pth")["model_state_dict"])
model.to(device)
model.eval()
'''
# Extract content and style features
with torch.no_grad():  # Disable gradient calculation for faster inference
    i = 0
    for content,style in test_loader:
        stylized_image = model(content.to(device),style.to(device))
        i +=1
        if (i <= 100):
            continue
        # Convert to PIL image for saving or displaying
        stylized_image = transforms.ToPILImage()(stylized_image[0][0])
        content_im = transforms.ToPILImage()(style[0])
        style_im = transforms.ToPILImage()(content[0])
        # Display or save the image
        
        stylized_image.show()
        content_im.show()
        style_im.show()
        
        break





# %%

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'cbam_AdaIN_SSIM=1_StyleScale=10(3).pth')




