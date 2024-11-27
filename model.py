import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



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
