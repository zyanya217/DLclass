# lab_UNet_008.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn, optim
from PIL import Image
import glob
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode

#--------------------------------------------------------------
def init_weights(net, init='norm', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
            
    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_model(model):
    model = init_weights(model)
    return model
#--------------------------------------
class UnetBlock(nn.Module):
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        if input_c is None: input_c = nf
        downconv = nn.Conv2d(input_c, ni, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)
        
        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout: up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class Unet(nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        for _ in range(n_down - 5):
            unet_block = UnetBlock(num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True)
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetBlock(output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True)
     
    def forward(self, x):
        x = self.model(x)
        out = torch.sigmoid(x)
        return out
     
#-------------------------------------------------
SIZE = 256
class ColorizationDataset(Dataset):
    def __init__(self):
        self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE), InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(), # A little data augmentation!
            ])
        self.split = 'train'
        self.size = SIZE
        self.path = 'c:/oopc/picasso/train/'
        
        self.filenames =  glob.glob(self.path + '*.*')
        print(len(self.filenames))
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img = Image.open(filename).convert("RGB")
        img = self.transforms(img)

        # RGB -> LAB L[0, 100] a[-127, 128] b[-128, 127]
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")
        
        #L --> [0, 255]
        L = img_lab[:, :, 0] * 2.55
        L = Image.fromarray(np.uint8(L))
        L = transforms.ToTensor()(np.array(L).astype('float32')[..., np.newaxis] / 255.0)
        # L --> [0, 1]
        # L.shape 是(1, 256, 256)

        #ab --> [0, 255]
        ab = np.clip(img_lab[:, :, 1:3] + 128, 0, 255)
        ab = transforms.ToTensor()(ab.astype('float32')) / 255.
        #ab --> [0, 1]
        #ab.shape 是(2, 256, 256)
        
        return L, ab
    
    def __len__(self):
        return len(self.filenames)

dataset = ColorizationDataset()

#-----------------------------------------------
real_label = 1.0
fake_label = 0.0
real_label = torch.tensor(real_label)
fake_label = torch.tensor(fake_label)

#-------------------------------------------------------------------------------------------
G = Unet(input_c=1, output_c=2, n_down=8, num_filters=64)
Net_G = init_model(G)

opt_G = optim.Adam(G.parameters(), lr=0.0004, betas=(0.5, 0.999))

#loss_func = nn.MSELoss()
L1criterion = nn.L1Loss()

#-------------------------------------------------------------------------------------------
new_ab = None
L = None

def train(data):
        global fake_image, Lx, L, new_ab, abx
        Lx, abx = data
        L = Lx.unsqueeze(0)
        ab = abx.unsqueeze(0)
 
        # ----  訓練G  -----------------------------
        Net_G.train()
        opt_G.zero_grad()

        # 由G生成new_x
        new_ab = Net_G(L)

        # cond + new_x
        new_image = torch.cat([L, new_ab], dim=1)
        loss_G_L1 = L1criterion(new_ab, ab) * 100
        
        loss_G = loss_G_L1
        loss_G.backward(retain_graph=True)
        opt_G.step()

        print(loss_G)
        #print(new_x.shape)
        
 
#------------------------------------------
epochs = 1
for e in range(epochs):
        print(e)
        for data in dataset:
              train(data)

#---------------------------------------------
#在Tensor層級匯合通道
lab_t = torch.cat([L, new_ab], dim=1)

#刪除Tensor的batch維度
lab_t = lab_t.squeeze(0)

#轉回np層級
lab_np = lab_t.detach().numpy()

#轉回(256, 256, 3)
lab = np.transpose(lab_np, (1, 2, 0))

#轉回Lab標準值
lab[:, :, 0:1] = lab[:, :, 0:1] * 100
lab[:, :, 1:3] = np.clip(lab[:, :, 1:3] * 255 - 128, -100, 100)

#從Lab轉回RGB
img = lab2rgb(lab.astype(np.float64))

#--------------------------------
FILE = 'c:/ox/lab_Unet_100.pt'
torch.save(Net_G, FILE)
print('model saved')
#--------------------------------

#繪出圖
plt.imshow(img)
plt.show()
#-----------------
# END
