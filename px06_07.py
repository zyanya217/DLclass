# px06_07.py
import numpy as np
import glob
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# 把圖片轉換成Tensor
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
    ])

# 實際讀取*.jpg圖片
apple_images = []
for imageName in glob.glob('c:/oopc/fruits/train/Apple/*.jpg'):
    image = Image.open(imageName).convert('RGB')
    image = transform(image)
    apple_images.append(image)

orange_images = []
for imageName in glob.glob('c:/oopc/fruits/train/Orange/*.jpg'):
    image = Image.open(imageName).convert('RGB')
    image = transform(image)
    orange_images.append(image)
    
apple_label= torch.ones(1)
orange_label = torch.zeros(1)

#--------------------------
class Discriminator(nn.Module):
    def __init__(self):
       super(Discriminator, self).__init__()
       self.layer = nn.Linear(10000, 1)

    def forward(self, x):
       y = self.layer(x)
       z = torch.sigmoid(y)
       return z

D = Discriminator()
# 損失函數
criterion = nn.BCELoss()
optimizer_d = torch.optim.Adam(D.parameters(), lr=0.005)

#-------------------------------
def train_D(X, T):
    Z = D(X)
    Z = Z.squeeze(0)
    Error = criterion(Z, T)

    optimizer_d.zero_grad()      
    Error.backward()
    optimizer_d.step()
    return Error

#-----------------------------------
# 共訓練500回合
print('\n-------- 訓練500回合 ----------')
for ep in range(500):
    # 1回合有5個iterations
    for i in range(5):
        #------- Apple -------------------------
        X = apple_images[i]
        X = torch.flatten(X, 1)/255
        A_Error = train_D(X, apple_label)

        #------- Orange -------------------------
        X = orange_images[i]
        X = torch.flatten(X, 1)/255
        O_Error = train_D(X, orange_label)
        
    if( ep%100 == 0):
        A_err = A_Error.detach().numpy()
        O_err = O_Error.detach().numpy()
        print('epoch=', ep, '  A_error:', np.round(A_err, 2),
                         '  O_error:', np.round(O_err, 2))
 
#------ Prediction ------------------------------------
print('\n-------- 預測 ----------')
for imageName in glob.glob('c:/oopc/fruits/test/*.jpg'):
    image = Image.open(imageName).convert('RGB')
    tx = transform(image)
    tx = torch.flatten(tx, 1)/255
    Z = D(tx)
    Z = Z.detach().numpy()
    print(imageName, ', 預測 Z=', np.round(Z, 3)[0])

#END  


       

