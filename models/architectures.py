import sys
sys.path.append("../FDFAPBG")

import torch.nn as nn

class Recover(nn.Module):
    def __init__(self):
        super(Recover, self).__init__()
        self.main = nn.Sequential(
                                  nn.Conv2d(
                                                     1, 256, kernel_size=4, stride=(1,3), padding=(2,18), bias=False
                                                     ),
                                  nn.BatchNorm2d(256),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(
                                                     256, 128, kernel_size=4, stride=(1,4), padding=(1,0), bias=False
                                                     ),
                                  nn.BatchNorm2d(128),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(
                                                     128, 64, kernel_size=4, stride=(1,4), padding=(2,0), bias=False
                                                     ),
                                  nn.BatchNorm2d(64),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(64, 1, kernel_size=2, stride=(1,2), padding=(0,0), bias=False),
                                  #nn.BatchNorm2d(64),
                                  #nn.ConvTranspose2d(64, 1, kernel_size=4, stride=(1,), padding=2, bias=False),
                                  )
    
    def forward(self, input):
        return self.main(input)

class Refiner(nn.Module):
    def __init__(self):
        super(Refiner, self).__init__()
        self.main = nn.Sequential(
                                  nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Dropout(0.1),
                                  nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(128),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Dropout(0.1),
                                  nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Dropout(0.1),
                                  nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(512),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Dropout(0.1),
                                  nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
                                  nn.Sigmoid(),
                                  )
    
    def forward(self, input):
        return self.main(input)
