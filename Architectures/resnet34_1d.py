'''

implémentation de l'achitecture restnet pour classifier des signaux 1d

source: https://arxiv.org/pdf/1512.03385.pdf

'''

import torch.nn as nn



class BasicBlock(nn.Module):
    
    '''
    
    Residual block
    
    '''
    
    
    
    
    def __init__(self, n_features, kernel_size = 3):
        
        super(BasicBlock, self).__init__()
        
        
        self.n_features = n_features
        self.kernel_size = kernel_size
        
        self.conv1 = nn.Conv1d(self.n_features, 
                               self.n_features,
                               self.kernel_size,
                               padding = 1)
        
        self.conv2 = nn.Conv1d(self.n_features, 
                               self.n_features,
                               self.kernel_size,
                               padding = 1)
        
        self.bn1 = nn.BatchNorm1d(self.n_features)
        self.bn2 = nn.BatchNorm1d(self.n_features)
        
        self.activation = nn.ReLU()
        
        
    def forward(self, inputs):
        
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + inputs
        
        out = self.activation(out)
        
        return out



class ReductionBlock(nn.Module):
    
    '''
    
    Composant pour la reduction de la longueur du signal et augmentation du nombre de features
    
    (divise par 2 la longueur du signal et multiplie par 2 le nombre de features)
    
    '''
    
    
    def __init__(self, in_features, out_features, kernel_size = 3):
        
        super(ReductionBlock, self).__init__()
        
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.kernel_size = kernel_size
        
        self.resize_conv = nn.Conv1d(self.in_features, self.out_features, 1, stride=2)
        self.bn0 = nn.BatchNorm1d(self.out_features)
        
        
        
        self.conv1 = nn.Conv1d(self.out_features, 
                               self.out_features,
                               self.kernel_size,
                               padding = 1)
        
        self.conv2 = nn.Conv1d(self.out_features, 
                               self.out_features,
                               self.kernel_size,
                               padding = 1)
        
        self.bn1 = nn.BatchNorm1d(self.out_features)
        self.bn2 = nn.BatchNorm1d(self.out_features)
        
        self.activation = nn.ReLU()
        
        
    def forward(self, inputs):
        
        out = self.resize_conv(inputs)
        out = self.bn0(out)
        
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        
        return out


class Resnet34_1d(nn.Module):

    def __init__(self, seq_len, basic_block, in_planes,n_classes):
        
        super(Resnet34_1d, self).__init__()
        
        self.seq_len = seq_len
        self.basic_block = basic_block
        self.in_planes = in_planes
        self.n_classes = n_classes

        self.conv1 = nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.activation = nn.ReLU()
        
        self.block1 = self.make_block(3, 64)
        
        self.reduction2 = ReductionBlock(64,128)
        self.block2 = self.make_block(3, 128)
        
        self.reduction3 = ReductionBlock(128,256)
        self.block3 = self.make_block(5, 256)
        
        self.reduction4 = ReductionBlock(256,512)
        self.block4 = self.make_block(2, 512)
        
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(512 * self.seq_len // 32, self.n_classes)



    def make_block(self,size, n_features, kernel_size = 3):
        
        '''
        
        Constructuion d'un block composé de size Basic_block successifs
        
        
        '''
        
        
        
        basic_blocks = []
        
        for i in range(size):
            
            basic_blocks.append(self.basic_block(n_features,kernel_size))
        
        
        return nn.Sequential(*basic_blocks)

    
    def forward(self,inputs):
        
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.maxpool(out)
        
        out = self.block1(out)
        
        out = self.reduction2(out)
        out = self.block2(out)
        
        out = self.reduction3(out)
        out = self.block3(out)
        
        out = self.reduction4(out)
        out = self.block4(out)
        
        
        out = self.flatten(out)
        out = self.fc1(out)
        
        
        
        return out


if __name__ == '__main__':
    
    import torch
    
    net = Resnet34_1d(256, BasicBlock, 3, 10)

    inputs = torch.randn(5,3,256)
    
    print(net(inputs).size())










