'''

Implémentation de VGG19: https://arxiv.org/pdf/1409.1556.pdf


'''

import torch.nn as nn

class ConvBlock(nn.Module):
    
    '''
    
    Construction d'un bloc de convolutions
    
    '''
    
    def __init__(self, size, in_features, out_features):
        
        super(ConvBlock, self).__init__()
        
        self.size = size
        self.in_features = in_features
        self.in_features = in_features
        
        seq = [nn.Conv2d(in_features,out_features,kernel_size=3, padding=1),nn.ReLU()]
        
        for i in range(size - 1):
            
            seq.append(nn.Conv2d(out_features,out_features,kernel_size=3, padding=1))
            seq.append(nn.ReLU())
        
        self.convolutions = nn.Sequential(*seq)
    
    def forward(self, inputs):
        
        
        
        out = self.convolutions(inputs)
        
        return out


class VGG(nn.Module):
    
    '''
    inputs: [batch,(224 × 224 RGB image)]
    outputs: [batch, n_classes]
    
    '''
    
    def __init__(self, block_sizes, block_features, n_classes):
        
        super(VGG, self).__init__()
        
        
        self.block_sizes = block_sizes
        self.n_classes = n_classes
        
        convolutions = []
        
        for i in range(len(block_sizes)):
            
            convolutions.append(ConvBlock(block_sizes[i],block_features[i],block_features[i+1]))
            convolutions.append(nn.MaxPool2d(2,2))
            
        self.convolutions = nn.Sequential(*convolutions)
        
        self.flatten = nn.Flatten(start_dim=1)
        
        self.linear_layers = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                           nn.ReLU(),
                                           nn.Linear(4096, 4096),
                                           nn.ReLU(),
                                           nn.Linear(4096, n_classes))
        
    def forward(self, inputs):
        
        out = self.convolutions(inputs)
        out = self.flatten(out)
        out = self.linear_layers(out)
        
        return out



if __name__ == '__main__':
    
    import torch
    
    VGG19 = VGG([2,2,4,4,4], [3,64,128,256,512,512], 10)

    inputs = torch.randn(5,3,224,224)
    
    print(VGG19(inputs).size())


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        