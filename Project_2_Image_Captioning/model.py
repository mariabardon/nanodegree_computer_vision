import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size,vocab_size)
        
 
    def forward(self, features, captions):
        captions = self.embed(captions[:,:-1])
        features = features.unsqueeze(dim=1)
        x = torch.cat((features,captions), dim=1)
        x, h = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
       

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        captions = []
        x = inputs
        h = states
        for n in range(max_len):
            x, h = self.lstm(x,h)
            x = self.fc(x)
            new_caption = torch.argmax(x, dim=2)
            captions.append(new_caption.item())
            x = self.embed(new_caption)
        return captions
        