#%%
import torch
import torch.nn as nn
import torch.nn.functional as F


FDD_MODEL_NAMES = ['ori-low-high','freq','freq-tfa']
TFA_MODEL_NAMES = ['tfa', 'freq-tfa']

class REDCNNOri(nn.Module):
    def __init__(self, in_ch=1, out_ch=96, model_choice='ori', alpha=0):
        super(REDCNNOri, self).__init__()
        
        self.model_choice = model_choice
        self.alpha = alpha 
        
        if self.model_choice == 'ori-low-high' and alpha == 0:
            self.reduce_ch = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1, padding=0)
        
        self.bn1 = nn.BatchNorm2d(out_ch) if self.model_choice == 'BN' else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_ch) if self.model_choice == 'BN' else nn.Identity()
        self.bn3 = nn.BatchNorm2d(out_ch) if self.model_choice == 'BN' else nn.Identity()
        self.bn4 = nn.BatchNorm2d(out_ch) if self.model_choice == 'BN' else nn.Identity()
        self.bn5 = nn.BatchNorm2d(out_ch) if self.model_choice == 'BN' else nn.Identity()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.bn1(out)
        out = self.relu(self.conv2(out))
        out = self.bn2(out)
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.bn3(out)
        out = self.relu(self.conv4(out))
        out = self.bn4(out)
        residual_3 = out
        mid_out = self.relu(self.conv5(out))
        out = self.bn5(mid_out)
        
        
        # decoder
        out = self.tconv1(mid_out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
      
        out = self.tconv5(self.relu(out))
        
        if self.model_choice == 'ori-low-high':
            if self.alpha == 0:
                residual_1 = self.reduce_ch(residual_1)
                residual_1 = F.relu(residual_1)
            elif self.alpha == 1:
                residual_1 = residual_1[:,:1,:,:] # only fetch the first dimension
            
        out += residual_1

        out = self.relu(out)
        return out
    
    def get_final_features(self, x, detach=True):
        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        # Forward pass up to the mid_out

        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        mid_out = self.relu(self.conv5(out))
        
        return func(mid_out)
    
class InstitudeAwareModule(nn.Module):
    def __init__(self, in_features, num_classes):
        # Abdomen: 0, Chest: 1, Head:2
        super(InstitudeAwareModule, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x
    
class REDCNNTFA(nn.Module):
    def __init__(self, in_ch=1, out_ch=96, model_choice='ori', alpha=0):
        super(REDCNNTFA, self).__init__()
        
        self.model_choice = model_choice
        self.alpha = alpha 
        self.tfa = TextFeatureAttention()
        
        if self.model_choice == 'ori-low-high' and alpha == 0:
            self.reduce_ch = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1, padding=0)
        

        self.bn1 = nn.BatchNorm2d(out_ch) if self.model_choice == 'BN' else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_ch) if self.model_choice == 'BN' else nn.Identity()
        self.bn3 = nn.BatchNorm2d(out_ch) if self.model_choice == 'BN' else nn.Identity()
        self.bn4 = nn.BatchNorm2d(out_ch) if self.model_choice == 'BN' else nn.Identity()
        self.bn5 = nn.BatchNorm2d(out_ch) if self.model_choice == 'BN' else nn.Identity()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x, text_features):
        # encoder
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.bn1(out)
        out = self.relu(self.conv2(out))
        out = self.bn2(out)
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.bn3(out)
        out = self.relu(self.conv4(out))
        out = self.bn4(out)
        residual_3 = out
        mid_out = self.relu(self.conv5(out))
        out = self.bn5(mid_out)
        
        out = self.tfa(text_features, out)
        
        # decoder
        out = self.tconv1(mid_out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
      
        out = self.tconv5(self.relu(out))
        
        if self.model_choice == 'ori-low-high':
            if self.alpha == 0:
                residual_1 = self.reduce_ch(residual_1)
                residual_1 = F.relu(residual_1)
            elif self.alpha == 1:
                residual_1 = residual_1[:,:1,:,:] # only fetch the first dimension
            
        out += residual_1

        out = self.relu(out)
        return out
    
    def get_final_features(self, x, detach=True):
        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        # Forward pass up to the mid_out

        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        mid_out = self.relu(self.conv5(out))
        
        return func(mid_out)

class REDCNNFreq(nn.Module):
    def __init__(self, alpha = 0):
        super(REDCNNFreq, self).__init__()
        self.alpha = alpha
        self.model_choice='freq'
        self.low_freq_red = REDCNNOri(in_ch=2, out_ch=96, model_choice='ori-low-high', alpha=1)
        self.high_freq_red = REDCNNOri(in_ch=2, out_ch=96, model_choice='ori-low-high', alpha=1)
        if self.alpha == 1:
            self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=0)
            self.conv2 = nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=0)
            self.tconv1 = nn.ConvTranspose2d(96, 96, kernel_size=5, stride=1, padding=0)
            self.tconv2 = nn.ConvTranspose2d(96, 1, kernel_size=5, stride=1, padding=0)
             
    def forward(self, x):
        x_low = self.low_freq_red(x[:,[1,0],:,:])
        x_high = self.high_freq_red(x[:,[2,0],:,:])
        
        if self.alpha == 0:
            out = x_low + x_high
            out = F.relu(out)
        elif self.alpha == 1:
            out = torch.cat([x_low + x_high, x_low, x_high], dim=1)
            res = out[:,[0],:,:]
            out = F.relu(self.conv1(out))
            out = F.relu(self.conv2(out))
            out = F.relu(self.tconv1(out))
            out = F.relu(self.tconv2(out))
            out = out + res # you cannot use += here, += is an inplace opeation which will cause the error
            out = F.relu(out) 
        return out
  
class REDCNNFreqTFA(nn.Module):
    def __init__(self, alpha = 0):
        super(REDCNNFreqTFA, self).__init__()
        self.alpha = alpha
        self.model_choice='freq-tfa'
        self.low_freq_red = REDCNNTFA(in_ch=2, out_ch=96, model_choice='ori-low-high', alpha=1)
        self.high_freq_red = REDCNNTFA(in_ch=2, out_ch=96, model_choice='ori-low-high', alpha=1)
        if self.alpha == 1:
            self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=0)
            self.conv2 = nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=0)
            self.tconv1 = nn.ConvTranspose2d(96, 96, kernel_size=5, stride=1, padding=0)
            self.tconv2 = nn.ConvTranspose2d(96, 1, kernel_size=5, stride=1, padding=0)
             
    def forward(self, x, text_features):
        x_low = self.low_freq_red(x[:,[1,0],:,:], text_features)
        x_high = self.high_freq_red(x[:,[2,0],:,:], text_features)
        
        if self.alpha == 0:
            out = x_low + x_high
            out = F.relu(out)
        elif self.alpha == 1:
            out = torch.cat([x_low + x_high, x_low, x_high], dim=1)
            res = out[:,[0],:,:]
            out = F.relu(self.conv1(out))
            out = F.relu(self.conv2(out))
            out = F.relu(self.tconv1(out))
            out = F.relu(self.tconv2(out))
            out = out + res # you cannot use += here, += is an inplace opeation which will cause the error
            out = F.relu(out) 
        return out
            
class TextFeatureAttention(nn.Module):
    def __init__(self, input_dim=512, output_dim=96):
        super(TextFeatureAttention, self).__init__()
        self.dim_reduction = nn.Linear(input_dim, output_dim)
    
    def forward(self, text_features, bottleneck_features):
        # Assuming text_features is [batch_size, 512]
        reduced_text_features = self.dim_reduction(text_features)  
        attention_weights=F.softmax(reduced_text_features, dim=-1)
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, 96, 1, 1]
        weighted_bottleneck = bottleneck_features * attention_weights  # Shape: [batch_size, 96, H, W]
        return weighted_bottleneck            

def get_ldct_model(choice, alpha = 0):
    # alpha means the variant of the model
    models = {
        'ori': REDCNNOri(),
        'ori-low-high': REDCNNOri(in_ch=3, model_choice='ori-low-high', alpha=alpha),
        'BN': REDCNNOri(model_choice='BN'),
        'freq': REDCNNFreq(alpha=alpha),
        'freq-tfa': REDCNNFreqTFA(alpha=alpha),
        'tfa': REDCNNTFA(model_choice='tfa', alpha=alpha),
    }
    assert choice in models.keys(), '--model_choice not in {}'.format(models.keys())
    return models[choice]
    