import torch
import torch.nn as nn
from transformers import BertModel

class PhyCell_Cell(nn.Module):
    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=1):
        super(PhyCell_Cell, self).__init__()
        self.input_dim  = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.F = nn.Sequential()
        self.F.add_module('conv1', nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim, kernel_size=self.kernel_size, stride=(1,1), padding=self.padding))
        self.F.add_module('bn1',nn.GroupNorm( 7 ,F_hidden_dim))        
        self.F.add_module('conv2', nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim, kernel_size=(1,1), stride=(1,1), padding=(0,0)))

        self.convgate = nn.Conv2d(in_channels=self.input_dim + self.input_dim,
                              out_channels= self.input_dim,
                              kernel_size=(3,3),
                              padding=(1,1), bias=self.bias)

    def forward(self, x, hidden): # x [batch_size, hidden_dim, height, width]      
        combined = torch.cat([x, hidden], dim=1)  # concatenate along channel axis
        combined_conv = self.convgate(combined)
        K = torch.sigmoid(combined_conv)
        hidden_tilde = hidden + self.F(hidden)        # prediction
        next_hidden = hidden_tilde + K * (x-hidden_tilde)   # correction , Haddamard product     
        return next_hidden

class PhyCell(nn.Module):
    def __init__(self, input_shape, input_dim, F_hidden_dims, n_layers, kernel_size, device):
        super(PhyCell, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H = []  
        self.device = device
             
        cell_list = []
        for i in range(0, self.n_layers):
            cell_list.append(PhyCell_Cell(input_dim=input_dim,
                                          F_hidden_dim=self.F_hidden_dims[i],
                                          kernel_size=self.kernel_size))                                     
        self.cell_list = nn.ModuleList(cell_list)
        
       
    def forward(self, input_, first_timestep=False): # input_ [batch_size, 1, channels, width, height]    
        batch_size = input_.data.size()[0]
        if (first_timestep):   
            self.initHidden(batch_size) # init Hidden at each forward start
              
        for j,cell in enumerate(self.cell_list):
            if j==0: # bottom layer
                # print(input_.shape, self.H[j].shape)
                self.H[j] = cell(input_, self.H[j])
            else:
                self.H[j] = cell(self.H[j-1],self.H[j])
        
        return self.H , self.H 
    
    def initHidden(self,batch_size):
        self.H = [] 
        for i in range(self.n_layers):
            self.H.append( torch.zeros(batch_size, self.input_dim, self.input_shape[0], self.input_shape[1]).to(self.device) )

    def setHidden(self, H):
        self.H = H

        
class ConvLSTM_Cell(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):              
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTM_Cell, self).__init__()
        
        self.height, self.width = input_shape
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)
                 
    # we implement LSTM that process only one timestep 
    def forward(self,x, hidden): # x [batch, hidden_dim, width, height]          
        h_cur, c_cur = hidden
        
        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dims, n_layers, kernel_size,device):
        super(ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H, self.C = [],[]   
        self.device = device
        
        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]
            print('layer ',i,'input dim ', cur_input_dim, ' hidden dim ', self.hidden_dims[i])
            cell_list.append(ConvLSTM_Cell(input_shape=self.input_shape,
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dims[i],
                                          kernel_size=self.kernel_size))                                     
        self.cell_list = nn.ModuleList(cell_list)
        
       
    def forward(self, input_, first_timestep=False): # input_ [batch_size, 1, channels, width, height]    
        batch_size = input_.data.size()[0]
        if (first_timestep):   
            self.initHidden(batch_size) # init Hidden at each forward start
              
        for j,cell in enumerate(self.cell_list):
            if j==0: # bottom layer
                self.H[j], self.C[j] = cell(input_, (self.H[j],self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j-1],(self.H[j],self.C[j]))
        
        return (self.H,self.C) , self.H   # (hidden, output)
    
    def initHidden(self,batch_size):
        self.H, self.C = [],[]  
        for i in range(self.n_layers):
            self.H.append( torch.zeros(batch_size,self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device) )
            self.C.append( torch.zeros(batch_size,self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device) )
    
    def setHidden(self, hidden):
        H,C = hidden
        self.H, self.C = H,C
 

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=(3,3), stride=stride, padding=1),
                nn.GroupNorm(16,nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_upconv, self).__init__()
        if (stride ==2):
            output_padding = 1
        else:
            output_padding = 0
        self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channels=nin,out_channels=nout,kernel_size=(3,3), stride=stride,padding=1,output_padding=output_padding),
                nn.GroupNorm(16,nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)
        
class encoder_E(nn.Module):
    def __init__(self, nc=1, nf=32):
        super(encoder_E, self).__init__()
        # input is (1) x 64 x 64
        self.c1 = dcgan_conv(nc, nf, stride=2) # (32) x 32 x 32
        self.c2 = dcgan_conv(nf, nf, stride=1) # (32) x 32 x 32
        self.c3 = dcgan_conv(nf, 2*nf, stride=2) # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)  
        h2 = self.c2(h1)    
        h3 = self.c3(h2)
        return h3

class decoder_D(nn.Module):
    def __init__(self, nc=1, nf=32):
        super(decoder_D, self).__init__()
        self.upc1 = dcgan_upconv(2*nf, nf, stride=2) #(32) x 32 x 32
        self.upc2 = dcgan_upconv(nf, nf, stride=1) #(32) x 32 x 32
        self.upc3 = nn.ConvTranspose2d(in_channels=nf,out_channels=nc,kernel_size=(3,3),stride=2,padding=1,output_padding=1)  #(nc) x 64 x 64

    def forward(self, input):      
        d1 = self.upc1(input) 
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)  
        return d3  


class encoder_specific(nn.Module):
    def __init__(self, nc=64, nf=64):
        super(encoder_specific, self).__init__()
        self.c1 = dcgan_conv(nc, nf, stride=1) # (64) x 16 x 16
        self.c2 = dcgan_conv(nf, nf, stride=1) # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)  
        h2 = self.c2(h1)     
        return h2

class decoder_specific(nn.Module):
    def __init__(self, nc=64, nf=64):
        super(decoder_specific, self).__init__()
        self.upc1 = dcgan_upconv(nf, nf, stride=1) #(64) x 16 x 16
        self.upc2 = dcgan_upconv(nf, nc, stride=1) #(32) x 32 x 32
        
    def forward(self, input):
        d1 = self.upc1(input) 
        d2 = self.upc2(d1)  
        return d2       

        
class EncoderRNN(torch.nn.Module):
    def __init__(self, phycell, convcell, nc, nf, device):
        super(EncoderRNN, self).__init__()
        self.encoder_E = encoder_E(nc=nc)   # general encoder 64x64x1 -> 32x32x32
        self.encoder_Ep = encoder_specific() # specific image encoder 32x32x32 -> 16x16x64
        self.encoder_Er = encoder_specific() 
        self.decoder_Dp = decoder_specific() # specific image decoder 16x16x64 -> 32x32x32 
        self.decoder_Dr = decoder_specific()     
        self.decoder_D = decoder_D(nc=nc)  # general decoder 32x32x32 -> 64x64x1 

        self.encoder_E = self.encoder_E.to(device)
        self.encoder_Ep = self.encoder_Ep.to(device) 
        self.encoder_Er = self.encoder_Er.to(device) 
        self.decoder_Dp = self.decoder_Dp.to(device) 
        self.decoder_Dr = self.decoder_Dr.to(device)               
        self.decoder_D = self.decoder_D.to(device)
        self.phycell = phycell.to(device)
        self.convcell = convcell.to(device)

    def forward(self, input, first_timestep=False, decoding=False):
        input = self.encoder_E(input) # general encoder 64x64x1 -> 32x32x32
    
        if decoding:  # input=None in decoding phase
            input_phys = None
        else:
            input_phys = self.encoder_Ep(input)
        input_conv = self.encoder_Er(input)     

        hidden1, output1 = self.phycell(input_phys, first_timestep)
        hidden2, output2 = self.convcell(input_conv, first_timestep)

        decoded_Dp = self.decoder_Dp(output1[-1])
        decoded_Dr = self.decoder_Dr(output2[-1])
        
        out_phys = torch.sigmoid(self.decoder_D(decoded_Dp)) # partial reconstructions for vizualization
        out_conv = torch.sigmoid(self.decoder_D(decoded_Dr))

        concat = decoded_Dp + decoded_Dr   
        output_image = torch.sigmoid( self.decoder_D(concat ))
        return out_phys, hidden1, output_image, out_phys, out_conv




class PhyDNet(nn.Module):
    def __init__(self, device, nc=3, nf=16):
        super(PhyDNet, self).__init__()
        # generation
        pretrain = torch.load('r3d34_K_200ep.pth', map_location='cpu')
        self.first_n_frames_encoder = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), n_classes=700)
        self.first_n_frames_encoder.load_state_dict(pretrain['state_dict'])
        self.first_n_masks_encoder = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), n_classes=700)
        self.first_n_masks_encoder.load_state_dict(pretrain['state_dict'])
        self.global_context = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), n_classes=700)
        self.global_context.load_state_dict(pretrain['state_dict'])
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(2304, 256)
        self.fc = nn.Linear(256, 1)
        self.phycell = PhyCell(input_shape=(32,32), input_dim=64, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7), device=device) 
        self.convcell = ConvLSTM(input_shape=(32,32), input_dim=64, hidden_dims=[128,128,64], n_layers=3, kernel_size=(3,3), device=device)   
        self.encoder = EncoderRNN(self.phycell, self.convcell, nc, nf, device)
        self.device = device

    def forward(self, task, images, masks, queries, use_teacher_forcing, first_n_frame_dynamics, max_seq_len):
        sequence_len = len(images)
        batch_size, channels, height, width = images[0].shape
        images_first_n_frames = []
        decoded_first_n_frames = []
        decoded_images = []
        assert first_n_frame_dynamics <= sequence_len

        for i in range(sequence_len):
            if i < first_n_frame_dynamics:
                # encode frames
                images_i = images[i].to(self.device)
                images_first_n_frames.append(images_i)

            elif i >= first_n_frame_dynamics:
                if not use_teacher_forcing:
                    images_i = output_image
                else:
                    images_i = images[i].to(self.device)

            decoder_output, decoder_hidden, output_image, _, _ = self.encoder(images_i, (i==0))

            # decode
            if i >= first_n_frame_dynamics - 1 and i < max_seq_len - 1:
                decoded_images.append(output_image)
            elif i < first_n_frame_dynamics - 1:
                decoded_first_n_frames.append(output_image)

        
        images_first_n_frames = torch.stack(images_first_n_frames).permute(1, 2, 0, 3, 4).to(self.device)   # batch_size, channels, seq_len, height, width
        masks_first_n_frames = torch.stack(masks).permute(1, 2, 0, 3, 4).to(self.device)    # batch_size, 1, seq_len, height, width
        masks_first_n_frames = masks_first_n_frames.repeat((1, 3, 1, 1, 1))

        # add classifier
        inp_decoded_images = torch.stack(decoded_images)  # seq_len, batch_size, channels, width, height
        # add first n frames and masks information
        encoded_first_n_frames = self.first_n_frames_encoder(
            images_first_n_frames)  # batch_size, 1, encoded_feature_size
        encoded_first_n_masks = self.first_n_masks_encoder(masks_first_n_frames)  # batch_size, 1, encoded_feature_size
        encoded_global_context = self.global_context(
            inp_decoded_images.permute(1, 2, 0, 3, 4))  # batch_size, 1, encoded_feature_size
        # add task type information
        input_ids = torch.squeeze(queries['input_ids'], dim=1).to(self.device)
        attention_mask = torch.squeeze(queries['attention_mask'], dim=1).to(self.device)
        token_type_ids = torch.squeeze(queries['token_type_ids'], dim=1).to(self.device)
        task_conditioning = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                      token_type_ids=token_type_ids).last_hidden_state[:, 0, :]
        final_image_features = torch.cat(
            [encoded_first_n_frames, encoded_first_n_masks, encoded_global_context, task_conditioning], dim=1)
        classification = self.fc(self.relu(self.dropout(self.linear(self.relu(self.dropout(final_image_features))))))

        return classification, decoded_images, torch.stack(decoded_first_n_frames)




class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x



def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def get_inplanes():
    return [64, 128, 256, 512]