import torch
import torch.nn as nn
import torchvision.models as models
import math
from transformers import BertModel
from torchvision import transforms


class Model(nn.Module):
    def __init__(self, device, span_num=1, model_type='pip', nc=3, nf=16):
        super(Model, self).__init__()
        # generation
        self.c1 = dcgan_conv(nc, nf, stride=1)  # input is (3) x 128 x 128
        self.c2 = dcgan_conv(nf, 2*nf, stride=1)
        self.c3 = dcgan_conv(2*nf, 4*nf, stride=2)
        self.c4 = dcgan_conv(4*nf, 4*nf, stride=1)
        self.c5 = dcgan_conv(4*nf, 8*nf, stride=2)
        self.c6 = dcgan_conv(8*nf, 8*nf, stride=1)
        self.ConvLSTMCell1 = ConvLSTMCell(input_shape=(32,32), input_dim=8*nf, hidden_dim=8*nf, kernel_size=(3,3), device=device)
        self.ConvLSTMCell2 = ConvLSTMCell(input_shape=(32,32), input_dim=8*nf, hidden_dim=8*nf, kernel_size=(3,3), device=device)
        self.ConvLSTMCell3 = ConvLSTMCell(input_shape=(32,32), input_dim=8*nf, hidden_dim=8*nf, kernel_size=(3,3), device=device)
        self.upc1 = dcgan_upconv(8*nf*2, 8*nf, stride=1)
        self.upc2 = dcgan_upconv(8*nf*2, 4*nf, stride=2)
        self.upc3 = dcgan_upconv(4*nf*2, 4*nf, stride=1)
        self.upc4 = dcgan_upconv(4*nf*2, 2*nf, stride=2)
        self.upc5 = dcgan_upconv(2*nf*2, nf, stride=1)
        self.upc6 = nn.ConvTranspose2d(in_channels=nf*2,out_channels=nc,kernel_size=(3,3),stride=1,padding=1,output_padding=0)

        if model_type == 'pip':
            # span selector
            self.span_predict = SpanPredict(span_num, device)
        elif model_type == 'ablation':
            self.ablation = Ablation(device)
        elif model_type == 'baseline':
            self.baseline = Baseline(device)
        self.device = device
        self.model_type = model_type

    def forward(self, task, images, masks, queries, teacher_forcing_batch, first_n_frame_dynamics, max_seq_len):
        sequence_len = len(images)
        batch_size, channels, height, width = images[0].shape
        images_first_n_frames = []
        decoded_images = []
        assert first_n_frame_dynamics <= sequence_len

        for i in range(sequence_len):
            if i < first_n_frame_dynamics:
                # encode frames
                images_i = images[i].to(self.device)
                images_first_n_frames.append(images_i)

            elif i >= first_n_frame_dynamics:
                if self.model_type == 'ablation' or self.model_type == 'baseline':
                    break
                images_i = images[i].to(self.device)
                for j in range(batch_size):
                    # teacher forcing
                    if not teacher_forcing_batch[j]:
                        images_i[j] = decoded_frame[j]

            if self.model_type == 'ablation' or self.model_type == 'baseline':
                continue

            # encode
            inp_1 = self.c1(images_i)
            inp_2 = self.c2(inp_1)
            inp_3 = self.c3(inp_2)
            inp_4 = self.c4(inp_3)
            inp_5 = self.c5(inp_4)
            lstm_inp = self.c6(inp_5)
            
            # ConvLSTM
            if i == 0:
                hx1, cx1 = self.ConvLSTMCell1(lstm_inp, batch_size)
                hx2, cx2 = self.ConvLSTMCell2(hx1, batch_size)
                hx3, cx3 = self.ConvLSTMCell3(hx2, batch_size)
            else:
                hx1, cx1 = self.ConvLSTMCell1(lstm_inp, batch_size, (hx1, cx1))
                hx2, cx2 = self.ConvLSTMCell2(hx1, batch_size, (hx2, cx2))
                hx3, cx3 = self.ConvLSTMCell3(hx2, batch_size, (hx3, cx3))

            # decode
            if i >= first_n_frame_dynamics - 1 and i < max_seq_len - 1:
                # decode frames
                out_1 = self.upc1(torch.cat([hx3, lstm_inp], dim=1))
                out_2 = self.upc2(torch.cat([out_1, inp_5], dim=1))
                out_3 = self.upc3(torch.cat([out_2, inp_4], dim=1))
                out_4 = self.upc4(torch.cat([out_3, inp_3], dim=1))
                out_5 = self.upc5(torch.cat([out_4, inp_2], dim=1))
                decoded_frame = self.upc6(torch.cat([out_5, inp_1], dim=1))
                decoded_images.append(decoded_frame)

        if self.model_type == 'pip':
            images_first_n_frames = torch.stack(images_first_n_frames).permute(1, 2, 0, 3, 4).to(self.device)   # batch_size, channels, seq_len, height, width
            masks_first_n_frames = torch.stack(masks).permute(1, 2, 0, 3, 4).to(self.device)    # batch_size, 1, seq_len, height, width
            classification, all_r, jsd_loss = self.span_predict(decoded_images, images_first_n_frames, masks_first_n_frames.repeat((1, 3, 1, 1, 1)), queries)
        elif self.model_type == 'ablation':
            images_first_n_frames = torch.stack(images_first_n_frames).permute(1, 2, 0, 3, 4).to(self.device)   # batch_size, channels, seq_len, height, width
            masks_first_n_frames = torch.stack(masks).permute(1, 2, 0, 3, 4).to(self.device)    # batch_size, 1, seq_len, height, width
            all_r, jsd_loss = None, None
            classification = self.ablation(images_first_n_frames, masks_first_n_frames.repeat((1, 3, 1, 1, 1)), queries)
        elif self.model_type == 'baseline':
            images_first_n_frames = torch.stack(images_first_n_frames).permute(1, 2, 0, 3, 4).to(self.device)   # batch_size, channels, seq_len, height, width
            masks_first_n_frames = torch.stack(masks).permute(1, 2, 0, 3, 4).to(self.device)    # batch_size, 1, seq_len, height, width
            all_r, jsd_loss = None, None
            classification = self.baseline(images_first_n_frames, masks_first_n_frames.repeat((1, 3, 1, 1, 1)), queries)

        return classification, decoded_images, all_r, jsd_loss


class SpanPredict(nn.Module):
    def __init__(self, span_num, device):
        super(SpanPredict, self).__init__()
        # span weights
        self.weights_z = nn.Parameter(torch.randn((2048+512*3+768), requires_grad=True))
        self.span_weights_p = nn.ParameterList([nn.Parameter(torch.randn((2048+512*3+768), requires_grad=True)) for i in range(span_num)])
        self.span_weights_q = nn.ParameterList([nn.Parameter(torch.randn((2048+512*3+768), requires_grad=True)) for i in range(span_num)])
        self.mixing_coefficients = nn.Parameter(torch.randn((1, 1, span_num), requires_grad=True)) # 1, 1, num_spans

        # encoders
        resnet50 = models.resnet50(pretrained=True)
        removed = list(resnet50.children())[:-1]
        self.frame_encoder = torch.nn.Sequential(*removed)
        pretrain = torch.load('r3d34_K_200ep.pth', map_location='cpu')
        self.first_n_frames_encoder = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), n_classes=700)
        self.first_n_frames_encoder.load_state_dict(pretrain['state_dict'])
        self.first_n_masks_encoder = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), n_classes=700)
        self.first_n_masks_encoder.load_state_dict(pretrain['state_dict'])
        self.global_context = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), n_classes=700)
        self.global_context.load_state_dict(pretrain['state_dict'])
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        self.span_weights_softmax = nn.Softmax(dim=1)
        self.mixing_coeff_softmax = nn.Softmax(dim=2)
        self.span_num = span_num
        self.normalize = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])
        self.device = device

    def forward(self, inp_decoded_images, images_first_n_frames, masks_first_n_frames, queries, eps=1e-8):
        inp_decoded_images = torch.stack(inp_decoded_images)    # seq_len, batch_size, channels, width, height
        decoded_images = inp_decoded_images.permute(1, 0, 2, 3, 4) # batch_size, seq_len, channels, width, height
        B, S, C, W, H = decoded_images.shape
        decoded_images = decoded_images.reshape(B*S, C, W, H)
        # get individual frame features
        normalized_decoded_images = self.normalize(decoded_images)
        decoded_image_feats = self.frame_encoder(normalized_decoded_images)
        decoded_image_feats = torch.squeeze(decoded_image_feats)
        _, decoded_image_feat_size = decoded_image_feats.shape
        decoded_image_feats = decoded_image_feats.reshape(B, S, decoded_image_feat_size) # batch_size, seq_len, image_feature_size
        # add first n frames and masks information
        encoded_first_n_frames = torch.unsqueeze(self.first_n_frames_encoder(images_first_n_frames), dim=1) # batch_size, 1, encoded_feature_size
        encoded_first_n_masks = torch.unsqueeze(self.first_n_masks_encoder(masks_first_n_frames), dim=1)    # batch_size, 1, encoded_feature_size
        encoded_global_context = torch.unsqueeze(self.global_context(inp_decoded_images.permute(1, 2, 0, 3, 4)), dim=1) # batch_size, 1, encoded_feature_size
        # add task type information
        input_ids = torch.squeeze(queries['input_ids'], dim=1).to(self.device)
        attention_mask = torch.squeeze(queries['attention_mask'], dim=1).to(self.device)
        token_type_ids = torch.squeeze(queries['token_type_ids'], dim=1).to(self.device)
        task_conditioning = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state[:,0,:]
        task_conditioning = torch.unsqueeze(task_conditioning, dim=1)
        final_image_features = torch.cat([decoded_image_feats, encoded_first_n_frames.repeat((1, S, 1)), encoded_first_n_masks.repeat((1, S, 1)), encoded_global_context.repeat((1, S, 1)), task_conditioning.repeat((1, S, 1))], dim=2)

        all_r = []
        for i in range(self.span_num):
            softmax_p = self.span_weights_softmax(torch.matmul(final_image_features, self.span_weights_p[i])) # batch_size, seq_len
            softmax_q = self.span_weights_softmax(torch.matmul(final_image_features, self.span_weights_q[i])) # batch_size, seq_len
            softmax_q_flipped = torch.flip(softmax_q, (1,)) # batch_size, seq_len
            p = torch.cumsum(softmax_p, dim=1)  # batch_size, seq_len
            q = torch.cumsum(softmax_q_flipped, dim=1)  # batch_size, seq_len
            q = torch.flip(q, (1,)) # batch_size, seq_len
            # r shows the contribution of each frame
            r = p*q
            r = r / (torch.sum(r) + eps)
            all_r.append(r)
        
        all_r = torch.stack(all_r).permute(1, 2, 0) # batch_size, seq_len, span_num

        # NOTE: Jensen-Shannon divergence loss
        pi_r = self.mixing_coeff_softmax(self.mixing_coefficients) * all_r  # batch_size, seq_len, span_num
        overlap_p = torch.sum(pi_r, dim=2)  # batch_size, seq_len
        log_overlap_p = torch.log(overlap_p + 1e-40)    # batch_size, seq_len
        overlap_p_entropy = torch.sum(-overlap_p*log_overlap_p, dim=1) # batch_size
        log_all_r = torch.log(all_r + 1e-40)    # batch_size, seq_len, span_num
        conciseness_p_entropy = torch.sum(-all_r*log_all_r, dim=1)    # batch_size, span_num
        conciseness_p_entropy = torch.sum(torch.squeeze(self.mixing_coeff_softmax(self.mixing_coefficients), dim=1)*conciseness_p_entropy, dim=1)   # batch_size
        jsd_loss = overlap_p_entropy - conciseness_p_entropy

        all_r = torch.unsqueeze(all_r, dim=3)   # batch_size, seq_len, span_num, 1
        final_image_features = torch.unsqueeze(final_image_features, dim=2)   # batch_size, seq_len, 1, image_feature_size

        m = all_r * final_image_features # batch_size, seq_len, span_num, image_feature_size
        m = m.permute(0, 2, 1, 3)   # batch_size, span_num, seq_len, image_feature_size
        averaged_m = torch.mean(m, dim=2)   # batch_size, span_num, image_feature_size
        z = torch.matmul(averaged_m, self.weights_z)    # batch_size, span_num
        out = torch.unsqueeze(torch.sum(z, dim=1), dim=1)    # batch_size, 1

        return out, torch.squeeze(all_r, 3).permute(0, 2, 1), jsd_loss


class Ablation(nn.Module):
    def __init__(self, device):
        super(Ablation, self).__init__()
        # encoders
        pretrain = torch.load('r3d34_K_200ep.pth', map_location='cpu')
        self.first_n_frames_encoder = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), n_classes=700)
        self.first_n_frames_encoder.load_state_dict(pretrain['state_dict'])
        self.first_n_masks_encoder = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), n_classes=700)
        self.first_n_masks_encoder.load_state_dict(pretrain['state_dict'])
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(1792, 256)
        self.fc = nn.Linear(256, 1)
        
        self.device = device

    def forward(self, images_first_n_frames, masks_first_n_frames, queries):
        # add first n frames and masks information
        encoded_first_n_frames = self.first_n_frames_encoder(images_first_n_frames) # batch_size, 1, encoded_feature_size
        encoded_first_n_masks = self.first_n_masks_encoder(masks_first_n_frames)    # batch_size, 1, encoded_feature_size
        # add task type information
        input_ids = torch.squeeze(queries['input_ids'], dim=1).to(self.device)
        attention_mask = torch.squeeze(queries['attention_mask'], dim=1).to(self.device)
        token_type_ids = torch.squeeze(queries['token_type_ids'], dim=1).to(self.device)
        task_conditioning = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state[:,0,:]
        final_image_features = torch.cat([encoded_first_n_frames, encoded_first_n_masks, task_conditioning], dim=1)
        classification = self.fc(self.relu(self.dropout(self.linear(self.relu(self.dropout(final_image_features))))))

        return classification


class Baseline(nn.Module):
    def __init__(self, device):
        super(Baseline, self).__init__()
        # encoders
        resnet50 = models.resnet50(pretrained=True)
        removed = list(resnet50.children())[:-1]
        self.first_n_frames_encoder = torch.nn.Sequential(*removed)
        self.first_n_masks_encoder = torch.nn.Sequential(*removed)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.relu = nn.ReLU()
        self.normalize = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])
        self.dropout = nn.Dropout(0.5)
        self.avgpool = nn.AvgPool1d(3)
        self.linear1 = nn.Linear(4864, 1024)
        self.linear2 = nn.Linear(1024, 256)
        self.fc = nn.Linear(256, 1)
        
        self.device = device

    def forward(self, images_first_n_frames, masks_first_n_frames, queries):
        # add first n frames and masks information
        B, S, C, W, H = images_first_n_frames.shape
        images_first_n_frames = images_first_n_frames.reshape(B*S, C, W, H)
        masks_first_n_frames = masks_first_n_frames.reshape(B*S, C, W, H)
        encoded_first_n_frames = self.first_n_frames_encoder(self.normalize(images_first_n_frames)) # batch_size * seq_len, channel, width, height
        encoded_first_n_masks = self.first_n_masks_encoder(self.normalize(masks_first_n_frames))    # batch_size * seq_len, channel, width, height
        encoded_first_n_frames = encoded_first_n_frames.reshape(B, S, -1)
        encoded_first_n_masks = encoded_first_n_masks.reshape(B, S, -1)
        B, S, F = encoded_first_n_frames.shape
        encoded_first_n_frames = encoded_first_n_frames.permute(0, 2, 1)
        encoded_first_n_masks = encoded_first_n_masks.permute(0, 2, 1)
        encoded_first_n_frames = torch.squeeze(self.avgpool(encoded_first_n_frames), dim=2)
        encoded_first_n_masks = torch.squeeze(self.avgpool(encoded_first_n_masks), dim=2)
        # add task type information
        input_ids = torch.squeeze(queries['input_ids'], dim=1).to(self.device)
        attention_mask = torch.squeeze(queries['attention_mask'], dim=1).to(self.device)
        token_type_ids = torch.squeeze(queries['token_type_ids'], dim=1).to(self.device)
        task_conditioning = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state[:,0,:]
        final_image_features = torch.cat([encoded_first_n_frames, encoded_first_n_masks, task_conditioning], dim=1)
        classification = self.fc(self.relu(self.dropout(self.linear2(self.relu(self.dropout(self.linear1(self.relu(self.dropout(final_image_features)))))))))

        return classification


class ConvLSTMCell(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, device, bias=1):              
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
        super(ConvLSTMCell, self).__init__()
        
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

        self.input_shape = input_shape
        self.device = device
                 
    # we implement LSTM that process only one timestep 
    def forward(self, x, batch_size , hidden=None): # x [batch, hidden_dim, width, height]
        if hidden is None:
            h_cur, c_cur = self.init_hidden(batch_size)
        else:
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

    def init_hidden(self, batch_size):
        init_h = torch.zeros(batch_size, self.hidden_dim, self.input_shape[0], self.input_shape[1]).to(self.device)
        init_c = torch.zeros(batch_size, self.hidden_dim, self.input_shape[0], self.input_shape[1]).to(self.device)

        return init_h, init_c    


class dcgan_conv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=(3,3), stride=stride, padding=1),
                nn.GroupNorm(16,nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, inp):
        return self.main(inp)


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

    def forward(self, inp):
        return self.main(inp)



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
