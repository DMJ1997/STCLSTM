import torch.nn.functional as F
import torch.nn as nn
import torch

def maps_2_cubes(x, b, d):
    x_b, x_c, x_h, x_w = x.shape
    #要求连续存储  把原先tensor中的数据按照行优先的顺序排成一个一维的数据（这里应该是因为要求地址是连续存储的），然后按照参数组合成其他维度的tensor。
    x = x.contiguous().view(b, d, x_c, x_h, x_w)

    return x.permute(0, 2, 1, 3, 4)



class R_3(nn.Module):
    def __init__(self, block_channel):
        super(R_3, self).__init__()

        num_features = 64 + block_channel[3] // 32 + 8
        self.conv0 = nn.Conv2d(num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)

        self.convh = nn.Conv2d(num_features, 8, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x0 =self.conv0(x)
        x0 =self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        h = self.convh(x1)
        pred_depth = self.conv2(x1)

        return h, pred_depth

class R_CLSTM_5(nn.Module):
    def __init__(self, block_channel):
        super(R_CLSTM_5, self).__init__()
        num_features = 64 + block_channel[3] // 32
        self.Refine = R_3(block_channel)
        #遗忘门
        self.F_t = nn.Sequential(nn.Conv2d(in_channels=num_features + 8, out_channels=8, kernel_size=3, padding=1),
                                 nn.Sigmoid())
        #输入门
        self.I_t = nn.Sequential(nn.Conv2d(in_channels=num_features + 8, out_channels=8, kernel_size=3, padding=1),
                                 nn.Sigmoid())
        #候选状态
        self.C_t = nn.Sequential(nn.Conv2d(in_channels=num_features + 8, out_channels=8, kernel_size=3, padding=1),
                                 nn.Tanh())
        #输出门
        self.Q_t = nn.Sequential(nn.Conv2d(in_channels=num_features + 8, out_channels=num_features, kernel_size=3, padding=1),
                                 nn.Sigmoid())

    def forward(self, input_tensor, b, d):
        input_tensor = maps_2_cubes(input_tensor, b, d)
        b, c, d, h, w = input_tensor.shape

        h_state_init = torch.zeros(b, 8, h, w).to('cuda')
        c_state_init = torch.zeros(b, 8, h, w).to('cuda')

        seq_len = d

        h_state, c_state = h_state_init, c_state_init
        output_inner = []

        for t in range(seq_len):
            input_cat = torch.cat((input_tensor[:,:,t,:,:], h_state), dim=1)

            c_state = self.F_t(input_cat) * c_state + self.I_t(input_cat) * self.C_t(input_cat)
            h_state, p_depth = self.Refine(torch.cat((c_state, self.Q_t(input_cat)), 1))

            output_inner.append(p_depth)

        layer_out = torch.stack(output_inner, dim=2)
        #测试时间的代码 等待GPU
        torch.cuda.synchronize()

        return layer_out





