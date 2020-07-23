import torch.nn as nn
import torch
class ResBlock(nn.Module):
    def __init__(self, in_features):
        super(ResBlock, self).__init__()

        conv_block = [  nn.Conv2d(in_features, in_features, 3, stride=1, padding=1),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_features, in_features, 3, stride=1, padding=1),
                        nn.InstanceNorm2d(in_features) ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class UNet(nn.Module):
    # Simple UNet
    def __init__(self, num_classes, n_colors=3, regression=False, num_downs=4, norm_layer=nn.BatchNorm2d):
        super(UNet, self).__init__()
        if regression:
            num_classes = 1
        use_bias = norm_layer == nn.InstanceNorm2d
        self.num_downs = num_downs
        filters = n_colors
        downs = []
        out_filters = 64
        for down in range(num_downs):
            downs.append([
                         nn.Conv2d(filters, out_filters, kernel_size=3, bias=use_bias, padding=1),
                         norm_layer(out_filters),
                         nn.ReLU(True),
                         nn.Conv2d(out_filters, out_filters, kernel_size=3, bias=use_bias, padding=1),
                         norm_layer(out_filters),
                         nn.ReLU(True),
                         nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1),
                         norm_layer(out_filters),
                         nn.LeakyReLU(inplace=True)])
            filters = out_filters
            out_filters *= 2

        mid = []
        for _ in range(2):
            mid += [ResBlock(filters)]

        ups = []
        out_filters = filters // 2
        for up in range(num_downs):
            ups.append([nn.Conv2d(filters * 2, out_filters, kernel_size=3, bias=use_bias, padding=1),
                        norm_layer(out_filters),
                        nn.ReLU(True),
                        nn.Conv2d(out_filters, out_filters, kernel_size=3, bias=use_bias, padding=1),
                        norm_layer(out_filters),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(out_filters, out_filters,
                                          kernel_size=4, stride=2, padding=1),
                        norm_layer(out_filters),
                        nn.ReLU(inplace=True)])
            filters = out_filters
            out_filters //= 2
        if num_classes == 2 or num_classes == 1:
            tail = [nn.Conv2d(filters, 1, kernel_size=3, padding=1),
                    nn.Sigmoid()]
        else:
            tail = [nn.Conv2d(filters, num_classes, kernel_size=3, padding=1)]

        self.downs = nn.ModuleList([nn.Sequential(*downsample) for downsample in downs])
        self.ups = nn.ModuleList([nn.Sequential(*upsample) for upsample in ups])
        self.tail = nn.Sequential(*tail)
        self.mid = nn.Sequential(*mid)

    def forward(self, x, need_feature=False):
        downs = []
        for i in range(self.num_downs):
            x = self.downs[i](x)
            downs.append(x)
        x = self.mid(x)
        for i in range(self.num_downs):
            x = torch.cat((x, downs[- i - 1]), dim=1)
            x = self.ups[i](x)
        out = self.tail(x)
        if need_feature:
            return out, x
        return out