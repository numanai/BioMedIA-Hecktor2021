import torch
from torch import nn
from torchmtlr import MTLR

#**Update
#Number of clin_var
n_clin_var = 13

def conv_3d_block (in_c, out_c, act='relu', norm='bn', num_groups=8, *args, **kwargs):
    activations = nn.ModuleDict ([
        ['relu', nn.ReLU(inplace=True)],
        ['lrelu', nn.LeakyReLU(0.1, inplace=True)]
    ])
    
    normalizations = nn.ModuleDict ([
        ['bn', nn.BatchNorm3d(out_c)],
        ['gn', nn.GroupNorm(int(out_c/num_groups), out_c)]
    ])
    
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, *args, **kwargs),
        normalizations[norm],
        activations[act],
    )

def flatten_layers(arr):
    return [i for sub in arr for i in sub]



class Dual_MTLR(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        self.cnn = nn.Sequential(#block 1
                                 conv_3d_block(1, 32, kernel_size=hparams['k1']),
                                 conv_3d_block(32, 64, kernel_size=hparams['k2']),
                                 nn.MaxPool3d(kernel_size=2, stride=2),

                                 #block 2
                                 conv_3d_block(64, 128, kernel_size=hparams['k1']),
                                 conv_3d_block(128, 256, kernel_size=hparams['k2']),
                                 nn.MaxPool3d(kernel_size=2, stride=2),

                                 nn.AdaptiveAvgPool3d(1),

                                 nn.Flatten()                               

                            )

        if hparams['n_dense'] <=0:
            self.mtlr = MTLR(256 + n_clin_var, 14)

        else:
            fc_layers = [[nn.Linear(256 + n_clin_var, 512 * hparams['dense_factor']), 
                          nn.BatchNorm1d(512 * hparams['dense_factor']),
                          nn.ReLU(inplace=True), 
                          nn.Dropout(hparams['dropout'])]]   
            
            if hparams['n_dense'] > 1:    
                fc_layers.extend([[nn.Linear(512 * hparams['dense_factor'], 512 * hparams['dense_factor']),
                                   nn.BatchNorm1d(512 * hparams['dense_factor']),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(hparams['dropout'])] for _ in range(hparams['n_dense'] - 1)])
            
            fc_layers = flatten_layers(fc_layers)
            self.mtlr = nn.Sequential(*fc_layers,
                                      MTLR(512 * hparams['dense_factor'], 14),)


    def forward(self, x):
        
        img, clin_var = x
        cnn = self.cnn(img)

        ftr_concat = torch.cat((cnn, clin_var), dim=1)
        
        return self.mtlr(ftr_concat)


"""
Inspired from the work of
Credits:
@article{
  kim_deep-cr_2020,
	title = {Deep-{CR} {MTLR}: a {Multi}-{Modal} {Approach} for {Cancer} {Survival} {Prediction} with {Competing} {Risks}},
	shorttitle = {Deep-{CR} {MTLR}},
	url = {https://arxiv.org/abs/2012.05765v1},
	language = {en},
	urldate = {2021-03-16},
	author = {Kim, Sejin and Kazmierski, Michal and Haibe-Kains, Benjamin},
	month = dec,
	year = {2020}
}
"""