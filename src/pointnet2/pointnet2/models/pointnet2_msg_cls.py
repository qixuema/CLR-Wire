import torch.nn as nn
from src.pointnet2.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG

from src.pointnet2.pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG_WOPL


class PointNet2ClassificationMSG_WOPL(PointNet2ClassificationSSG_WOPL):
    def _build_model(self):
        super()._build_model()

        self.SA_modules = nn.ModuleList()
        c_in = 3
        with_bn = False
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 32], [c_in, 64]],
                use_xyz=self.hparams["model.use_xyz"],
                bn=with_bn,
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[
                    [c_in, 64],
                    [c_in, 128],
                ],
                use_xyz=self.hparams["model.use_xyz"],
                bn=with_bn,
            )
        )
        c_out_1 = 64 + 128
        
        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[
                    [c_in, 128],
                    [c_in, 256],
                ],
                use_xyz=self.hparams["model.use_xyz"],
                bn=with_bn,
            )
        )       
        c_out_2 = 128 + 256

        c_in = c_out_2        
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=16,
                radii=[0.4, 0.8],
                nsamples=[16, 32],
                mlps=[
                    [c_in, 512],
                    [c_in, 512],
                ],
                use_xyz=self.hparams["model.use_xyz"],
                bn=with_bn,
            )
        )
                
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[1024, 1024],
                use_xyz=self.hparams["model.use_xyz"],
                bn=with_bn,
            )
        )