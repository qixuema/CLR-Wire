import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
from src.pointnet2.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule

def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(lr_sched.LambdaLR):
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(type(model)._name_)
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def state_dict(self):
        return dict(last_epoch=self.last_epoch)

    def load_state_dict(self, state):
        self.last_epoch = state["last_epoch"]
        self.step(self.last_epoch)


lr_clip = 1e-5
bnm_clip = 1e-2


class PointNet2ClassificationSSG_WOPL(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[3, 64, 64, 128],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024], use_xyz=self.hparams["model.use_xyz"]
            )
        )
    
        # self.fc_layer = nn.Sequential(
        #     nn.Linear(1024, 512, bias=False),
        #     nn.LayerNorm(512),
        #     nn.SiLU(True),
        #     nn.Linear(512, 256, bias=False),
        #     nn.LayerNorm(256),
        #     nn.SiLU(True),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 768),
        # )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 1024, bias=False),
            nn.LayerNorm(1024),
            nn.SiLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 768),
        )        

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))