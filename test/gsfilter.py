import torch
from internal.utils.gaussian_model_loader import GaussianModelLoader

device = torch.device("cuda")
ckpt_file = "C:\\Users\\10527\\Desktop\\Research of 2DGS\\data\\epoch=28-step=1000.ckpt"
print(ckpt_file)
ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
model = GaussianModelLoader.initialize_model_from_checkpoint(
        ckpt,
        device=device)
model.freeze()
model.pre_activate_all_properties()#用激活函数激活变换模型中的各参数
means = model.gaussians["means"].detach().cpu().numpy()
opacities = model.gaussians["opacities"].detach().cpu().numpy()
scales = model.gaussians["scales"].detach().cpu().numpy()
rotations = model.gaussians["rotations"].detach().cpu().numpy()
shs = model.gaussians["shs"].detach().cpu().numpy()