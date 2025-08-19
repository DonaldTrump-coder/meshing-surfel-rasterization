from dataclasses import dataclass
from jsonargparse import CLI
import os
import glob
import torch
import TSDF_forGS
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.cameras.add_cameras import visualising_cameras,add_cameras,get_extrinsic,get_intrinsic
from internal.utils.gs2d_mesh_utils import GS2DMeshUtils, post_process_mesh
import open3d as o3d
import numpy as np

def extract_bounding(ply_path:str):
    ply=o3d.io.read_point_cloud(ply_path)
    xmin,ymin,zmin=ply.get_min_bound()
    xmax,ymax,zmax=ply.get_max_bound()
    return np.array([[xmin,ymin,zmin,xmax,ymax,zmax]],dtype=np.float64)

#根据文件夹名搜索ckpt模型文件
def search_ckpt_file(model_folder:str) -> str:
    # if a directory path is provided, auto search checkpoint or ply
    if os.path.isdir(model_folder) is False:
        return None
    # find checkpoint with max iterations
    load_from = None
    previous_checkpoint_iteration = -1
    for i in glob.glob(os.path.join(model_folder, "*.ckpt")):
        try:
            checkpoint_iteration = int(i[i.rfind("=") + 1:i.rfind(".")])
        except Exception as err:
            print("error occurred when parsing iteration from {}: {}".format(i, err))
            continue
        if checkpoint_iteration > previous_checkpoint_iteration:
            previous_checkpoint_iteration = checkpoint_iteration
            load_from = i

    assert load_from is not None, "not a checkpoint can be found"

    return load_from

#定义传进去重建mesh的命令行参数
@dataclass
class CLIArgs:
    model_path: str

    dataset_path: str = None

    voxel_size: float = -1.

    depth_trunc: float = -1.

    sdf_trunc: float = -1.

    num_cluster: int = 50

    unbounded: bool = False

    mesh_res: int = 1024

if __name__ == "__main__":
    args = CLI(CLIArgs)
    device = torch.device("cuda")
    loadable_file = search_ckpt_file(args.model_path)#读取ckpt文件
    print(loadable_file)
    dataparser_config = None
    ckpt = torch.load(loadable_file, map_location="cpu", weights_only=False)

    # initialize model
    model = GaussianModelLoader.initialize_model_from_checkpoint(
        ckpt,
        device=device)
    model.freeze()
    model.pre_activate_all_properties()#用激活函数激活变换模型中的各参数

    # initialize renderer
    renderer = GaussianModelLoader.initialize_renderer_from_checkpoint(
        ckpt,
        stage="validate",
        device=device,
        )
    try:
        dataparser_config = ckpt["datamodule_hyper_parameters"]["parser"]
    except:
        pass

    dataset_path = ckpt["datamodule_hyper_parameters"]["path"]
    if args.dataset_path is not None:
        dataset_path = args.dataset_path
    if dataparser_config is None:
        from internal.dataparsers.colmap_dataparser import Colmap
        dataparser_config = Colmap()
    dataparser_outputs = dataparser_config.instantiate(#用数据路径等参数初始化
        path=dataset_path,
        output_path=os.getcwd(),
        global_rank=0,
    ).get_outputs()#处理好相片、深度对齐的数据
    cameras = [i.to_device(device) for i in dataparser_outputs.train_set.cameras]
    cameras=add_cameras("data/geometry_gt/jinguilou_post/lidar.ply",cameras,device)
    for camera in cameras:
        camera.idx.to(device)
        camera.to_device(device)
    visualising_cameras(cameras,"data/geometry_gt/jinguilou_post/lidar.ply")
    bounding=extract_bounding("data/geometry_gt/jinguilou_post/lidar.ply")

    # set the active_sh to 0 to export only diffuse texture
    model.active_sh_degree = 0#只保留基础球谐函数
    bg_color = torch.zeros((3,), dtype=torch.float, device=device)#背景颜色
    maps = GS2DMeshUtils.render_views(model, renderer, cameras, bg_color)#第一次渲染得到的深度图、rgb图
    rgbs,depths,weights=maps

    name = 'fuse.ply'
    depth_trunc = args.depth_trunc
    voxel_size = args.voxel_size
    sdf_trunc = args.sdf_trunc

    tsdf=TSDF_forGS.TSDF()
    tsdf.addGrids(bounding[0,0],bounding[0,1],bounding[0,2],bounding[0,3],bounding[0,4],bounding[0,5],args.voxel_size,args.sdf_trunc,args.depth_trunc)
    tsdf.TSDF_Integration(get_intrinsic(cameras[0]),get_extrinsic(cameras[0]),depths[0][0],weights[0])
    points,triangles=tsdf.extract_mesh()
    mesh=o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    o3d.io.write_triangle_mesh(name, mesh)