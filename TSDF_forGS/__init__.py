from . import _C
import numpy as np

__version__="0.0.6" # 手写TSDF的版本

class TSDF:
    def __init__(self):
        self.tsdf=_C.TSDF()
    
    def addGrids(self, xmin, ymin, zmin, xmax, ymax, zmax, voxel_size, sdf_trunc, depth_trunc):
        self.tsdf.addGrids(xmin,ymin,zmin,xmax,ymax,zmax,voxel_size,sdf_trunc,depth_trunc)

    def TSDF_Integration(self, K: np.ndarray, Rt: np.ndarray,red_map:np.ndarray, green_map:np.ndarray, blue_map:np.ndarray, depth_map: np.ndarray, weight_map: np.ndarray):

        height, width = weight_map.shape
        self.tsdf.TSDF_Integration(K, Rt, red_map, green_map, blue_map, depth_map, weight_map, width, height)

    def Gaussian_Integration(self, means: np.ndarray, sh: np.ndarray, normal: np.ndarray, u: np.ndarray, v: np.ndarray, scale: np.ndarray, opacity: float):
        self.tsdf.Gaussian_Integration(means, sh, normal, u, v, scale, opacity)

    def extract_mesh(self):
        self.tsdf.Marching_Cubes()
        points=self.tsdf.getPoints()
        triangles = self.tsdf.getTriangles()
        colors=self.tsdf.getColors()
        return points, triangles, colors