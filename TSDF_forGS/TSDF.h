#include "cameras.h"
#include <algorithm>
#include <vector>
#include <cstddef>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

//Storing TSDF directly in the Vertices
struct Vertex
{
    double tsdf = 0;
    double weight = 0;
    bool seen = 0;//Whether been seen by the cameras
    double x,y,z;
};
//顶点结构体，存储两个变量  Storing the TSDF and the weight

struct Line
{
    double starting_x;
    double starting_y;
    double starting_z;

    double ending_x;
    double ending_y;
    double ending_z;
    bool added=0;
};
//a line of a triangle mesh

struct Voxel
{
    Vertex* vert1;
    Vertex* vert2;
    Vertex* vert3;
    Vertex* vert4;

    Vertex* vert5;
    Vertex* vert6;
    Vertex* vert7;
    Vertex* vert8;
    std::vector<Line*> lines;//storing the pointers to lines of the voxel
};

struct Plane
{
    Vertex* vert1;
    Vertex* vert2;
    Vertex* vert3;
    Vertex* vert4;
};

struct Point
{
    double x,y,z;
    size_t index;
};

struct Triangle
{
    size_t v1,v2,v3;//Indexs of the vertices
};

class Grids
{
private:
    double voxel_size;
    double xmin, ymin, zmin;
    Vertex* vertices;//索引所有顶点  index all the vertices
    double sdf_trunc;
    double depth_trunc;
    double back_sdf_trunc;
public:
    int x_length;//格网x方向体素数(不是顶点数)
    int y_length;//格网y方向体素数
    int z_length;//格网z方向体素数
    Grids(double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, double voxel_size);//Initialize boundaries and size
    ~Grids();//clear the storage
    Vertex* get_vertex(int i, int j, int k);
    void Set_Param(double sdf_trunc,//max sdf
                   double depth_trunc//max depth
                );//set other parameters
    void TSDF_Integration(const glm::mat3 K, const glm::mat4x3 Rt, double* depth_map, double* weight_map, int width, int height);
    void setVoxel(Voxel& voxel, int i, int j, int k);//Set the vertices of a voxel
    void get_Voxel_Planes(Voxel& voxel, Plane& front, Plane& back, Plane& left, Plane& right, Plane& bottom, Plane& top);
    void add_Plane_Lines(std::vector<Line*>& lines, Plane plane);//match and add the lines of a plane to the lines of the voxel
    void Searching_for_Triangles(std::vector<Point>& points, std::vector<Triangle>& triangles, std::vector<Line*>& lines);//find the triangles of the voxel, and add them to the vector
    void clear_Voxel(Voxel& voxel);
    bool seen(Voxel& voxel);
};
//格网类

class TSDF
{
    private:
        Grids** grids=NULL;
        int grids_num=0;

        //Used for marching cubes
        Voxel vox;
        Plane front;
        Plane back;
        Plane left;
        Plane right;
        Plane bottom;
        Plane top;

        std::vector<Point> points;
        std::vector<Triangle> triangles;
    public:
        void addGrids(double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, double voxel_size, double sdf_trunc, double depth_trunc);
        void TSDF_Integration(const glm::mat3 K, const glm::mat4x3 Rt, double* depth_map, double* weight_map, int width, int height);
        void Marching_Cubes();
        void clearGrids();
        py::array_t<double> getPoints();
        py::array_t<int> getTriangles();
};
//manage all the grids and mesh extraction