#include "cameras.h"
#include <algorithm>
#include <vector>
#include <cstddef>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

const float SH_C0 = 0.28209479177387814f;

//Storing TSDF directly in the Vertices
struct Vertex
{
    float tsdf = 0;
    float weight = 0;
    bool seen = 0;//Whether been seen by the cameras
    float x,y,z;
    float R=0;
    float G=0;
    float B=0;
};
//顶点结构体，存储两个变量  Storing the TSDF and the weight

struct Line
{
    float starting_x;
    float starting_y;
    float starting_z;
    float R;
    float G;
    float B;

    float ending_x;
    float ending_y;
    float ending_z;
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
    float x,y,z;
    size_t index;
};

struct Color
{
    float R;
    float G;
    float B;
    Color(float rr, float gg, float bb): R(rr), G(gg), B(bb) {} 
};

struct Triangle
{
    size_t v1,v2,v3;//Indexs of the vertices
};

struct Gaussian //Parameters of a 2D Gaussian
{
    glm::vec3 means;
    float R;
    float G;
    float B;
    glm::vec3 normal;
    glm::vec3 u;
    glm::vec3 v;
    glm::vec2 scale;
    float opacity;
};

class Grids
{
private:
    float voxel_size;
    float xmin, ymin, zmin;
    Vertex* vertices;//索引所有顶点  index all the vertices
    float sdf_trunc;
    float depth_trunc;
    float back_sdf_trunc;
public:
    int x_length;//格网x方向体素数(不是顶点数)
    int y_length;//格网y方向体素数
    int z_length;//格网z方向体素数
    Grids(float xmin, float ymin, float zmin, float xmax, float ymax, float zmax, float voxel_size);//Initialize boundaries and size
    ~Grids();//clear the storage
    Vertex* get_vertex(int i, int j, int k);
    void Set_Param(float sdf_trunc,//max sdf
                   float depth_trunc//max depth
                );//set other parameters
    void TSDF_Integration(const glm::mat3 K, const glm::mat4x3 Rt, float* red_map, float* green_map, float* blue_map, float* depth_map, float* weight_map, int width, int height);
    void Gaussian_Integration(Gaussian& gs);
    void setVoxel(Voxel& voxel, int i, int j, int k);//Set the vertices of a voxel
    void get_Voxel_Planes(Voxel& voxel, Plane& front, Plane& back, Plane& left, Plane& right, Plane& bottom, Plane& top);
    void add_Plane_Lines(std::vector<Line*>& lines, Plane plane);//match and add the lines of a plane to the lines of the voxel
    void Searching_for_Triangles(std::vector<Point>& points, std::vector<Triangle>& triangles, std::vector<Color>& colors, std::vector<Line*>& lines);//find the triangles of the voxel, and add them to the vector
    void clear_Voxel(Voxel& voxel);
    bool seen(Voxel& voxel);
    bool Vertex_near_Gaus(Vertex* vert, Gaussian& gs);
    bool get_DF_sign(float x, float y, float z, Gaussian& gs);
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

        Gaussian gs;

        std::vector<Point> points;
        std::vector<Triangle> triangles;
        std::vector<Color> colors;
    public:
        void addGrids(float xmin, float ymin, float zmin, float xmax, float ymax, float zmax, float voxel_size, float sdf_trunc, float depth_trunc);
        void TSDF_Integration(const glm::mat3 K, const glm::mat4x3 Rt, float* red_map, float* green_map, float* blue_map, float* depth_map, float* weight_map, int width, int height);
        void Gaussian_Integration(const glm::vec3 means, const glm::vec3 sh, const glm::vec3 normal, const glm::vec3 u, const glm::vec3 v, const glm::vec2 scale, const float opacity);
        void Marching_Cubes();
        void clearGrids();
        py::array_t<float> getPoints();
        py::array_t<float> getColors();
        py::array_t<int> getTriangles();
};
//manage all the grids and mesh extraction