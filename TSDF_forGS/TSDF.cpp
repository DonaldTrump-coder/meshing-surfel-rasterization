#include "TSDF.h"

Grids::Grids(double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, double voxel_size)
{
    this->voxel_size = voxel_size;
    double dx=xmax-xmin;
    double dy=ymax-ymin;
    double dz=zmax-zmin;
    x_length=(int)(dx/voxel_size)+1;
    y_length=(int)(dy/voxel_size)+1;
    z_length=(int)(dz/voxel_size)+1;//确定格网各方向的格网个数
    this->xmin=xmin;
    this->ymin=ymin;
    this->zmin=zmin;
    vertices=new Vertex[(x_length+1)*(y_length+1)*(z_length+1)];
    //index vertices[i,j,k]: vertices[i*(y_length+1)*(z_length+1) + j*(z_length+1) + k]
    for(int i=0;i<x_length+1;i++)
    {
        for(int j=0;j<y_length+1;j++)
        {
            for(int k=0;k<z_length+1;k++)
            {
                get_vertex(i,j,k)->x=xmin+i*voxel_size;
                get_vertex(i,j,k)->y=ymin+j*voxel_size;
                get_vertex(i,j,k)->z=zmin=k*voxel_size;
            }
        }
    }
}

Grids::~Grids()
{
    delete[] vertices;
}

Vertex* Grids::get_vertex(int i, int j, int k)
{
    return &vertices[i*(y_length+1)*(z_length+1) + j*(z_length+1) + k];
}

void Grids::Set_Param(double sdf_trunc, double depth_trunc)
{
    this->sdf_trunc=sdf_trunc;
    this->depth_trunc=depth_trunc;
}

void Grids::TSDF_Integration(const glm::mat3 K, //Inner Matrix of camera(3×3)
                             const glm::mat4x3 Rt,//Outer Matrix of camera(3×4)
                             double* depth_map, //depth map of camera
                             double* weight_map, // weight map of camera
                             int width,
                             int height
                            )
{
    for(int i=0;i<x_length+1;i++)
    {
        for(int j=0;j<y_length+1;j++)
        {
            for(int k=0;k<z_length+1;k++)
            {
                Vertex* v=get_vertex(i,j,k);
                const glm::vec3 vertex_p(xmin+i*voxel_size, ymin+j*voxel_size, zmin+k*voxel_size);
                glm::vec2 uv=projectPointToPixel(vertex_p, K*Rt);
                if(uv.x<0||uv.x>(width-1)||uv.y<0||uv.y>(height-1))
                {
                    continue;// if not in the image, pass the vertex
                }
                glm::vec3 camera_P=Rt*glm::vec4(vertex_p, 1.0);
                if(camera_P.z<0)
                {
                    continue;
                }
                double depth=get_value(depth_map,uv.x,uv.y,width,height);
                double weight=get_value(weight_map,uv.x,uv.y,width,height);
                double sdf=depth-camera_P.z;
                double tsdf=std::clamp(sdf/sdf_trunc, -1.0, 1.0);
                v->tsdf = (v->weight * v->tsdf + tsdf*weight) / (v->weight + weight);
                v->weight += weight;
                v->seen=1;
            }
        }
    }
}

void Grids::setVoxel(Voxel& voxel,//the voxel to be set vertices
                     int i,//the index of the voxel
                     int j,
                     int k
                    )
{
    //Set the vertices of a voxel
    voxel.vert1=get_vertex(i,j,k);
    voxel.vert2=get_vertex(i+1,j,k);
    voxel.vert3=get_vertex(i,j,k+1);
    voxel.vert4=get_vertex(i+1,j,k+1);
    voxel.vert5=get_vertex(i,j+1,k);
    voxel.vert6=get_vertex(i+1,j+1,k);
    voxel.vert7=get_vertex(i,j+1,k+1);
    voxel.vert8=get_vertex(i+1,j+1,k+1);
}

void Grids::get_Voxel_Planes(Voxel& voxel, Plane& front, Plane& back, Plane& left, Plane& right, Plane& bottom, Plane& top)
{
    front.vert1=voxel.vert1;
    front.vert2=voxel.vert2;
    front.vert3=voxel.vert3;
    front.vert4=voxel.vert4;

    back.vert1=voxel.vert6;
    back.vert2=voxel.vert5;
    back.vert3=voxel.vert8;
    back.vert4=voxel.vert7;

    left.vert1=voxel.vert5;
    left.vert2=voxel.vert1;
    left.vert3=voxel.vert7;
    left.vert4=voxel.vert3;

    right.vert1=voxel.vert2;
    right.vert2=voxel.vert6;
    right.vert3=voxel.vert4;
    right.vert4=voxel.vert8;

    bottom.vert1=voxel.vert1;
    bottom.vert2=voxel.vert2;
    bottom.vert3=voxel.vert5;
    bottom.vert4=voxel.vert6;

    top.vert1=voxel.vert3;
    top.vert2=voxel.vert4;
    top.vert3=voxel.vert7;
    top.vert4=voxel.vert8;
}

void Grids::add_Plane_Lines(std::vector<Line*>& lines, Plane plane)
{
    if(plane.vert1->tsdf>=0 && plane.vert2->tsdf>=0 && plane.vert3->tsdf>=0 && plane.vert4->tsdf>=0)
    {
        return;
    }
    if(plane.vert1->tsdf<0 && plane.vert2->tsdf<0 && plane.vert3->tsdf<0 && plane.vert4->tsdf<0)
    {
        return;
    }

    if((plane.vert1->tsdf>=0 && plane.vert2->tsdf<0 && plane.vert3->tsdf<0 && plane.vert4->tsdf<0) || (plane.vert1->tsdf<0 && plane.vert2->tsdf>=0 && plane.vert3->tsdf>=0 && plane.vert4->tsdf>=0))
    {
        Line* line1=new Line;
        Linear_Interp(line1->starting_x,
                      line1->starting_y,
                      line1->starting_z,
                      plane.vert1->x,
                      plane.vert1->y,
                      plane.vert1->z,
                      plane.vert3->x,
                      plane.vert3->y,
                      plane.vert3->z,
                      plane.vert1->tsdf,
                      plane.vert3->tsdf);
        Linear_Interp(line1->ending_x,
                      line1->ending_y,
                      line1->ending_z,
                      plane.vert1->x,
                      plane.vert1->y,
                      plane.vert1->z,
                      plane.vert2->x,
                      plane.vert2->y,
                      plane.vert2->z,
                      plane.vert1->tsdf,
                      plane.vert2->tsdf);
        lines.push_back(line1);
        return;
    }

    if((plane.vert3->tsdf>=0 && plane.vert2->tsdf<0 && plane.vert1->tsdf<0 && plane.vert4->tsdf<0) || (plane.vert3->tsdf<0 && plane.vert2->tsdf>=0 && plane.vert1->tsdf>=0 && plane.vert4->tsdf>=0))
    {
        Line* line1=new Line;
        Linear_Interp(line1->starting_x,
                      line1->starting_y,
                      line1->starting_z,
                      plane.vert1->x,
                      plane.vert1->y,
                      plane.vert1->z,
                      plane.vert3->x,
                      plane.vert3->y,
                      plane.vert3->z,
                      plane.vert1->tsdf,
                      plane.vert3->tsdf);
        Linear_Interp(line1->ending_x,
                      line1->ending_y,
                      line1->ending_z,
                      plane.vert3->x,
                      plane.vert3->y,
                      plane.vert3->z,
                      plane.vert4->x,
                      plane.vert4->y,
                      plane.vert4->z,
                      plane.vert3->tsdf,
                      plane.vert4->tsdf);
        lines.push_back(line1);
        return;
    }

    if((plane.vert4->tsdf>=0 && plane.vert2->tsdf<0 && plane.vert1->tsdf<0 && plane.vert3->tsdf<0) || (plane.vert4->tsdf<0 && plane.vert2->tsdf>=0 && plane.vert1->tsdf>=0 && plane.vert3->tsdf>=0))
    {
        Line* line1=new Line;
        Linear_Interp(line1->starting_x,
                      line1->starting_y,
                      line1->starting_z,
                      plane.vert3->x,
                      plane.vert3->y,
                      plane.vert3->z,
                      plane.vert4->x,
                      plane.vert4->y,
                      plane.vert4->z,
                      plane.vert3->tsdf,
                      plane.vert4->tsdf);
        Linear_Interp(line1->ending_x,
                      line1->ending_y,
                      line1->ending_z,
                      plane.vert2->x,
                      plane.vert2->y,
                      plane.vert2->z,
                      plane.vert4->x,
                      plane.vert4->y,
                      plane.vert4->z,
                      plane.vert2->tsdf,
                      plane.vert4->tsdf);
        lines.push_back(line1);
        return;
    }

    if((plane.vert2->tsdf>=0 && plane.vert4->tsdf<0 && plane.vert1->tsdf<0 && plane.vert3->tsdf<0) || (plane.vert2->tsdf<0 && plane.vert4->tsdf>=0 && plane.vert1->tsdf>=0 && plane.vert3->tsdf>=0))
    {
        Line* line1=new Line;
        Linear_Interp(line1->starting_x,
                      line1->starting_y,
                      line1->starting_z,
                      plane.vert1->x,
                      plane.vert1->y,
                      plane.vert1->z,
                      plane.vert2->x,
                      plane.vert2->y,
                      plane.vert2->z,
                      plane.vert1->tsdf,
                      plane.vert2->tsdf);
        Linear_Interp(line1->ending_x,
                      line1->ending_y,
                      line1->ending_z,
                      plane.vert2->x,
                      plane.vert2->y,
                      plane.vert2->z,
                      plane.vert4->x,
                      plane.vert4->y,
                      plane.vert4->z,
                      plane.vert2->tsdf,
                      plane.vert4->tsdf);
        lines.push_back(line1);
        return;
    }

    if((plane.vert1->tsdf>=0 && plane.vert4->tsdf>=0 && plane.vert2->tsdf<0 && plane.vert3->tsdf<0) || (plane.vert1->tsdf<0 && plane.vert4->tsdf<0 && plane.vert2->tsdf>=0 && plane.vert3->tsdf>=0))
    {
        Line* line1=new Line;
        Line* line2=new Line;
        Linear_Interp(line1->starting_x,
                      line1->starting_y,
                      line1->starting_z,
                      plane.vert1->x,
                      plane.vert1->y,
                      plane.vert1->z,
                      plane.vert3->x,
                      plane.vert3->y,
                      plane.vert3->z,
                      plane.vert1->tsdf,
                      plane.vert3->tsdf);
        Linear_Interp(line1->ending_x,
                      line1->ending_y,
                      line1->ending_z,
                      plane.vert1->x,
                      plane.vert1->y,
                      plane.vert1->z,
                      plane.vert2->x,
                      plane.vert2->y,
                      plane.vert2->z,
                      plane.vert1->tsdf,
                      plane.vert2->tsdf);
        lines.push_back(line1);
        Linear_Interp(line2->starting_x,
                      line2->starting_y,
                      line2->starting_z,
                      plane.vert4->x,
                      plane.vert4->y,
                      plane.vert4->z,
                      plane.vert3->x,
                      plane.vert3->y,
                      plane.vert3->z,
                      plane.vert4->tsdf,
                      plane.vert3->tsdf);
        Linear_Interp(line2->ending_x,
                      line2->ending_y,
                      line2->ending_z,
                      plane.vert4->x,
                      plane.vert4->y,
                      plane.vert4->z,
                      plane.vert2->x,
                      plane.vert2->y,
                      plane.vert2->z,
                      plane.vert4->tsdf,
                      plane.vert2->tsdf);
        lines.push_back(line2);
        return;
    }

    if((plane.vert1->tsdf>=0 && plane.vert3->tsdf>=0 && plane.vert2->tsdf<0 && plane.vert4->tsdf<0) || (plane.vert1->tsdf<0 && plane.vert3->tsdf<0 && plane.vert2->tsdf>=0 && plane.vert4->tsdf>=0))
    {
        Line* line1=new Line;
        Linear_Interp(line1->starting_x,
                      line1->starting_y,
                      line1->starting_z,
                      plane.vert4->x,
                      plane.vert4->y,
                      plane.vert4->z,
                      plane.vert3->x,
                      plane.vert3->y,
                      plane.vert3->z,
                      plane.vert4->tsdf,
                      plane.vert3->tsdf);
        Linear_Interp(line1->ending_x,
                      line1->ending_y,
                      line1->ending_z,
                      plane.vert1->x,
                      plane.vert1->y,
                      plane.vert1->z,
                      plane.vert2->x,
                      plane.vert2->y,
                      plane.vert2->z,
                      plane.vert1->tsdf,
                      plane.vert2->tsdf);
        lines.push_back(line1);
        return;
    }

    if((plane.vert3->tsdf>=0 && plane.vert4->tsdf>=0 && plane.vert1->tsdf<0 && plane.vert2->tsdf<0) || (plane.vert3->tsdf<0 && plane.vert4->tsdf<0 && plane.vert1->tsdf>=0 && plane.vert2->tsdf>=0))
    {
        Line* line1=new Line;
        Linear_Interp(line1->starting_x,
                      line1->starting_y,
                      line1->starting_z,
                      plane.vert1->x,
                      plane.vert1->y,
                      plane.vert1->z,
                      plane.vert3->x,
                      plane.vert3->y,
                      plane.vert3->z,
                      plane.vert1->tsdf,
                      plane.vert3->tsdf);
        Linear_Interp(line1->ending_x,
                      line1->ending_y,
                      line1->ending_z,
                      plane.vert4->x,
                      plane.vert4->y,
                      plane.vert4->z,
                      plane.vert2->x,
                      plane.vert2->y,
                      plane.vert2->z,
                      plane.vert4->tsdf,
                      plane.vert2->tsdf);
        lines.push_back(line1);
        return;
    }
}

void Grids::Searching_for_Triangles(std::vector<Point>& points, std::vector<Triangle>& triangles, std::vector<Line*>& lines)
{
    size_t lines_num=lines.size();
    size_t points_num=points.size();
    if(lines_num==0)
        return;
    std::vector<Point> temp_points;
    bool finished=0;
    Point starting;
    Point ending;
    while(finished==0)
    {
        temp_points.clear();
        for(size_t i=0;i<lines_num;i++)
        {
            if(lines[i]->added==0)
            {
                starting.x=lines[i]->starting_x;
                starting.y=lines[i]->starting_y;
                starting.z=lines[i]->starting_z;
                starting.index=points_num;
                points_num++;
                ending.x=lines[i]->ending_x;
                ending.y=lines[i]->ending_y;
                ending.z=lines[i]->ending_z;
                ending.index=points_num;
                points_num++;

                points.push_back(starting);
                points.push_back(ending);
                temp_points.push_back(starting);
                temp_points.push_back(ending);
                lines[i]->added=1;

                break;
            }
        }
        bool matched = 0;
        while(matched==0)
        {
            bool extended=false;//whether find the next line
            for(size_t i=0;i<lines_num;i++)
            {
                if(lines[i]->added==1)
                    continue;

                if(get_dist(ending.x, ending.y, ending.z, lines[i]->starting_x, lines[i]->starting_y, lines[i]->starting_z)<0.000001)
                {
                    ending.x=lines[i]->ending_x;
                    ending.y=lines[i]->ending_y;
                    ending.z=lines[i]->ending_z;
                    if(get_dist(ending.x,ending.y,ending.z,starting.x,starting.y,starting.z)<0.000001)
                    {
                        extended=true;
                        matched=1;
                        break;
                    }
                    ending.index=points_num;
                    points_num++;
                    points.push_back(ending);
                    temp_points.push_back(ending);
                    lines[i]->added=1;
                    extended=true;
                    break;
                }

                if(get_dist(ending.x, ending.y, ending.z, lines[i]->ending_x, lines[i]->ending_y, lines[i]->ending_z)<0.000001)
                {
                    ending.x=lines[i]->starting_x;
                    ending.y=lines[i]->starting_y;
                    ending.z=lines[i]->starting_z;
                    if(get_dist(ending.x,ending.y,ending.z,starting.x,starting.y,starting.z)<0.000001)
                    {
                        extended=true;
                        matched=1;
                        break;
                    }
                    ending.index=points_num;
                    points_num++;
                    points.push_back(ending);
                    temp_points.push_back(ending);
                    lines[i]->added=1;
                    extended=true;
                    break;
                }
            }
            if(!extended)
                break;
        }
        size_t temp_points_num=temp_points.size();
        if(temp_points_num>3)
        {
            for(size_t i=0;i<temp_points_num-2;i++)
            {
                triangles.push_back(Triangle{temp_points[0].index,temp_points[i+1].index,temp_points[i+2].index});
            }
        }
        else if(temp_points_num==3)
        {
            triangles.push_back(Triangle{temp_points[0].index,temp_points[1].index,temp_points[2].index});
        }

        finished=1;
        for(size_t i=0;i<lines_num;i++)
        {
            if(lines[i]->added==0)
            {
                finished=0;
                break;
            }
        }
    }
}

void Grids::clear_Voxel(Voxel& voxel)
{
    for(size_t i=0;i<voxel.lines.size();i++)
    {
        delete voxel.lines[i];
    }
    voxel.lines.clear();
}

bool Grids::seen(Voxel& voxel)
{
    if(voxel.vert1->seen==0)
        return 0;
    if(voxel.vert2->seen==0)
        return 0;
    if(voxel.vert3->seen==0)
        return 0;
    if(voxel.vert4->seen==0)
        return 0;
    if(voxel.vert5->seen==0)
        return 0;
    if(voxel.vert6->seen==0)
        return 0;
    if(voxel.vert7->seen==0)
        return 0;
    if(voxel.vert8->seen==0)
        return 0;
    return 1;
}

void TSDF::addGrids(double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, double voxel_size, double sdf_trunc, double depth_trunc)
{
    if(grids==NULL)
    {
        Grids* new_g=new Grids(xmin, ymin, zmin, xmax, ymax, zmax, voxel_size);
        new_g->Set_Param(sdf_trunc,depth_trunc);
        grids=new Grids*[1];
        grids[0]=new_g;
        grids_num++;
    }
    else
    {

    }
}

void TSDF::TSDF_Integration(const glm::mat3 K, const glm::mat4x3 Rt, double* depth_map, double* weight_map, int width, int height)
{
    for(int i=0;i<grids_num;i++)
    {
        grids[i]->TSDF_Integration(K,Rt,depth_map,weight_map,width,height);
    }
}

void TSDF::Marching_Cubes()
{
    for(int n=0;n<grids_num;n++)
    {
        for(int i=0;i<grids[n]->x_length;i++)
        {
            for(int j=0;j<grids[n]->y_length;j++)
            {
                for(int k=0;k<grids[n]->z_length;k++)
                {
                    //go through every voxel grid
                    grids[n]->setVoxel(vox,i,j,k);
                    if(grids[n]->seen(vox)==0)
                        continue;
                    grids[n]->get_Voxel_Planes(vox,front,back,left,right,bottom,top);

                    //add lines to a voxel grid
                    grids[n]->add_Plane_Lines(vox.lines,front);
                    grids[n]->add_Plane_Lines(vox.lines,back);
                    grids[n]->add_Plane_Lines(vox.lines,left);
                    grids[n]->add_Plane_Lines(vox.lines,right);
                    grids[n]->add_Plane_Lines(vox.lines,bottom);
                    grids[n]->add_Plane_Lines(vox.lines,top);
                    grids[n]->Searching_for_Triangles(points,triangles,vox.lines);
                    grids[n]->clear_Voxel(vox);
                }
            }
        }
    }
    clearGrids();//Clear the storage
}

void TSDF::clearGrids()
{
    for(int i=0;i<grids_num;i++)
    {
        delete grids[i];
    }
    delete[] grids;
}

py::array_t<double> TSDF::getPoints()
{
    std::vector<std::ptrdiff_t> shape = { static_cast<std::ptrdiff_t>(points.size()), 3 };
    py::array_t<double> arr(shape);
    auto buf = arr.mutable_unchecked<2>();
    for (size_t i=0; i<points.size(); i++) 
    {
        buf(i,0) = points[i].x;
        buf(i,1) = points[i].y;
        buf(i,2) = points[i].z;
    }
    return arr;
}

py::array_t<int> TSDF::getTriangles() 
{
    std::vector<std::ptrdiff_t> shape = { static_cast<std::ptrdiff_t>(triangles.size()), 3 };
    py::array_t<int> arr(shape);
    auto buf = arr.mutable_unchecked<2>();
    for (size_t i=0; i<triangles.size(); i++)
    {
        buf(i,0) = static_cast<int>(triangles[i].v1);
        buf(i,1) = static_cast<int>(triangles[i].v2);
        buf(i,2) = static_cast<int>(triangles[i].v3);
    }
    return arr;
}