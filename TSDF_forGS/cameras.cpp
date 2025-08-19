#include "cameras.h"

glm::vec2 projectPointToPixel(const glm::vec3& P_w,//3D points coordinate
                              const glm::mat4x3& P//Matrix of the camera
                            )
{
    glm::vec3 Pc_h = P * glm::vec4(P_w, 1.0);//摄像机矩阵乘以齐次坐标
    glm::vec2 uv;
    uv.x = Pc_h.x / Pc_h.z;
    uv.y = Pc_h.y / Pc_h.z;
    return uv;
}

double get_value(const double** image, double u, double v)
{
    int u_idx=(int)(u+0.5);
    int v_idx=(int)(v+0.5);
    return image[u_idx][v_idx];
}

void Linear_Interp(double& x, double& y, double& z, const double& x1, const double& y1, const double& z1, const double& x2, const double& y2, const double& z2, const double& value1, const double& value2)
{
    double t=-value1/(value2-value1);
    x=x1+(x2-x1)*t;
    y=y1+(y2-y1)*t;
    z=z1+(z2-z1)*t;
    return;
}

double get_dist(const double x1, const double y1, const double z1, const double x2, const double y2, const double z2)
{
    double dx=x2-x1;
    double dy=y2-y1;
    double dz=z2-z1;
    return sqrt(dx*dx+dy*dy+dz*dz);
}