#include "cameras.h"

glm::vec2 projectPointToPixel(const float x,
                              const float y,
                              const float z,// 3D points coordinate
                              const glm::mat4x3& P// Matrix of the camera
                            )
{
    //glm::mat3x4 P_=glm::transpose(P);
    glm::vec3 Pc_h = P * glm::vec4(x, y, z, 1.0);//摄像机矩阵乘以齐次坐标
    glm::vec2 uv;
    uv.x = Pc_h.x / Pc_h.z;
    uv.y = Pc_h.y / Pc_h.z;
    return uv;
}

float get_value(const float* image, float u, float v, int width, int height)
{
    int u_idx=(int)(u+0.5);
    int v_idx=(int)(v+0.5);
    return image[v_idx * width + u_idx];
}

void Linear_Interp(float& x, float& y, float& z, const float& x1, const float& y1, const float& z1, const float& x2, const float& y2, const float& z2, const float& value1, const float& value2)
{
    float t=-value1/(value2-value1);
    x=x1+(x2-x1)*t;
    y=y1+(y2-y1)*t;
    z=z1+(z2-z1)*t;
    return;
}

float get_dist(const float x1, const float y1, const float z1, const float x2, const float y2, const float z2)
{
    float dx=x2-x1;
    float dy=y2-y1;
    float dz=z2-z1;
    return sqrt(dx*dx+dy*dy+dz*dz);
}

float dot(glm::vec3 v1, glm::vec3 v2)
{
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}