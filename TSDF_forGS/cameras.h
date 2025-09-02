#include <glm/glm.hpp>

glm::vec2 projectPointToPixel(const float x, const float y, const float z, const glm::mat4x3& P);
//project a 3D point to pixel

float get_value(const float* image, float u, float v, int width, int height);
//get the corresponding value of (u,v) on the image

void Linear_Interp(float& x, float& y, float& z, const float& x1, const float& y1, const float& z1, const float& x2, const float& y2, const float& z2, const float& value1, const float& value2);

float get_dist(const float x1, const float y1, const float z1, const float x2, const float y2, const float z2);

float dot(glm::vec3 v1, glm::vec3 v2);