#include <glm/glm.hpp>

glm::vec2 projectPointToPixel(const glm::vec3& P_w, const glm::mat4x3& P);
//project a 3D point to pixel

double get_value(const double** image, double u, double v);
//get the corresponding value of (u,v) on the image

void Linear_Interp(double& x, double& y, double& z, const double& x1, const double& y1, const double& z1, const double& x2, const double& y2, const double& z2, const double& value1, const double& value2);

double get_dist(const double x1, const double y1, const double z1, const double x2, const double y2, const double z2);