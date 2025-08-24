#include "TSDF.h"

glm::mat3 np_to_glm3(pybind11::array_t<double> arr);
glm::mat4x3 np_to_glm4x3(pybind11::array_t<double> arr);
float* np_to_array1d(pybind11::array_t<float> arr);

namespace py = pybind11;

PYBIND11_MODULE(_C,m)
{
    m.doc() = "TSDF in C++";
    
    py::class_<TSDF>(m,"TSDF")
    .def(py::init<>())
    .def("addGrids",
         &TSDF::addGrids,
         py::arg("xmin"),
         py::arg("ymin"),
         py::arg("zmin"),
         py::arg("xmax"),
         py::arg("ymax"),
         py::arg("zmax"),
         py::arg("voxel_size"),
         py::arg("sdf_trunc"),
         py::arg("depth_trunc")
    )
    .def("TSDF_Integration",
         [](TSDF &tsdf,
            py::array_t<double> K_np,
            py::array_t<double> Rt_np,
            py::array_t<float> depth_np,
            py::array_t<float> weight_np,
            int width,
            int height)
        {
            glm::mat3 K=np_to_glm3(K_np);
            glm::mat4x3 Rt = np_to_glm4x3(Rt_np);
            
            float* depth=np_to_array1d(depth_np);
            float* weight=np_to_array1d(weight_np);

            tsdf.TSDF_Integration(K, Rt, depth, weight, width, height);
        },
         py::arg("K"),
         py::arg("Rt"),
         py::arg("depth_map"),
         py::arg("weight_map"),
         py::arg("width"),
         py::arg("height")
        )
    .def("Marching_Cubes",&TSDF::Marching_Cubes)
    .def("getPoints",&TSDF::getPoints)
    .def("getTriangles", &TSDF::getTriangles);
}

glm::mat3 np_to_glm3(pybind11::array_t<double> arr) 
{
    auto buf = arr.request();
    if (buf.size != 9) throw std::runtime_error("Expected 3x3 array");
    double* ptr = (double*)buf.ptr;
    return glm::mat3(
        ptr[0], ptr[3], ptr[6],
        ptr[1], ptr[4], ptr[7],
        ptr[2], ptr[5], ptr[8]
    );
}

glm::mat4x3 np_to_glm4x3(pybind11::array_t<double> arr)
{
    auto buf = arr.request();
    if (buf.size != 12) throw std::runtime_error("Expected 3x4 array");
    double* ptr = (double*)buf.ptr;
    return glm::mat4x3(
        ptr[0], ptr[4], ptr[8],
        ptr[1], ptr[5], ptr[9],
        ptr[2], ptr[6], ptr[10],
        ptr[3], ptr[7], ptr[11]
    );
}

float* np_to_array1d(pybind11::array_t<float> arr)
{
    auto buf = arr.request();
    if (buf.ndim != 2) throw std::runtime_error("Expected 2D array");

    // 直接返回 NumPy 内部连续内存指针
    return static_cast<float*>(buf.ptr);
}