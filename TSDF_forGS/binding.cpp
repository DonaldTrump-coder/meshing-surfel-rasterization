#include "TSDF.h"

glm::mat3 np_to_glm3(pybind11::array_t<double> arr);
glm::mat4x3 np_to_glm4x3(pybind11::array_t<double> arr);
double** np_to_array2d(pybind11::array_t<double> arr);

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
            py::array_t<double> depth_np,
            py::array_t<double> weight_np,
            int width,
            int height)
        {
            glm::mat3 K=np_to_glm3(K_np);
            glm::mat4x3 Rt = np_to_glm4x3(Rt_np);
            double** depth_map = np_to_array2d(depth_np);
            double** weight_map = np_to_array2d(weight_np);
            tsdf.TSDF_Integration(K, Rt, const_cast<const double**>(depth_map), const_cast<const double**>(weight_map), width, height);
            delete[] depth_map;
            delete[] weight_map;
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

double** np_to_array2d(pybind11::array_t<double> arr)
{
    auto buf = arr.request();
    if (buf.ndim != 2) throw std::runtime_error("Expected 2D array");

    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];

    double* data = static_cast<double*>(buf.ptr);
    double** cpp_array = new double*[rows];

    for (size_t i = 0; i < rows; i++)
    {
        cpp_array[i] = data + i * cols;
    }

    return cpp_array;
}