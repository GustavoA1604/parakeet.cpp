// Tiny standalone probe for OpenCL device extensions / FP16 support /
// subgroup support / OpenCL C version. Used by qvac-17997 to triage why
// ggml-opencl's device-init refuses NVIDIA / AMD / etc. on a given box
// without having to install the full clinfo package.
//
//   cc -DCL_TARGET_OPENCL_VERSION=300 scripts/probe-opencl.cpp -lOpenCL -lstdc++ -o /tmp/probe-opencl
//   /tmp/probe-opencl
//
// Not part of the build; standalone tool.

#include <CL/cl.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::string get_str(cl_device_id d, cl_device_info p) {
    size_t n = 0;
    if (clGetDeviceInfo(d, p, 0, nullptr, &n) != CL_SUCCESS) return {};
    std::string s(n, '\0');
    clGetDeviceInfo(d, p, n, s.data(), nullptr);
    if (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

int main() {
    cl_uint nplat = 0;
    clGetPlatformIDs(0, nullptr, &nplat);
    std::vector<cl_platform_id> plats(nplat);
    clGetPlatformIDs(nplat, plats.data(), nullptr);
    std::printf("found %u platform(s)\n", nplat);
    for (cl_uint p = 0; p < nplat; ++p) {
        char buf[1024];
        clGetPlatformInfo(plats[p], CL_PLATFORM_NAME, sizeof(buf), buf, nullptr);
        std::printf("\n[platform %u] %s\n", p, buf);
        clGetPlatformInfo(plats[p], CL_PLATFORM_VERSION, sizeof(buf), buf, nullptr);
        std::printf("              version: %s\n", buf);

        cl_uint ndev = 0;
        clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 0, nullptr, &ndev);
        std::vector<cl_device_id> devs(ndev);
        clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, ndev, devs.data(), nullptr);
        for (cl_uint d = 0; d < ndev; ++d) {
            std::printf("  [device %u] name        : %s\n", d, get_str(devs[d], CL_DEVICE_NAME).c_str());
            std::printf("            version     : %s\n",    get_str(devs[d], CL_DEVICE_VERSION).c_str());
            std::printf("            opencl_c    : %s\n",    get_str(devs[d], CL_DEVICE_OPENCL_C_VERSION).c_str());
            std::printf("            driver      : %s\n",    get_str(devs[d], CL_DRIVER_VERSION).c_str());
            const std::string ext = get_str(devs[d], CL_DEVICE_EXTENSIONS);
            const bool has_fp16   = ext.find("cl_khr_fp16")     != std::string::npos;
            const bool has_subg   = ext.find("cl_khr_subgroups")!= std::string::npos ||
                                    ext.find("cl_intel_subgroups") != std::string::npos;
            const bool has_qcom_lb= ext.find("cl_qcom_large_buffer") != std::string::npos;
            std::printf("            fp16 ext    : %s\n", has_fp16   ? "yes" : "NO");
            std::printf("            subgroup ext: %s\n", has_subg   ? "yes" : "NO");
            std::printf("            qcom lb     : %s\n", has_qcom_lb? "yes" : "no");
            std::printf("            extensions  : %s\n", ext.c_str());
        }
    }
    return 0;
}
