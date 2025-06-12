// maigem_core.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

/**
 * @brief MAI-GEM的时间核心：选择性扫描操作。
 *        这是一个简化的、离散化的SSM前向扫描过程。
 *        它实现了 h_t = A_t * h_{t-1} + B_t * x_t 的序列化计算，
 *        其中 A_t 和 B_t 是根据选择性参数 delta_t 动态计算的。
 *
 * @param x_seq 输入序列 (L, D)
 * @param delta_seq 选择性参数Δ (L, D) - 控制遗忘/记忆的速率
 * @param A_in 基础状态转换矩阵 (D, N)
 * @param B_in 基础输入矩阵 (D, N)
 * @return py::array_t<double> 输出序列 (L, D)
 */
py::array_t<double> selective_scan(py::array_t<double> x_seq,
                                   py::array_t<double> delta_seq,
                                   py::array_t<double> A_in,
                                   py::array_t<double> B_in)
{
    // 获取Numpy数组的访问器
    auto x_buf = x_seq.request();
    auto delta_buf = delta_seq.request();
    auto A_buf = A_in.request();
    auto B_buf = B_in.request();

    if (x_buf.ndim != 2 || delta_buf.ndim != 2 || A_buf.ndim != 2 || B_buf.ndim != 2) {
        throw std::runtime_error("Input arrays must be 2-dimensional");
    }

    // 获取维度
    ssize_t L = x_buf.shape[0]; // Sequence length
    ssize_t D = x_buf.shape[1]; // Hidden dimension
    ssize_t N = A_buf.shape[1]; // State dimension

    // 创建输出数组
    auto result = py::array_t<double>(x_buf.size);
    result.resize({L, D});
    auto res_buf = result.request();

    // 获取原始指针
    double *x_ptr = static_cast<double *>(x_buf.ptr);
    double *delta_ptr = static_cast<double *>(delta_buf.ptr);
    double *A_ptr = static_cast<double *>(A_buf.ptr);
    double *B_ptr = static_cast<double *>(B_buf.ptr);
    double *res_ptr = static_cast<double *>(res_buf.ptr);

    // 初始化隐藏状态 h
    std::vector<double> h(N, 0.0);

    // 沿时间序列进行扫描
    for (ssize_t t = 0; t < L; ++t) {
        // --- 动态计算 A_t 和 B_t ---
        // A_t = exp(delta_t * A)
        // B_t = delta_t * B
        // 这是SSM离散化的一个常用方法 (ZOH)
        // 为了简化，我们这里使用一阶近似：
        // A_t ≈ 1 + delta_t * A
        // B_t ≈ delta_t * B
        
        std::vector<double> h_next(N, 0.0);
        
        // 计算 h_next = (1 + delta_t * A) * h_t
        for (ssize_t n = 0; n < N; ++n) {
            double ah_sum = 0.0;
            for (ssize_t d = 0; d < D; ++d) {
                // delta_t 是一个向量，每个维度有自己的值
                ah_sum += delta_ptr[t * D + d] * A_ptr[d * N + n] * h[n];
            }
            h_next[n] = h[n] + ah_sum;
        }

        // 计算 h_next += (delta_t * B) * x_t
        for (ssize_t n = 0; n < N; ++n) {
            double bx_sum = 0.0;
            for (ssize_t d = 0; d < D; ++d) {
                bx_sum += delta_ptr[t * D + d] * B_ptr[d * N + n] * x_ptr[t * D + d];
            }
            h_next[n] += bx_sum;
        }
        
        h = h_next;

        // 计算输出 y_t = C * h_t (这里简化为 y_t = h_t，并假设 D=N)
        // 在实际模型中，会有一个可学习的矩阵C
        if (D == N) {
            for (ssize_t d = 0; d < D; ++d) {
                res_ptr[t * D + d] = h[d];
            }
        }
    }

    return result;
}


// 使用pybind11定义Python模块
PYBIND11_MODULE(maigem_core, m) {
    m.doc() = "MAI-GEM Core C++ extension for high-performance computation";
    m.def("selective_scan", &selective_scan, "Performs the selective scan operation for the temporal core");
}


// # 本代码采用 GNU General Public License v3.0 (GPL 3.0) 开源协议。  
// # 作者：vincent  
// # 协议链接：https://www.gnu.org/licenses/gpl-3.0.html  