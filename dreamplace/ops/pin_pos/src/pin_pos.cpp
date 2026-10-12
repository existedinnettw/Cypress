/**
 * @file   pin_pos.cpp
 * @author Yibo Lin
 * @date   Aug 2019
 * @brief  Given cell locations, compute pin locations on CPU
 */
#include <cmath>
#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief Given cell locations and orientations, compute pin locations.
/// Pin offsets are relative to the lower-left corner of the unrotated cell;
/// each cell is rotated counterclockwise by theta around its center.
/// Mirrors computePinPos in pin_pos_cuda_kernel.cu.
/// @param x cell locations in x direction
/// @param y cell locations in y direction
/// @param pin_offset_x pin offset in x direction
/// @param pin_offset_y pin offset in y direction
/// @param theta cell rotation angles in radians
/// @param pin2node_map map pin index to node index
/// @param num_pins number of pins
/// @param h cell heights
/// @param w cell widths
/// @param num_threads number of threads
/// @param pin_x pin positions in x direction
/// @param pin_y pin positions in y direction
template <typename T>
int computePinPosLauncher(const T* x, const T* y, const T* pin_offset_x,
                          const T* pin_offset_y, const T* theta,
                          const long* pin2node_map, int num_pins, const T* h,
                          const T* w, const int num_threads, T* pin_x,
                          T* pin_y) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_pins; ++i) {
    int node_id = pin2node_map[i];
    T hi = h[node_id];
    T wi = w[node_id];
    T cos_theta = std::cos(theta[node_id]);
    T sin_theta = std::sin(theta[node_id]);
    T ofx = pin_offset_x[i] - wi / 2;
    T ofy = pin_offset_y[i] - hi / 2;
    pin_x[i] = (x[node_id] + wi / 2) + ofx * cos_theta - ofy * sin_theta;
    pin_y[i] = (y[node_id] + hi / 2) + ofx * sin_theta + ofy * cos_theta;
  }

  return 0;
}

/// @brief Gradient of pin locations w.r.t. cell locations and orientations.
/// Mirrors computeNodeGrad in pin_pos_cuda_kernel.cu.
template <typename T>
int computePinPosGradLauncher(const T* grad_out_x, const T* grad_out_y,
                              const T* pin_offset_x, const T* pin_offset_y,
                              const T* theta, const int* flat_node2pin_map,
                              const int* flat_node2pin_start_map, const T* h,
                              const T* w, int num_nodes, const int num_threads,
                              T* grad_x, T* grad_y, T* grad_theta) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_nodes; ++i) {
    T gx = 0;
    T gy = 0;
    T gt = 0;
    T cos_theta = std::cos(theta[i]);
    T sin_theta = std::sin(theta[i]);
    T hi = h[i];
    T wi = w[i];
    for (int j = flat_node2pin_start_map[i]; j < flat_node2pin_start_map[i + 1];
         ++j) {
      int pin_id = flat_node2pin_map[j];
      gx += grad_out_x[pin_id];
      gy += grad_out_y[pin_id];
      T px_pt = cos_theta * (hi / 2 - pin_offset_y[pin_id]) +
                sin_theta * (wi / 2 - pin_offset_x[pin_id]);
      T py_pt = cos_theta * (pin_offset_x[pin_id] - wi / 2) +
                sin_theta * (pin_offset_y[pin_id] - hi / 2);
      gt += grad_out_x[pin_id] * px_pt + grad_out_y[pin_id] * py_pt;
    }
    grad_x[i] = gx;
    grad_y[i] = gy;
    grad_theta[i] = gt;
  }

  return 0;
}

/// @brief Given cell locations, compute pin locations
/// @param pos cell locations in x and then y direction
/// @param pin_offset_x pin offset in x direction
/// @param pin_offset_y pin offset in y direction
/// @param theta cell rotation angles in radians
/// @param pin2node_map map pin index to node index
/// @param flat_node2pin_map map node index to pins
/// @param flat_node2pin_start_map start index of flat_node2pin_map for each
/// node
/// @param h cell heights
/// @param w cell widths
/// @return pin positions in x and then y direction
at::Tensor pin_pos_forward(at::Tensor pos, at::Tensor pin_offset_x,
                           at::Tensor pin_offset_y, at::Tensor theta,
                           at::Tensor pin2node_map,
                           at::Tensor flat_node2pin_map,
                           at::Tensor flat_node2pin_start_map, at::Tensor h,
                           at::Tensor w) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  auto out = at::zeros(pin_offset_x.numel() * 2, pos.options());
  int num_nodes = pos.numel() / 2;
  int num_pins = pin_offset_x.numel();

  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "computePinPosLauncher", [&] {
    computePinPosLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes,
        DREAMPLACE_TENSOR_DATA_PTR(pin_offset_x, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pin_offset_y, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(theta, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, long), num_pins,
        DREAMPLACE_TENSOR_DATA_PTR(h, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(w, scalar_t), at::get_num_threads(),
        DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t) + num_pins);
  });

  return out;
}

at::Tensor pin_pos_backward(at::Tensor grad_out, at::Tensor pos,
                            at::Tensor pin_offset_x, at::Tensor pin_offset_y,
                            at::Tensor theta, at::Tensor pin2node_map,
                            at::Tensor flat_node2pin_map,
                            at::Tensor flat_node2pin_start_map, at::Tensor h,
                            at::Tensor w, int num_physical_nodes) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(grad_out);
  CHECK_EVEN(grad_out);
  CHECK_CONTIGUOUS(grad_out);

  int num_nodes = pos.numel() / 2;
  int num_pins = pin_offset_x.numel();
  auto out = at::zeros(num_nodes * 3, pos.options());  // grad for x, y, theta

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computePinPosGradLauncher", [&] {
        computePinPosGradLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t) + num_pins,
            DREAMPLACE_TENSOR_DATA_PTR(pin_offset_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pin_offset_y, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(theta, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(flat_node2pin_map, int),
            DREAMPLACE_TENSOR_DATA_PTR(flat_node2pin_start_map, int),
            DREAMPLACE_TENSOR_DATA_PTR(h, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(w, scalar_t), num_physical_nodes,
            at::get_num_threads(), DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t) + num_nodes,
            DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t) + num_nodes * 2);
      });

  return out;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::pin_pos_forward, "PinPos forward");
  m.def("backward", &DREAMPLACE_NAMESPACE::pin_pos_backward, "PinPos backward");
}
