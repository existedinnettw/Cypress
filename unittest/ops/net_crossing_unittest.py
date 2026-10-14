##
# @file  net_crossing.py
# @author Niansong Zhang
# @date  Jul 2024
#

import os
import sys
import numpy as np
import unittest

import torch

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from dreamplace.ops.net_crossing import net_crossing

sys.path.pop()

EPSILON = 1e-5  # the kernels' guard against parallel segments


def bell_func(x, lambda_, mu_, sigma_):
    if torch.abs(x) <= 0.5:
        return 1 - lambda_ * torch.pow(x, 2)
    elif torch.abs(x) <= 1:
        return mu_ * torch.pow(torch.abs(x) - sigma_, 2)
    else:
        return 0


def golden_netcrossing(pin_x, pin_y, net2pin_map, pin_side, _lambda, _mu, _sigma):
    num_nets = len(net2pin_map)
    net_crossing = torch.zeros(num_nets, dtype=pin_x.dtype)
    for i in range(num_nets):
        for j in range(i + 1, num_nets):
            pins_i_idx = net2pin_map[i]
            pins_j_idx = net2pin_map[j]
            src_i, src_j = pins_i_idx[0], pins_j_idx[0]
            for i_sink_pin_idx in pins_i_idx[1:]:
                x1, y1 = pin_x[src_i], pin_y[src_i]
                x2, y2 = pin_x[i_sink_pin_idx], pin_y[i_sink_pin_idx]
                for j_sink_pin_idx in pins_j_idx[1:]:
                    # segments wholly on opposite sides of the board cannot cross
                    if (pin_side[src_i] == pin_side[i_sink_pin_idx]
                            and pin_side[src_j] == pin_side[j_sink_pin_idx]
                            and pin_side[src_i] != pin_side[src_j]):
                        continue
                    x3, y3 = pin_x[src_j], pin_y[src_j]
                    x4, y4 = pin_x[j_sink_pin_idx], pin_y[j_sink_pin_idx]
                    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) + EPSILON
                    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
                    net_crossing[i] += bell_func(
                        t - 0.5, _lambda, _mu, _sigma
                    ) * bell_func(u - 0.5, _lambda, _mu, _sigma)
    return net_crossing.sum()


class NetCrossingOpTest(unittest.TestCase):
    pin_pos = np.array(
        [[0, 0], [2, 2], [2, -1], [-2, -2], [-1, 1], [2, 1], [-1, -2]],
        dtype=np.float32,
    )
    net2pin_map = [[0, 1, 2, 3], [4, 5, 6]]
    _lambda, _mu, _sigma = 2, 2, 1

    def run_case(self, pin_side, device):
        dtype = torch.float32
        flat_net2pin_map = np.concatenate(self.net2pin_map).astype(np.int32)
        flat_net2pin_start_map = np.cumsum(
            [0] + [len(p) for p in self.net2pin_map]
        ).astype(np.int32)
        net_mask = np.ones(len(self.net2pin_map), dtype=np.uint8)
        pin_side = np.asarray(pin_side, dtype=np.int32)

        pos = torch.from_numpy(np.transpose(self.pin_pos).copy()).reshape([-1])
        n = pos.numel() // 2

        golden_var = pos.clone().requires_grad_()
        golden = golden_netcrossing(
            golden_var[:n], golden_var[n:], self.net2pin_map, pin_side,
            self._lambda, self._mu, self._sigma,
        )
        if golden.requires_grad:
            golden.backward()
            golden_grad = golden_var.grad.numpy()
        else:  # nothing crosses: no graph, zero gradient
            golden_grad = np.zeros(pos.numel(), dtype=np.float32)

        t = lambda a: torch.from_numpy(a).to(device)
        custom = net_crossing.NetCrossing(
            flat_netpin=t(flat_net2pin_map),
            netpin_start=t(flat_net2pin_start_map),
            net_mask=t(net_mask),
            pin_side=t(pin_side),
            _lambda=torch.tensor(self._lambda, dtype=dtype, device=device),
            _mu=torch.tensor(self._mu, dtype=dtype, device=device),
            _sigma=torch.tensor(self._sigma, dtype=dtype, device=device),
        )
        var = pos.clone().to(device).requires_grad_()
        result = custom.forward(var)
        result.backward()

        np.testing.assert_allclose(
            result.detach().cpu().numpy(), float(golden), rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            var.grad.cpu().numpy(), golden_grad, rtol=1e-4, atol=1e-4
        )
        return float(result)

    def check(self, device):
        same_side = self.run_case([1] * 7, device)
        self.assertGreater(same_side, 0)
        # net 0 on the top, net 1 on the bottom: no crossing between them
        self.assertEqual(self.run_case([1, 1, 1, 1, 0, 0, 0], device), 0)

    def test_net_crossing_cpu(self):
        self.check("cpu")

    @unittest.skipUnless(torch.cuda.device_count(), "no CUDA device")
    def test_net_crossing_cuda(self):
        self.check("cuda")


if __name__ == "__main__":
    unittest.main()
