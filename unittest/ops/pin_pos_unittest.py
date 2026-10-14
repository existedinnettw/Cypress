##
# @file   pin_pos_unitest.py
# @author Yibo Lin
# @date   Aug 2019
#

import math
import os
import sys
import unittest

import numpy as np
import torch

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
from dreamplace.ops.pin_pos import pin_pos
sys.path.pop()


def golden_pin_pos(pos, pin_offset_x, pin_offset_y, theta, pin2node_map,
                   num_physical_nodes, h, w):
    """Pin locations of cells rotated counterclockwise by theta around their
    centres; pin offsets are from the lower-left corner of the unrotated cell
    (pin_pos_cuda_kernel.cu::computePinPos)."""
    num_nodes = pos.numel() // 2
    node = pin2node_map.long()
    x = pos[0:num_physical_nodes][node]
    y = pos[num_nodes:num_nodes + num_physical_nodes][node]
    wi, hi = w[node], h[node]
    cos_t, sin_t = torch.cos(theta[node]), torch.sin(theta[node])
    ofx = pin_offset_x - wi / 2
    ofy = pin_offset_y - hi / 2
    pin_x = x + wi / 2 + ofx * cos_t - ofy * sin_t
    pin_y = y + hi / 2 + ofx * sin_t + ofy * cos_t
    return torch.cat([pin_x, pin_y])


class PinPosOpTest(unittest.TestCase):
    def setUp(self):
        # three physical cells and one filler (fillers carry no pins)
        node2pin_map = [[0, 4], [1, 2, 3], [5]]
        self.num_physical_nodes = len(node2pin_map)
        num_pins = sum(len(p) for p in node2pin_map)
        self.pin2node_map = np.zeros(num_pins, dtype=np.int32)
        for node_id, pins in enumerate(node2pin_map):
            self.pin2node_map[pins] = node_id
        self.flat_node2pin_map = np.concatenate(node2pin_map).astype(np.int32)
        self.flat_node2pin_start_map = np.cumsum(
            [0] + [len(p) for p in node2pin_map]).astype(np.int32)

        self.pos = np.array([1, 2, 3, 40, 10, 20, 30, 50], dtype=np.float32)
        self.w = np.array([4, 2, 6, 1], dtype=np.float32)
        self.h = np.array([2, 2, 3, 1], dtype=np.float32)
        self.pin_offset_x = np.array([0.5, 0.2, 1.5, 1.8, 3.5, 5.0],
                                     dtype=np.float32)
        self.pin_offset_y = np.array([0.25, 1.75, 0.5, 1.0, 1.5, 2.5],
                                     dtype=np.float32)
        # every quarter turn plus a fractional angle for the gradient check
        self.theta = np.array([math.pi / 2, math.pi, 0.3], dtype=np.float32)

    def tensors(self, device):
        t = lambda a: torch.from_numpy(a).to(device)
        return dict(pin_offset_x=t(self.pin_offset_x),
                    pin_offset_y=t(self.pin_offset_y),
                    pin2node_map=t(self.pin2node_map),
                    flat_node2pin_map=t(self.flat_node2pin_map),
                    flat_node2pin_start_map=t(self.flat_node2pin_start_map),
                    h=t(self.h),
                    w=t(self.w))

    def run_op(self, device):
        d = self.tensors(device)
        pos = torch.from_numpy(self.pos).to(device).requires_grad_()
        theta = torch.from_numpy(self.theta).to(device).requires_grad_()
        out = pin_pos.PinPosFunction.apply(pos, d["pin_offset_x"],
                                           d["pin_offset_y"], theta,
                                           d["pin2node_map"].long(),
                                           d["flat_node2pin_map"],
                                           d["flat_node2pin_start_map"],
                                           self.num_physical_nodes, d["h"],
                                           d["w"])
        # weight the pins so every gradient term matters
        weights = torch.arange(1, out.numel() + 1, dtype=out.dtype,
                               device=device)
        (out * weights).sum().backward()
        return (out.detach().cpu().numpy(), pos.grad.cpu().numpy(),
                theta.grad.cpu().numpy())

    def golden(self):
        d = self.tensors("cpu")
        pos = torch.from_numpy(self.pos).requires_grad_()
        theta = torch.from_numpy(self.theta).requires_grad_()
        out = golden_pin_pos(pos, d["pin_offset_x"], d["pin_offset_y"], theta,
                             d["pin2node_map"], self.num_physical_nodes,
                             d["h"], d["w"])
        weights = torch.arange(1, out.numel() + 1, dtype=out.dtype)
        (out * weights).sum().backward()
        return (out.detach().numpy(), pos.grad.numpy(), theta.grad.numpy())

    def check(self, device):
        for got, want in zip(self.run_op(device), self.golden()):
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

    def test_pin_pos_cpu(self):
        self.check("cpu")

    @unittest.skipUnless(torch.cuda.device_count(), "no CUDA device")
    def test_pin_pos_cuda(self):
        self.check("cuda")


if __name__ == '__main__':
    unittest.main()
