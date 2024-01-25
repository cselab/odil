#!/usr/bin/env python3

import unittest
import odil
import numpy as np

class TestIO(unittest.TestCase):

    def test_raw_xmf(self):
        name = "foo"
        xmfpath = "data.xmf"
        nx, ny, nz = 3, 4, 5
        lx, ly, lz = 4, 5, 6
        for dtype in [np.float32, np.float64]:
            u_src = np.linspace(0, 1, nx * ny * nz).reshape(
                (nz, ny, nx)).astype(dtype)
            spacing = (lx / nx, ly / ny, lz / nz)
            odil.write_raw_with_xmf(u_src, xmfpath, spacing=spacing, name=name)


            u, meta = odil.read_raw_with_xmf(xmfpath)
            self.assertEqual(meta['count'], u_src.shape)
            np.testing.assert_array_almost_equal(meta['spacing'],
                                                 spacing,
                                                 decimal=8)
            self.assertEqual(meta['name'], name)
            self.assertEqual(meta['precision'], np.dtype(dtype).itemsize)
