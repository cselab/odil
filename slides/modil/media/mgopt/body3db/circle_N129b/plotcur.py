#!/usr/bin/env python3


import plot
import re

if __name__ == "__main__":
    args = plot.parse_args()
    lines = [
        ('bfgs_i1/train.csv', 'L-BFGS_TF 698 points', 'C0-'),
        ('ref/train.csv', 'L-BFGS_TF (forward)', 'C2-'),
    ]
    plot.plot(lines, args)
