import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math


def oscillator(d, w0, x):
    w = math.sqrt(w0**2 - d**2)
    phi = math.atan2(-d, w)
    y = torch.exp(-d * x) * torch.cos(phi + w * x) / math.cos(phi)
    return y


torch.manual_seed(123)
torch.set_default_dtype(torch.float32)
d, w0 = 2.0, 20.0
x = torch.linspace(0, 1, 500)[:, None]
y = oscillator(d, w0, x)
x_data = x[0:20:10]
y_data = y[0:20:10]
x_physics = torch.linspace(0, 1, 25)[:, None].requires_grad_(True)
model = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
ones = torch.ones_like(x_physics)
for i in range(5000):
    optimizer.zero_grad()
    yhp = model(x_physics)
    dx, = torch.autograd.grad(yhp, x_physics, ones, create_graph=True)
    dx2, = torch.autograd.grad(dx, x_physics, ones, create_graph=True)
    physics = dx2 + 2 * d * dx + w0 * w0 * yhp
    loss = torch.mean(
        (model(x_data) - y_data)**2) + 1e-4 * torch.mean(physics**2)
    loss.backward()
    optimizer.step()
plt.plot(x, y)
plt.plot(x, model(x).detach(), '-')
plt.show()
