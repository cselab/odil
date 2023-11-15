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

torch.set_default_tensor_type(torch.FloatTensor)
d, w0 = 2.0, 20.0
x = torch.linspace(0, 1, 500)[:, None]
y = oscillator(d, w0, x)
x_data = x[0:200:20]
y_data = y[0:200:20]
x_physics = torch.linspace(0, 1, 30)[:, None].requires_grad_(True)
torch.manual_seed(123)
model = nn.Sequential(nn.Linear(1, 32), nn.Tanh(),  #
                      nn.Linear(32, 16), nn.Tanh(), #
                      nn.Linear(16, 1),)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
ones = torch.ones_like(x_physics)
for i in range(20000):
    optimizer.zero_grad()
    loss1 = torch.mean((model(x_data) - y_data)**2)
    yhp = model(x_physics)
    dx, = torch.autograd.grad(yhp, x_physics, ones, create_graph=True)
    dx2, = torch.autograd.grad(dx, x_physics, ones, create_graph=True)
    physics = dx2 + 2 * d * dx + w0 * w0 * yhp
    loss2 = 1e-4 * torch.mean(physics**2)
    loss = loss1 + loss2
    loss.backward()
    optimizer.step()
plt.plot(x, y)
plt.plot(x, model(x).detach(), '-')
plt.show()
