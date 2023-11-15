import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def oscillator(d, w0, x):
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d / w)
    A = 1 / (2 * np.cos(phi))
    cos = torch.cos(phi + w * x)
    exp = torch.exp(-d * x)
    y = exp * 2 * A * cos
    return y


d, w0 = 2, 20
x = torch.linspace(0, 1, 500)[:, None]
y = oscillator(d, w0, x)
x_data = x[0:200:20]
y_data = y[0:200:20]
x_physics = torch.linspace(0, 1, 30)[:, None].requires_grad_(True)
torch.manual_seed(123)
model = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32,
                                                             32), nn.Tanh(),
                      nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for i in range(20000):
    optimizer.zero_grad()
    yh = model(x_data)
    loss1 = torch.mean((yh - y_data)**2)
    yhp = model(x_physics)
    dx, = torch.autograd.grad(yhp,
                              x_physics,
                              torch.ones_like(yhp),
                              create_graph=True)
    dx2, = torch.autograd.grad(dx,
                               x_physics,
                               torch.ones_like(dx),
                               create_graph=True)
    physics = dx2 + 2 * d * dx + w0 * w0 * yhp
    loss2 = 1e-4 * torch.mean(physics**2)
    loss = loss1 + loss2
    loss.backward()
    optimizer.step()
plt.plot(x, y)
plt.plot(x, model(x).detach(), '-')
plt.show()
