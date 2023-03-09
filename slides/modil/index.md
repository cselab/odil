---
theme: am205
title: mODIL
---

## mODIL (multiresolution ODIL)<br>Optimizing a Discrete Loss

> [[arXiv:2303.04679]](https://arxiv.org/abs/2303.04679)

<style>
.bodyimg img,
.bodyimg video {
  max-width:290px;margin:-5px -10px -15px -10px;
}
</style>

<script>
function pause_all(elem, nparent=2, rate=1.) {
  for (let i = 0; i < nparent; i++) {
    elem = elem.parentNode;
  }
  elem.querySelectorAll('video').forEach((video) => {
    if (video.paused) {
      video.playbackRate = rate;
      video.play();
    } else {
      video.pause();
    }
  })
}
function stop_all(elem, nparent=2) {
  for (let i = 0; i < nparent; i++) {
    elem = elem.parentNode;
  }
  elem.querySelectorAll('video').forEach((video) => {
    video.pause();
    video.currentTime = 0;
  })
}
</script>

## Multigrid Decomposition

  * Idea to [speed up the convergence of optimizers]{.color5} (L-BFGS, Adam, etc)
  * Use a hierarchy of grids, e.g. $N_1=65,\; N_2=33, \dots, N_L=3$
  * Decompose the solution $u$ on the fine grid $N_1$ as
    $$
    u = M_L(u_1, u_2, \dots, u_L) = u_1 + T_1 u_2 + T_1 T_2 u_3 + \dots + T_1\dots T_{L-1} u_L
    $$
    where $T_i$ is an interpolation operator from grid $N_{i+1}$ to $N_{i}$
  * Instead of minimizing the loss $$L(u) \rightarrow \min$$
    minimize $$L(M_L(u_1, u_2, \dots)) \rightarrow \min$$

## Multigrid Decomposition

  <div class="rowf">
  <div class="columnf3" style="text-align:left;width:41%;margin-right:-6%">
  * Cavity 2D
  * L-BFGS
  * $\mathrm{Re}=100$
  * grid $65\times 65$
  * [10x speedup]{.color5} with 5 levels  
    (65, 33, 17, 9, 5)
  </div>

  <div class="columnf3" style="text-align:left;width:32%;">
  <img style="height:180px;" data-src="media/mgopt/cav2d_re100_bfgs/train_loss.svg"></img>
  </div>

  <div class="columnf3" style="text-align:left;width:32%;">
  <img style="height:180px;" data-src="media/mgopt/cav2d_re100_bfgs/train_err.svg"></img>
  </div>
  </div>

## Multigrid Decomposition

  <div class="rowf">
  <div class="columnf3" style="text-align:left;width:41%;margin-right:-6%">
  * Cavity 2D
  * L-BFGS
  * $\mathrm{Re}=400$
  * grid $65\times 65$
  * [10x speedup]{.color5} with 5 levels  
    (65, 33, 17, 9, 5)
  </div>

  <div class="columnf3" style="text-align:left;width:32%;">
  <img style="height:180px;" data-src="media/mgopt/cav2d_re400_bfgs/train_loss.svg"></img>
  </div>

  <div class="columnf3" style="text-align:left;width:32%;">
  <img style="height:180px;" data-src="media/mgopt/cav2d_re400_bfgs/train_err.svg"></img>
  </div>

  </div>

## Multigrid Decomposition

  <div class="rowf">
  <div class="columnf3" style="text-align:left;width:41%;margin-right:-6%">
  * Poisson 1D
  * Adam
  * grid $65$
  </div>

  <div class="columnf3" style="text-align:left;width:32%;">
  <img style="height:180px;" data-src="media/mgopt/1d_adam/train_loss.svg"></img>
  </div>

  <div class="columnf3" style="text-align:left;width:32%;">
  <img style="height:180px;" data-src="media/mgopt/1d_adam/train_err.svg"></img>
  </div>

  </div>

## Optical Flow

* [Optical flow problem: velocity from tracer field]{.color5}

* Find a velocity field $\mathbf{u}(x,y,t)$ given that the tracer field $c(x,y,t)$  
  satisfies the advection equation
  $$\frac{\partial c}{\partial t} + \mathbf{u}\cdot\nabla{c} = 0$$
  and takes known initial $\left.c\right|_{t=0}=c_0$
  and final $\left.c\right|_{t=1}=c_1$ values

* The loss function is a discretization of
  $$\begin{align*}
  L(c,\mathbf{u})&\textstyle=
    \int\big(\frac{\partial c}{\partial t}+\mathbf{u}\cdot\nabla{c}\big)^2
    {\rm d}x {\rm d}y {\rm d}t
    +\int(\left.c\right|_{t=0}-c_0)^2 {\rm d}x {\rm d}y
    +\int(\left.c\right|_{t=1}-c_1)^2 {\rm d}x {\rm d}y
    \\
    &\textstyle
    +\int|k_\mathrm{xreg}\nabla^2 \mathbf{u}|^2{\rm d}x{\rm d}y {\rm d}t
    +\int|k_\mathrm{treg}\frac{\partial}{\partial t} \mathbf{u}|^2{\rm d}x{\rm d}y {\rm d}t
    \end{align*}
  $$

* Grid of $65\times 65\times 65$ points

## Optical Flow <a id='123' onclick="pause_all(this, 2, 0.25)" style="cursor: pointer;">&#x23ef;</a> <a id='123' onclick="stop_all(this)" style="cursor: pointer;">&#x23f9;</a>

  <div class="rowf">
  <div class="columnf" style="text-align:left;width:60%;">
  * Adam, grid $N=129$, 7 levels
    - runtime 360 ms / epoch (GPU)
    - runtime 1000 ms / epoch (CPU)
    - 500 epochs
  </div>
  <div class="columnf" style="text-align:left;width:40%;">
  <video height="150px" data-autoplay loop>
  <source data-src="media/mgopt/adv2d/nx129_lvl7/u.webm" type="video/webm">
  </video>
  </div>
  </div>

  <div class="rowf">
  <div class="columnf" style="text-align:left;width:60%;">
  * Adam, grid $N=257$, 7 levels
    - runtime 550 ms / epoch (GPU)
    - runtime 14 000 ms / epoch (CPU)
    - 500 epochs
  </div>
  <div class="columnf" style="text-align:left;width:40%;">
  <video height="150px" data-autoplay loop>
  <source data-src="media/mgopt/adv2d/nx257_lvl7/u.webm" type="video/webm">
  </video>
  </div>
  </div>

  <div class="rowf">
  <div class="columnf" style="text-align:left;width:60%;">
  * Adam, grid $N=257$, 5 levels
    - runtime 480 ms / epoch (GPU)
    - 500 epochs
  </div>
  <div class="columnf" style="text-align:left;width:40%;">
  <video height="150px" data-autoplay loop>
  <source data-src="media/mgopt/adv2d/nx257_lvl5/u.webm" type="video/webm">
  </video>
  </div>
  </div>

## Optical Flow

  <div class="rowf">
  <div class="columnf3" style="text-align:left;width:41%;margin-right:-6%">
  * Optical flow
  * Adam
  </div>

  <div class="columnf3" style="text-align:left;width:32%;">
  <img style="height:180px;" data-src="media/mgopt/adv2d/train_loss.svg"></img>
  </div>

  <div class="columnf3" style="text-align:left;width:32%;">
  <img style="height:180px;" data-src="media/mgopt/adv2d/train_epochtime.svg"></img>
  </div>

  </div>

## Body Shape Inference

* [Inferring body shape from velocity measurements]{.color5}

* Find a body fraction field $\chi(\mathbf{x})$ given that the velocity field $\mathbf{u}(\mathbf{x})$  
  satisfies the Navier-Stokes equations with penalization
  $$\begin{aligned}
  \nabla \cdot \mathbf{u} &= 0 \\ 
  (\mathbf{u}\cdot\nabla)\mathbf{u} &= -\nabla p + \frac{D}{\mathrm{Re}}\nabla^2\mathbf{u} - \lambda\chi\mathbf{u}
  \end{aligned}$$
  and takes known values $\mathbf{u}(\mathbf{x}_i)=\mathbf{u}_i$  
  in the measurement points $\mathbf{x}_i$ for $i=1,\dots,N$

* Transform $\chi = 1 / (1 + e^{-\hat\chi+5})$

## Body Shape Inference

* Penalization as mixture
  $$(1-\chi)\big((\mathbf{u}\cdot\nabla)\mathbf{u} +\nabla p - \nu\nabla^2\mathbf{u}\big) + \lambda\chi\mathbf{u} = 0$$
  with $\lambda=1$

* Properties:
  - exact $\mathbf{u}=0$ inside the body
  - small $\chi>0$ does not cause large force

## Body Shape Inference <a id='123' onclick="pause_all(this, 2)" style="cursor: pointer;">&#x23ef;</a> <a id='123' onclick="stop_all(this)" style="cursor: pointer;">&#x23f9;</a>

  <div class="rowf">
  <div class="columnf3" style="text-align:left;width:50%;">
  * Circle 2D, $\mathrm{Re}=60$
  * grid $129\times 65$
  * L-BFGS, 25000 epochs
  * runtime 30 min (CPU)
  </div>
  <div class="columnf3" style="width:25%;">
    <img style="height:110px;margin:0px;" data-src="media/mgopt/body2db/circle/ref/chi.png"></img>
    <img style="height:110px;margin:0px;" data-src="media/mgopt/body2db/circle/ref/omega.png"></img>
    <small>reference</small>
  </div>
  <div class="columnf3" style="width:25%;">
    <video height="110px" poster=media/mgopt/body2db/circle/bfgs/chi.png>
    <source data-src="media/mgopt/body2db/circle/bfgs/chi.webm" type="video/webm">
    </video>
    <video height="110px" poster=media/mgopt/body2db/circle/bfgs/omega.png>
    <source data-src="media/mgopt/body2db/circle/bfgs/omega.webm" type="video/webm">
    </video>
    <small>190 points</small>
  </div>
  </div>

  <img style="height:180px;" data-src="media/mgopt/body2db/circle/train_loss.svg"></img>
  <img style="height:180px;" data-src="media/mgopt/body2db/circle/train_error.svg"></img>
  <img style="height:180px;" data-src="media/mgopt/body2db/circle/train_error_chi.svg"></img>
  <img style=";" data-src="media/mgopt/body2db/circle/train_leg.svg"></img>

## Body Shape Inference <a id='123' onclick="pause_all(this, 2)" style="cursor: pointer;">&#x23ef;</a> <a id='123' onclick="stop_all(this)" style="cursor: pointer;">&#x23f9;</a>

  <div class="rowf">
  <div class="columnf3" style="text-align:left;width:50%;">
  * Half-circle 2D, $\mathrm{Re}=60$
  * grid $129\times 65$
  * L-BFGS, 25000 epochs
  * runtime 30 min (CPU)
  </div>
  <div class="columnf3" style="width:25%;">
    <img style="height:110px;margin:0px;" data-src="media/mgopt/body2db/half/ref/chi.png"></img>
    <img style="height:110px;margin:0px;" data-src="media/mgopt/body2db/half/ref/omega.png"></img>
    <small>reference</small>
  </div>
  <div class="columnf3" style="width:25%;">
    <video height="110px" poster=media/mgopt/body2db/half/bfgs/chi.png>
    <source data-src="media/mgopt/body2db/half/bfgs/chi.webm" type="video/webm">
    </video>
    <video height="110px" poster=media/mgopt/body2db/half/bfgs/omega.png>
    <source data-src="media/mgopt/body2db/half/bfgs/omega.webm" type="video/webm">
    </video>
    <small>190 points</small>
  </div>
  </div>

  <img style="height:180px;" data-src="media/mgopt/body2db/half/train_loss.svg"></img>
  <img style="height:180px;" data-src="media/mgopt/body2db/half/train_error.svg"></img>
  <img style="height:180px;" data-src="media/mgopt/body2db/half/train_error_chi.svg"></img>
  <img style=";" data-src="media/mgopt/body2db/half/train_leg.svg"></img>


## Body Shape Inference <a id='123' onclick="pause_all(this, 2)" style="cursor: pointer;">&#x23ef;</a> <a id='123' onclick="stop_all(this)" style="cursor: pointer;">&#x23f9;</a>

  <div class="rowf">
  <div class="columnf" style="text-align:left;margin-right:-5%;width:75%;">
  * Sphere 3D, $\mathrm{Re}=60$
  * grid $129\times 65\times 65$
  * L-BFGS, $20\,000$ epochs
  * runtime 2 hours (GPU)
  <div>
  <img style="height:140px;" data-src="media/mgopt/body3db/circle/train_loss.svg"></img>
  <img style="height:140px;" data-src="media/mgopt/body3db/circle/train_error.svg"></img>
  <img style="height:140px;" data-src="media/mgopt/body3db/circle/train_error_chi.svg"></img>
  <br><img style=";" data-src="media/mgopt/body3db/circle/train_leg.svg"></img>
  </div>
  </div>
  <div class="columnf" style="text-align:center;width:30%;">
  <div class="bodyimg" style="position:absolute;right:270px;width:260px;overflow:hidden;">
    <img data-src="media/mgopt/body3db/circle/ref/omega.png"></img>
    <br><small>reference</small>
  </div>
  <div class="bodyimg">
    <video poster=media/mgopt/body3db/circle/bfgs_684/omega.png>
    <source data-src="media/mgopt/body3db/circle/bfgs_684/omega.webm" type="video/webm">
    </video>
    <br><small>684 points</small>
  </div>
  <div class="bodyimg">
    <video poster=media/mgopt/body3db/circle/bfgs_171/omega.png>
    <source data-src="media/mgopt/body3db/circle/bfgs_171/omega.webm" type="video/webm">
    </video>
    <br><small>171 points</small>
  </div>
  <img style="max-width:290px;margin:0;" data-src="media/mgopt/body3d/cbar.svg"></img>
  </div>
  </div>

## Body Shape Inference <a id='123' onclick="pause_all(this, 2)" style="cursor: pointer;">&#x23ef;</a> <a id='123' onclick="stop_all(this)" style="cursor: pointer;">&#x23f9;</a>

  <div class="rowf">
  <div class="columnf" style="text-align:left;margin-right:-5%;width:75%;">
  * Half-sphere 3D, $\mathrm{Re}=60$
  * grid $129\times 65\times 65$
  * L-BFGS, $20\,000$ epochs
  * runtime 2 hours (GPU)
  <div>
  <img style="height:140px;" data-src="media/mgopt/body3db/half/train_loss.svg"></img>
  <img style="height:140px;" data-src="media/mgopt/body3db/half/train_error.svg"></img>
  <img style="height:140px;" data-src="media/mgopt/body3db/half/train_error_chi.svg"></img>
  <br><img style=";" data-src="media/mgopt/body3db/half/train_leg.svg"></img>
  </div>
  </div>
  <div class="columnf" style="text-align:center;width:30%;">
  <div class="bodyimg" style="position:absolute;right:270px;width:260px;overflow:hidden;">
    <img data-src="media/mgopt/body3db/half/ref/omega.png"></img>
    <br><small>reference</small>
  </div>
  <div class="bodyimg">
    <video poster=media/mgopt/body3db/half/bfgs_684/omega.png>
    <source data-src="media/mgopt/body3db/half/bfgs_684/omega.webm" type="video/webm">
    </video>
    <br><small>684 points</small>
  </div>
  <div class="bodyimg">
    <video poster=media/mgopt/body3db/half/bfgs_171/omega.png>
    <source data-src="media/mgopt/body3db/half/bfgs_171/omega.webm" type="video/webm">
    </video>
    <br><small>171 points</small>
  </div>
  <img style="max-width:290px;margin:0;" data-src="media/mgopt/body3d/cbar.svg"></img>
  </div>
  </div>

## Body Shape Inference

  <div class="rowf">
  <div class="columnf" style="text-align:left;margin-right:-5%;width:75%;">
  * Sphere 3D, $\mathrm{Re}=60$
  * grid $257\times 129\times 129$
  * L-BFGS, $100\,000$ epochs
  * runtime 8 hours (GPU)
  <div>
  <img style="height:140px;" data-src="media/mgopt/body3db/circle_N129b/train_loss.svg"></img>
  <img style="height:140px;" data-src="media/mgopt/body3db/circle_N129b/train_error.svg"></img>
  <img style="height:140px;" data-src="media/mgopt/body3db/circle_N129b/train_error_chi.svg"></img>
  <br><img style=";" data-src="media/mgopt/body3db/circle_N129b/train_leg.svg"></img>
  </div>
  </div>
  <div class="columnf" style="text-align:center;width:30%;">
  <div class="bodyimg" style="position:absolute;right:270px;width:260px;overflow:hidden;">
    <img data-src="media/mgopt/body3db/circle_N129b/ref/omega.png"></img>
    <br><small>reference</small>
  </div>
  <div class="bodyimg">
    <img data-src="media/mgopt/body3db/circle_N129b/bfgs_i1/omega.png"></img>
    <br><small>698 points</small>
  </div>
  <img style="max-width:290px;margin:0;" data-src="media/mgopt/body3d/cbar.svg"></img>
  </div>
  </div>

## Body Shape Inference

  <div class="rowf">
  <div class="columnf" style="text-align:left;margin-right:-5%;width:75%;">
  * Sphere 3D, $\mathrm{Re}=60$
  * grid $257\times 129\times 129$
  * L-BFGS, $40\,000$ epochs
  * runtime 3 hours (GPU)
  <div>
  <img style="height:140px;" data-src="media/mgopt/body3db/half_N129b/train_loss.svg"></img>
  <img style="height:140px;" data-src="media/mgopt/body3db/half_N129b/train_error.svg"></img>
  <img style="height:140px;" data-src="media/mgopt/body3db/half_N129b/train_error_chi.svg"></img>
  <br><img style=";" data-src="media/mgopt/body3db/half_N129b/train_leg.svg"></img>
  </div>
  </div>
  <div class="columnf" style="text-align:center;width:30%;">
  <div class="bodyimg" style="position:absolute;right:270px;width:260px;overflow:hidden;">
    <img data-src="media/mgopt/body3db/half_N129b/ref/omega.png"></img>
    <br><small>reference</small>
  </div>
  <div class="bodyimg">
    <img data-src="media/mgopt/body3db/half_N129b/bfgs_i1/omega.png"></img>
    <br><small>698 points</small>
  </div>
  <img style="max-width:290px;margin:0;" data-src="media/mgopt/body3d/cbar.svg"></img>
  </div>
  </div>
