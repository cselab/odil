import numpy as np
from argparse import Namespace


class Optimizer:

    def __init__(self, name=None, displayname=None, dtype=None):
        self.name = name
        self.displayname = displayname if displayname is not None else name
        self.dtype = dtype
        self.pinfo = None
        self.evals = 0

    def run(self,
            x0,
            loss_grad,
            epochs,
            callback=None,
            epoch_start=0,
            **kwargs):
        optinfo = Namespace()
        optinfo.evals = 0  # Number of `loss_grad()` evaluations.
        optinfo.epochs = 0  # Number of epochs actually done.
        return x0, optinfo


class EarlyStopError(Exception):

    def __init__(self, msg, optinfo):
        super().__init__(msg)
        self.optinfo = optinfo


class LbfgsbOptimizer(Optimizer):

    def __init__(self,
                 pgtol=1e-16,
                 m=50,
                 maxls=50,
                 factr=0,
                 dtype=None,
                 mod=None,
                 **kwargs):
        """
        pgtol: `float`
            Gradient convergence condition.
        m: `int`
            Maximum number of variable metric corrections used to define
            the limited memory matrix.
        maxls: `int`
            Maximum number of line search steps (per iteration).
        factr: `float`
            Convergence condition:
               1e12 (low accuracy),
               1e7 (moderate accuracy),
               10.0 (extremely high accuracy)
        """
        super().__init__(name="lbfgsb", displayname="L-BFGS-B", dtype=dtype)
        self.mod = mod
        self.pgtol = pgtol
        self.m = m
        self.maxls = maxls
        self.factr = factr
        self.evals = 0

    def run(self,
            x0,
            loss_grad,
            epochs=None,
            callback=None,
            epoch_start=0,
            **kwargs):
        self.epoch = epoch_start
        mod = self.mod
        dtype = self.dtype

        def split_by_sizes(array, sizes, axis=0):
            cumsum = np.cumsum(sizes)[:-1]
            return np.split(array, cumsum, axis=axis)

        def flat_to_arrays(x):
            x = np.array(x, dtype=dtype)
            sizes = [np.prod(a.shape) for a in x0]
            split = split_by_sizes(x[:sum(sizes)], sizes)
            arrays = [mod.reshape(s, a.shape) for s, a in zip(split, x0)]
            return arrays

        def arrays_to_flat(arrays):
            arrays = [np.array(a, dtype=np.float64) for a in arrays]
            x = np.concatenate([a.flatten() for a in arrays], axis=0)
            return x

        def callback_wrap(x):
            self.epoch += 1
            if callback:
                arrays = flat_to_arrays(x)
                callback(arrays, self.epoch, self.pinfo)

        def func_wrap(x):
            self.evals += 1
            arrays = flat_to_arrays(x)
            loss, grads, pinfo = loss_grad(arrays)
            loss = np.array(loss, dtype=np.float64)
            grad = arrays_to_flat(grads)
            self.pinfo = pinfo
            return loss, grad

        x0 = [np.array(a, dtype=np.float64) for a in x0]
        x0_flat = arrays_to_flat(x0)

        from scipy import optimize
        x, f, sinfo = optimize.fmin_l_bfgs_b(func=func_wrap,
                                             x0=x0_flat,
                                             maxiter=epochs,
                                             pgtol=self.pgtol,
                                             m=self.m,
                                             maxls=self.maxls,
                                             factr=self.factr,
                                             maxfun=np.inf,
                                             callback=callback_wrap)
        optinfo = Namespace()
        optinfo.warnflag = sinfo['warnflag']
        optinfo.task = sinfo['task']
        optinfo.evals = sinfo['funcalls']
        optinfo.epochs = sinfo['nit']
        if optinfo.warnflag not in [0, 1] or optinfo.epochs < epochs:
            raise EarlyStopError(
                ", ".join("{:}={:}".format(key, sinfo.get(key, ''))
                          for key in ['warnflag', 'task', 'funcalls', 'nit']),
                optinfo)
        arrays = flat_to_arrays(x)
        return arrays, optinfo


class LbfgsOptimizer(Optimizer):

    def __init__(self,
                 pgtol=1e-16,
                 m=50,
                 maxls=50,
                 factr=0,
                 dtype=None,
                 mod=None,
                 **kwargs):
        """
        pgtol: `float`
            Gradient convergence condition.
        m: `int`
            Maximum number of variable metric corrections used to define
            the limited memory matrix.
        maxls: `int`
            Maximum number of line search steps (per iteration).
        factr: `float`
            Convergence condition:
               1e12 (low accuracy),
               1e7 (moderate accuracy),
               10.0 (extremely high accuracy)
        """
        super().__init__(name="lbfgs", displayname="L-BFGS_TF", dtype=dtype)
        self.mod = mod
        self.pgtol = pgtol
        self.m = m
        self.maxls = maxls
        self.factr = factr
        self.evals = 0
        self.last_x = None

    def run(self,
            x0,
            loss_grad,
            epochs=None,
            callback=None,
            epoch_start=0,
            **kwargs):
        self.epoch = epoch_start
        mod = self.mod
        tf = mod.tf
        import tensorflow_probability as tfp

        def flat_to_arrays(x):
            sizes = [np.prod(a.shape) for a in x0]
            split = mod.split_by_sizes(x[:sum(sizes)], sizes)
            arrays = [mod.reshape(s, a.shape) for s, a in zip(split, x0)]
            return arrays

        def arrays_to_flat(arrays):
            x = mod.concatenate([mod.flatten(a) for a in arrays], axis=0)
            return x

        def callback_wrap(x):
            self.epoch += 1
            if callback:
                sizes = [np.prod(a.shape) for a in x0]
                split = mod.split_by_sizes(x[:sum(sizes)], sizes)
                arrays = [mod.reshape(s, a.shape) for s, a in zip(split, x0)]
                callback(arrays, self.epoch, self.pinfo)

        def func_wrap(x):
            self.last_x = x
            self.evals += 1
            arrays = flat_to_arrays(x)
            loss, grads, pinfo = loss_grad(arrays)
            loss = mod.constant(loss)
            grad = arrays_to_flat(grads)
            self.pinfo = pinfo
            return loss, grad

        def stopping_condition(converged, failed):
            callback_wrap(self.last_x)
            if self.epoch > self.epochs:
                return tf.ones(converged.shape, dtype=bool)
            return tfp.optimizer.converged_all(converged, failed)

        x0_flat = arrays_to_flat(x0)
        self.last_x = x0_flat
        self.epochs = epochs
        res = tfp.optimizer.lbfgs_minimize(
            func_wrap,
            initial_position=x0_flat,
            max_iterations=epochs,
            num_correction_pairs=self.m,
            max_line_search_iterations=self.maxls,
            tolerance=-1,
            x_tolerance=-1,
            f_relative_tolerance=-1,
            stopping_condition=stopping_condition,
        )

        arrays = flat_to_arrays(res.position)
        optinfo = Namespace()
        optinfo.epochs = 0
        optinfo.evals = self.evals
        return arrays, optinfo


class AdamTfOptimizer(Optimizer):

    def __init__(self, dtype=None, mod=None, **kwargs):
        super().__init__(name="adam_tf", displayname="AdamTf", dtype=dtype)
        self.mod = mod

    def run(self,
            x0,
            loss_grad,
            epochs=None,
            callback=None,
            lr=1e-3,
            epoch_start=0,
            **kwargs):

        mod = self.mod
        tf = mod.tf

        class CustomModel(tf.keras.Model):

            def __init__(self, x):
                super().__init__()
                self.x = x
                self.evals = 0
                self.epoch = epoch_start

            def __call__(self):
                self.evals += 1
                return loss_grad(self.x)

            def train_step(self, _):
                loss, grads, pinfo = self()
                self.optimizer.apply_gradients(zip(grads, self.x))
                self.pinfo = pinfo
                return {'loss': loss}

        x = [mod.tf.Variable(e) for e in x0]
        model = CustomModel(x)

        class CustomCallback(tf.keras.callbacks.Callback):

            def on_epoch_end(self, epoch, logs=None):
                model.epoch = epoch_start + epoch
                if epoch > 0 and callback:
                    callback(model.x, model.epoch, model.pinfo)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      run_eagerly=True)
        dummy = [1]  # Unused input.
        model.fit(dummy,
                  epochs=epochs + 1,
                  callbacks=[CustomCallback()],
                  verbose=0)
        optinfo = Namespace()
        optinfo.epochs = epochs
        optinfo.evals = model.evals
        return x, optinfo


class GdOptimizer(Optimizer):

    def __init__(self, dtype=None, mod=None, **kwargs):
        super().__init__(name="gd", displayname="GD", dtype=dtype)
        self.mod = mod

    def run(self,
            x0,
            loss_grad,
            epochs=None,
            callback=None,
            lr=1e-3,
            epoch_start=0,
            **kwargs):

        mod = self.mod
        x = [mod.copy(e) for e in x0]
        for epoch in range(epoch_start + 1, epoch_start + epochs + 1):
            self.evals += 1
            loss, grads, pinfo = loss_grad(x)
            for i in range(len(x)):
                x[i] -= grads[i] * lr
            if epoch > 0 and callback is not None:
                callback(x, epoch, pinfo)

        optinfo = Namespace()
        optinfo.epochs = epochs
        optinfo.evals = self.evals
        return x, optinfo


class AdamNativeOptimizer(Optimizer):

    def __init__(self, dtype=None, mod=None, **kwargs):
        super().__init__(name="adamn", displayname="AdamNative", dtype=dtype)
        self.mod = mod

    def run(self,
            x0,
            loss_grad,
            epochs=None,
            callback=None,
            lr=1e-3,
            epoch_start=0,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            jit=True,
            **kwargs):
        '''
        Based on
        https://github.com/keras-team/keras/blob/v2.12.0/keras/optimizers/adam.py
        '''

        dtype = self.dtype
        mod = self.mod
        lr = mod.cast(lr, dtype)
        beta_1 = mod.cast(beta_1, dtype)
        beta_2 = mod.cast(beta_2, dtype)

        def _step(x, m, v, grads, local_epoch):
            local_epoch = mod.cast(local_epoch, dtype)
            beta_1_power = beta_1**local_epoch
            beta_2_power = beta_2**local_epoch
            alpha = lr * mod.sqrt(1 - beta_2_power) / (1 - beta_1_power)
            m = [m + (g - m) * (1 - beta_1) for m, g in zip(m, grads)]
            v = [
                v + (mod.square(g) - v) * (1 - beta_2)
                for v, g in zip(v, grads)
            ]
            x = [
                x - (m * alpha) / (mod.sqrt(v) + epsilon)
                for x, m, v in zip(x, m, v)
            ]
            return x, m, v

        step = _step
        if jit:
            if mod.jax:
                step = mod.jax.jit(_step)
            elif mod.tf:
                step = mod.tf.function(_step, jit_compile=True)

        x = [mod.copy(e) for e in x0]
        m = [mod.zeros_like(e) for e in x0]
        v = [mod.zeros_like(e) for e in x0]
        for epoch in range(epoch_start + 1, epoch_start + epochs + 1):
            self.evals += 1
            loss, grads, pinfo = loss_grad(x)
            x, m, v = step(x, m, v, grads, mod.constant(epoch - epoch_start))
            if epoch > 0 and callback is not None:
                callback(x, epoch, pinfo)

        optinfo = Namespace()
        optinfo.epochs = epochs
        optinfo.evals = self.evals
        return x, optinfo


def make_optimizer(name, dtype=None, mod=None, **kwargs):
    if name == "lbfgsb":
        optimizer = LbfgsbOptimizer(dtype=dtype, mod=mod, **kwargs)
    elif name == "lbfgs":
        optimizer = LbfgsOptimizer(dtype=dtype, mod=mod, **kwargs)
    elif name == "adam_tf":
        optimizer = AdamTfOptimizer(dtype=dtype, mod=mod, **kwargs)
    elif name == "adam" or name == "adamn":
        optimizer = AdamNativeOptimizer(dtype=dtype, mod=mod, **kwargs)
    elif name == "gd":
        optimizer = GdOptimizer(dtype=dtype, mod=mod, **kwargs)
    else:
        raise ValueError("Unknown optimizer '{}'".format(name))
    return optimizer
