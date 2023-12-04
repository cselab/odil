# Tests

Run a single test with the TensorFlow backend
```
make test_domain_tf
```

Run a single test with the JAX backend
```
make test_domain_jax
```

Run all tests with both backends in parallel on a CPU
```
make -j4 all_tf all_jax
```

Run all tests with both backends in parallel on GPU 0
```
CUDA_VISIBLE_DEVICES=0 make -j4 all_tf all_jax
```
