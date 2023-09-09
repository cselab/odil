# Tests

Run a single test with the TensorFlow backend
```
make test_domain_tf
```

Run a single test with the JAX backend
```
make test_domain_jax
```

Run all tests with both backends in parallel
```
make -j4 all_tf all_jax
```
