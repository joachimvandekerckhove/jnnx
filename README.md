# jnnx: Just Another Neural Network Exchange

A small interface that lets JAGS call ONNX-format neural networks as functions.

---

Example

```r
model {
  f <- fnnet(x)
  y ~ dnorm(f, sigma)
}
```

---

The goal is to make it possible to evaluate trained neural networks inside Gibbs samplers.
