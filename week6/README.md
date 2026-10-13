# Week 6

Use uv for Python commands and CPU by default. Save figures in `evidence/`
and raw runs with their settings in `artifacts/`. Downloads in `inputs/`
and raw runs are ignored by git.

## Download MNIST and CIFAR-10

Use the [original MNIST host](http://yann.lecun.com/exdb/mnist/)
and the [University of Toronto CIFAR-10 download](https://cave.cs.toronto.edu/kriz/cifar.html).

Requests to the original MNIST file URLs returned HTTP 404 from this Mac during verification.

Run from `week6/` in bash or zsh:

Retry connection failures and resume the existing partial CIFAR-10 archive.

```sh
mkdir -p inputs/mnist inputs/cifar10
mnist_url=http://yann.lecun.com/exdb/mnist
for split in train t10k; do
  for kind in images-idx3-ubyte labels-idx1-ubyte; do
    name="$split-$kind.gz"
    curl -fL --retry 3 --retry-all-errors "$mnist_url/$name" -o "inputs/mnist/$name" \
      --write-out "%{url_effective}\n"
  done
done >> README.md
curl -fL --retry 3 --retry-all-errors -C - \
  https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz \
  -o inputs/cifar10/cifar-10-binary.tar.gz \
  --write-out "%{url_effective}\n" >> README.md
```

Download sources:

- http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
- http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
- http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
- http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
- https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz
