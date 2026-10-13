# Week 6

Use uv for Python commands and CPU by default. Save figures in `evidence/`
and raw runs with their settings in `artifacts/`. Downloads in `inputs/`
and raw runs are ignored by git.

## Download MNIST and CIFAR-10

Use the [Amazon S3 MNIST mirror used by torchvision](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py)
and the [University of Toronto CIFAR-10 download](https://www.cs.toronto.edu/~kriz/cifar.html).

Run from `week6/` in bash or zsh:

```sh
mkdir -p inputs/mnist inputs/cifar10
mnist_url=https://ossci-datasets.s3.amazonaws.com/mnist
for split in train t10k; do
  for kind in images-idx3-ubyte labels-idx1-ubyte; do
    name="$split-$kind.gz"
    curl -fL "$mnist_url/$name" -o "inputs/mnist/$name" \
      --write-out "%{url_effective}\n"
  done
done >> README.md
curl -fL https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz \
  -o inputs/cifar10/cifar-10-binary.tar.gz \
  --write-out "%{url_effective}\n" >> README.md
```

Download sources:

- https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
- https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
- https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
- https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
- https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz
