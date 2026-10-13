# Week 6

Use uv for Python commands and CPU by default. Save figures in `evidence/`
and raw runs with their settings in `artifacts/`. Downloads in `inputs/`
and raw runs are ignored by git.

## Download MNIST and CIFAR-10

Use the [CVDF MNIST mirror](https://github.com/cvdfoundation/mnist)
and the [MindSpore CIFAR-10 mirror](https://www.mindspore.cn/tutorials/en/r1.5/dataset.html).

Run from `week6/` in bash or zsh:

```sh
mkdir -p inputs/mnist inputs/cifar10
mnist_url=https://storage.googleapis.com/cvdf-datasets/mnist
for split in train t10k; do
  for kind in images-idx3-ubyte labels-idx1-ubyte; do
    name="$split-$kind.gz"
    curl -fL "$mnist_url/$name" -o "inputs/mnist/$name" \
      --write-out "%{url_effective}\n"
  done
done >> README.md
cifar_host=https://mindspore-website.obs.cn-north-4.myhuaweicloud.com
curl -fL "$cifar_host/notebook/datasets/cifar-10-binary.tar.gz" \
  -o inputs/cifar10/cifar-10-binary.tar.gz \
  --write-out "%{url_effective}\n" >> README.md
```

Download sources:

- https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
- https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
- https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
- https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
- https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz
