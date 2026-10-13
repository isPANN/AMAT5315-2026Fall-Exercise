# Week 6

Use uv for Python commands and CPU by default. Save figures in `evidence/`
and raw runs with their settings in `artifacts/`. Downloads in `inputs/`
and raw runs are ignored by git.

## Download MNIST and CIFAR-10 in mainland China

Use the Huawei Cloud mirrors linked in the
[MindSpore dataset documentation](https://www.mindspore.cn/tutorials/en/r1.5/dataset.html).
MNIST is distributed as a ZIP of uncompressed IDX files; the command below
repackages them into the four original `.gz` filenames. CIFAR-10 remains
the binary TAR archive.

Run from `week6/` in bash or zsh:

```sh
set -eu
mkdir -p inputs/mnist inputs/cifar10
base=https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets

curl -fL "$base/MNIST_Data.zip" -o inputs/mnist/MNIST_Data.zip.part
uv run --no-project python - <<'PY'
import gzip
from pathlib import Path
from zipfile import ZipFile

root = Path("inputs/mnist")
with ZipFile(root / "MNIST_Data.zip.part") as archive:
    for directory, split in [("train", "train"), ("test", "t10k")]:
        for kind in ["images-idx3-ubyte", "labels-idx1-ubyte"]:
            name = f"{split}-{kind}"
            data = archive.read(f"MNIST_Data/{directory}/{name}")
            output = root / f"{name}.gz.part"
            output.write_bytes(gzip.compress(data, mtime=0))
            output.replace(root / f"{name}.gz")
(root / "MNIST_Data.zip.part").replace(root / "MNIST_Data.zip")
PY

curl -fL "$base/cifar-10-binary.tar.gz" -o inputs/cifar10/cifar-10-binary.tar.gz.part
gzip -t inputs/cifar10/cifar-10-binary.tar.gz.part
mv inputs/cifar10/cifar-10-binary.tar.gz.part inputs/cifar10/cifar-10-binary.tar.gz
```

Download sources:

- https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
- https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz

Original MNIST source used for the existing files:

- https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
- https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
- https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
- https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz

Original CIFAR-10 source:

- https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz
