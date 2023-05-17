# Operation-Lever Early Stopping (OLES)

Anonymous open-source code of  "Operation-Level Early Stopping for Robustifying Differentiable NAS".

We just make few modifications on the original DARTS code.

## Search and Evaluate

### CIFAR-10

```
python train_search.py
```

### CIFAR-100

```
python train_search.py --cifar100
```

### ImageNet

```
python train_imagenet.py --auxiliary
```