# Adversarial Distributional Training

This repository contains the code for adversarial distributional training (ADT) of our submission: *Adversarial Distributional Training for Robust Deep Learning*, to NeurIPS 2020.

<img src="algos_adt.pdf">

Figure 1: An illustration of three different ADT methods, including (a) ADT<sub>EXP</sub>; (b) ADT<sub>EXP-AM</sub>; (c) ADT<sub>IMP-AM</sub>.



## Prerequisites
* Python (3.6.8)
* Pytorch (1.3.0)
* torchvision (0.4.1)
* numpy

## Training

We have proposed three different methods for ADT. The command for each training method is specified below.

### Training ADT<sub>EXP</sub>

```
python adt_exp.py --model-dir adt-exp --dataset cifar10 (or cifar100/svhn)
```

### Training ADT<sub>EXP-AM</sub>

```
python adt_expam.py --model-dir adt-expam --dataset cifar10 (or cifar100/svhn)
```

### Training ADT<sub>IMP-AM</sub>

```
python adt_impam.py --model-dir adt-impam --dataset cifar10 (or cifar100/svhn)
```

The checkpoints will be saved at each model folder.

## Evaluation

### Evaluation under White-box Attacks

- For FGSM attack, run

```
python evaluate_attacks.py --model-path ${MODEL-PATH} --attack-method FGSM --dataset cifar10 (or cifar100/svhn)
```

- For PGD attack, run

```
python evaluate_attacks.py --model-path ${MODEL-PATH} --attack-method PGD --num-steps 20 (or 100) --dataset cifar10 (or cifar100/svhn)
```

- For MIM attack, run

```
python evaluate_attacks.py --model-path ${MODEL-PATH} --attack-method MIM --num-steps 20 --dataset cifar10 (or cifar100/svhn)
```

- For C&W attack, run

```
python evaluate_attacks.py --model-path ${MODEL-PATH} --attack-method CW --num-steps 30 --dataset cifar10 (or cifar100/svhn)
```

- For FeaAttack, run

```
python feature_attack.py --model-path ${MODEL-PATH} --dataset cifar10 (or cifar100/svhn)
```


### Evaluation under Transfer-based Black-box Attacks

First change the `--white-box-attack` argument in `evaluate_attacks.py` to `False`. Then run

```
python evaluate_attacks.py --source-model-path ${SOURCE-MODEL-PATH} --target-model-path ${TARGET-MODEL-PATH} --attack-method PGD (or MIM)
```

### Evaluation under SPSA

```
python spsa.py --model-path ${MODEL-PATH} --samples_per_draw 256 (or 512/1024/2048)
```

## Pretrained Models

We will release the pretrained models after the review process. It's easy to reproduce the results in our paper, as ADT<sub>EXP-AM</sub> and ADT<sub>IMP-AM</sub> need less than one GPU day to finish training.