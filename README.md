# Assignment #1: Adversarial Attacks on Neural Networks

**Course:** Trustworthy Artificial Intelligence  
**Task:** Implementation of Targeted/Untargeted FGSM and PGD attacks on MNIST and CIFAR-10.

## Repository Structure
* `test.py`: Implementation of adversarial attack functions and model architectures. The main execution script that loads data, trains models, performs attacks, and saves visualized images.
* `requirements.txt`: A list of external dependency packages required to run the project.
* `results/`: A directory where the original images, adversarial images, and 10x magnified perturbation visualization results are saved (automatically generated when `test.py` is executed). And terminal results image file is added.
* `report.pdf`: Result analysis report.
---

## Environment Setup & Requirements

This project was tested and optimized in a `Python 3.11` and `CUDA 12.2` server environment.

### Conda Virtual Environment Creation and Activation

```bash
conda create -n trustworthy_ai python=3.11
conda activate trustworthy_ai
```