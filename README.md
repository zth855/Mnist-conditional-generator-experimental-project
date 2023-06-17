# Mnist-conditional-generator-experimental-project
# CGAN_PROJECT



## 1. Introduction

This is a project for the course of Artificial Intelligence. The project is to implement a Conditional generative adversarial network（CGAN）and use it to generate images.

It contains these files:

1. `model.py`: used to define model.
2. `train.py`: contains the code to train indivisual model.
3. `aigcmn.py`: contains the code to generate 0-9 images.

We trained using both CPU and CUDA, and the training results were saved respectively in "cgan_generator_cpu.pth" and "cgan_generator_cuda.pth". By default, the "aigcmn.py" script uses the results from CUDA training. You can view the results from CPU training by modifying the following code in "aigcmn.py".

~~~
state_dict = torch.load('cgan_generator_cpu.pth')
~~~



## 2. Run

### train:

```bash
python train.py 
```

### aigcmn  :

```bash
python aigcmn.py
```

### Check results on screen or in results folder.
