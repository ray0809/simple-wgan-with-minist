# simple-wgan-with-minist

The original gan code is written by  https://github.com/Zackory/Keras-MNIST-GAN

Here I change the loss and label for implementing wgan 

I don't clip the weight in discriminator(which caused a bad result in my test), it can also have a good result.

GAN：iterate 201 epoches

![image](https://github.com/ray0809/simple-wgan-with-minist/blob/master/gan_generated_img_epoch_201.png)

WGAN:iterate 200 epoches

![image](https://github.com/ray0809/simple-wgan-with-minist/blob/master/wgan_generated_img_epoch_200.png)


```math
\left (\frac{1}{\left | X_{k} \right |}\sum_{x\in X_{k}}^{n} x^{p_{k}} \right )^{\frac{1}{p_{k}}}
```
