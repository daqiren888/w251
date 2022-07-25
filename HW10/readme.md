### What changes did you make to the notebook[s]? Did your changes result in lower losses?

1. Use the decoder weights of the VAE to initialize the generator. First train a VAE, because the VAE can ensure that the output is as similar to the input as possible during the training process. After training, the decoder is taken out as the original generator. At this time, the generator already has a certain ability to fake. On the basis of this generation network, combined with The real sample trains the discriminant network separately. When the discriminant network has a certain ability to discriminate, then let the two fight each other, which greatly improves the efficiency and success rate of GAN training.
2. Soft labels and labels with noise: You can add random noise of (-0.2, 0) to label 1 of real samples, and add random noise of (0, 0.2) to label 0 of fake samples. The addition of soft labels and noisy labels can solve the problem of gradient disappearance in the generative network to a certain extent.
3. Reverse the label (Real=False, fake=True): Sometimes reversing the label will help training. The discriminant network also relies on the BP algorithm to update the weight, and the weight update may encounter relative extremes Point and stop, and reverse the label can break this deadlock.
4. Do not use early stopping: In the early training process of GAN, both the generation network and the discriminant network are in an unstable state. The training time of GAN is very long, and the initial small loss value and generated samples are almost impossible. Show any trends and progress. It is still worth waiting a while before ending the training process and adjusting the settings.
5. Monitor gradient changes: Monitor gradient and loss changes in the network. Ideally, a generative network should accept large gradients early in training, as it needs to learn how to generate realistic-looking data. On the other hand, a discriminative network should not always accept large gradients early in training, as it can easily distinguish between real and generated data.
6. Adjusted network structure: Choose a common or more complex network according to the application, consider the network capacity, and the activation function, use sigmoid, or tanh, or convolution, etc.


### In your own words, how does the Discriminator improve its ability to detect fakes?

The growth of the discriminator can theoretically be improved gradually, because the training of the discriminant network is always fed to it alternately with true and false samples, and let it know the true and false, which is equivalent to supervised learning,
After the discriminant network can be used, let it assist the generation network. In the worst case, the fake samples made by the generation network are too fake, which makes the discriminant network learn slowly. Therefore, the best case is that the generation network has good performance and can create Not bad fake samples, of great value to the discriminator.

### Share a copy of the output image from the last step in 
