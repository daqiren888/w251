### What changes did you make to the notebook[s]? Did your changes result in lower losses?

1) Initialize the generator with the decoder weights of the VAE.
2) Soft labels and noisy labels: You can add random noise of (-0.2, 0) to label 1 of the real sample, and random noise of (0, 0.2) to label 0 of the fake sample. The addition of soft labels and noise labels can solve the problem of gradient disappearance in the generative network to a certain extent.
3) Not using early stopping: GANs take a long time to train and it is still worth waiting a while before ending the training process and adjusting the settings.
4) Monitor Gradient Changes: Monitor gradient and loss changes in the network.
5) Adjust the network structure: consider the network capacity and activation function, use sigmoid, or tanh, or convolution, etc.


### In your own words, how does the Discriminator improve its ability to detect fakes?

The growth of the discriminator can theoretically be improved gradually, because the training of the discriminant network is always fed to it alternately with true and false samples, and let it know the true and false, which is equivalent to supervised learning,
After the discriminant network can be used, let it assist the generation network. In the worst case, the fake samples made by the generation network are too fake, which makes the discriminant network learn slowly. Therefore, the best case is that the generation network has good performance and can create Not bad fake samples, of great value to the discriminator.

### Share a copy of the output image from the last step in 

<img src="./outputimages/.png" width="497"/>
<img src="./outputimages/.png" width="498"/>

