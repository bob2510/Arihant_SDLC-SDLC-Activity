# Deep-iterative reconstruction phase retrieval

Developed a DNN-HIO iterative system that retrieves the phase of the image by using the amplitude of the fourier transform of the image

Image reconstruction from a diffraction pattern, as in Coherent Diffraction Imaging (CDI)

This is an implementation of the iterative Deep Neural Network(DNN) - Hybrid input-output (HIO) algorithm, done in python.

#HIO Algorithm

The HIO algorithm works by iteratively transforming an image between real space and Fourier space, applying constraints at each step. 
In this implementation, the real space image is constrained to be positive, real, and required to have compact support (this is valid since the diffraction pattern is taken with oversampling). 
The Fourier constraint is that the intensity of the transformed image must be the measured intensity given as input. 

#U-net Architecture

![Diffract](https://github.com/bob2510/Arihant_SDLC-implementation/blob/96c7ac296ee217002af68dcc8c61831f7a05cca5/6.%20Images/unet.PNG)

#Objective

For example, given a diffraction pattern: 

![Diffract](https://github.com/bob2510/Arihant_SDLC-implementation/blob/96c7ac296ee217002af68dcc8c61831f7a05cca5/6.%20Images/transform.png)

We aim to reconstruct the original image:

![Progress](https://github.com/bob2510/Arihant_SDLC-implementation/blob/96c7ac296ee217002af68dcc8c61831f7a05cca5/6.%20Images/progress.gif)

Citations:

J. R. Fienup, "Phase retrieval algorithms: a comparison," Appl. Opt. 21, 2758-2769 (1982)
Cagatay Isil, "Deep iterative reconstruction for phase retrieval," Vol. 58, No. 20 (2019)
