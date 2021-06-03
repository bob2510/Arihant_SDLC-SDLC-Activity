# Deep-iterative reconstruction phase retrieval

Developed a DNN-HIO iterative system that retrieves the phase of the image by using the amplitude of the fourier transform of the image
Image reconstruction from a diffraction pattern, as in Coherent Diffraction Imaging (CDI)
This is an implementation of the iterative Deep Neural Network(DNN) - Hybrid input-output (HIO) algorithm, done in python.
![Project](https://github.com/bob2510/Arihant_SDLC-implementation/blob/7f2395ba48e848070d7abb3ba66991274256f98c/5.%20Images/all.PNG)

## HIO Algorithm

The HIO algorithm works by iteratively transforming an image between real space and Fourier space, applying constraints at each step. 
In this implementation, the real space image is constrained to be positive, real, and required to have compact support (this is valid since the diffraction pattern is taken with oversampling). 
The Fourier constraint is that the intensity of the transformed image must be the measured intensity given as input. 

## U-net Architecture

![Diffract](https://github.com/bob2510/Arihant_SDLC-implementation/blob/e4b76d54417a3b15500a018e9639a16bb6ccf634/5.%20Images/unet.PNG)

## Objective

For example, given a diffraction pattern: 

![Diffract](https://github.com/bob2510/Arihant_SDLC-implementation/blob/e4b76d54417a3b15500a018e9639a16bb6ccf634/5.%20Images/transform.png)

We aim to reconstruct the original image:

![Progress](https://github.com/bob2510/Arihant_SDLC-implementation/blob/e4b76d54417a3b15500a018e9639a16bb6ccf634/5.%20Images/progress.gif)

## Citations:

J. R. Fienup, "Phase retrieval algorithms: a comparison," Appl. Opt. 21, 2758-2769 (1982)

Cagatay Isil, "Deep iterative reconstruction for phase retrieval," Vol. 58, No. 20 (2019)

## Folder Structure
Folder             | Description
-------------------| -----------------------------------------
`1. Requirements`   | Documents detailing requirements and research
`2. Architecture`         | Documents specifying design details
`3. Implementation` | All code and documentation
`4. Test plan and Output`      | Documents with test plans and procedures
`4. Images`      | Some images used in this repo


## Credits

1. Reference for Architecture and overall layout - [Sanchana-2k](https://github.com/Sanchana-2k/LTTS_C_MiniProject)
2. Reference requirements and architecture and test plan - [Manjari_AP](https://github.com/256152/Mini_Project_1_April_2021.git)
3. Reference for High Level Requirements and Low Level Requirements Table - [arc-arnob](https://github.com/arc-arnob/LnT_Mini_Project.git)
