# colorFix
Using GAN to fix color in images. Implemented in Keras.

Project was made during free time on Thanksgiving. I had not much experience in ML.

Given an image, we can adjust the brightness and contrast by simply using the equation: Image = Image * alpha + beta (pixel-wise operation)
Then, reconstruct the initial image using Image = Image *(1/alpha) - beta/alpha. Of course, since Image are limited from 0 to 255, certain feature would be lost.
In this project, I attempt to fix the lost using GAN.

Discriminator:
I simply downsampling images, with a final FCN with sigmoid to check if the image got "good" coloring or not.
A simple MSE loss function.

Generator:
An U-net. I downsample the original image (256,256) to (16,16) and upscale it (default upscale2d in keras), then add the layers together.
I believe the original U-net used Crop and Concat, but I used add because I like it. No padding as it would create strange color striped border in my experience.
For loss function, I used SSIM (or D-SSIM: (1-SSIM)/2) with a L1 loss.


Result:
I have just trained for about 10 epoches. The picture looks like it is becoming better. I will update the final result when available.
Many improvement could be done with the Generator, for example using Sub-pixel convolution instead of normal upscaling, better loss function and training scheme.
Any suggestions and help is appreciated, as I am just starting. Feel free to do PR.

Plans:
I plan to finish my final exam before this. Any help would be appreciated.
