# Image-Colorization
The goal of this project was to study how a grayscale image can be colorized using a
convolutional neural network (CNN). The method presented in the paper Colorful
Image Colorization, written by R. Zhang et al., was used as reference [[1]](https://arxiv.org/abs/1603.08511). The
data used to train the model presented in this paper is confined to portrait images
of people’s faces. The dataset used in this study is thus more narrow than what
is used in the original paper, which is a wide variety of image categories. Two
different loss functions are implemented with results showing that a multinomial
cross-entropy loss function produces much better results than a euclidian loss
function. Although it has been trained for far fewer iterations, the model presented
in this paper colorizes portrait images more realistically in 58% of the cases when
compared to the model presented by R. Zhang et al. in [[1]](https://arxiv.org/abs/1603.08511) in a qualitative study.

See below for some examples of results of images colored by our model compared against the ground truth:

![results](https://user-images.githubusercontent.com/32018604/120300610-62b3ce80-c2cc-11eb-9f48-ba8cc9896350.png)

