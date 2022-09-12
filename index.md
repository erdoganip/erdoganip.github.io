<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-TLK47QPQQP"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-TLK47QPQQP');
</script>

## Portfolio

---

### Pattern Recognition
[Expectation Maximization](/expectation_maximization)
<br/>
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1yFvVDUvC9DnX8tVPWCbFdOVhbv_p6XYo?usp=sharing)
<br/>
In this project, I implemented Expectation Maximization algorithm as a solution to cluster a dataset which is a Gaussian Mixture model, includes three different Gaussian distribution.
<br/>
<img src="images/pr_hw2_final.png" width="50%" height="50%"/>

---
[Logistic Regression and Gradient Descent from Scratch](/log_res)
<br/>
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1tppXFSwrH6FGpr4SyDZq87c1143Hchi5?usp=sharing)
<br/>
This project is about implementing a logistic regression model by scratch and updating its gradients using stochastic gradient descent method.
<br/>
<img src="images/pr3_hw_accuracies.jpg" width="50%" height="50%"/>

---

### Computer Vision
[Image Classification Using Traditional CV Methods](/computer_vision)
<br/>
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1q-qOlJVMTtxnwDGHX8TCQUZ7zYPww_Zk?usp=sharing)
<br/>
In this project, the aim is to classify a set of images using various methods. The pipeline consists of 4 steps: Feature Extraction, Finding Dictionary Centers, Feature Quantization and Classification. (In my pipeline, I used SIFT (OpenCV implementation), K-Means Algorithm (my implementation), Bag of Visual Words (my teammate's implementation) and Random Forest (Sklearn implementation) respectively.) For training and testing, we used “Caltech20” dataset provided by TAs.
<br>

---

### Deep Learning

[MLP as a Neural Language Model](/mlp_language)
<br/>
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1cGu0hQV7VCacA_hB8RMrfe5i83WBRHys?usp=sharing)
<br/>
In this project, I implemented a neural language model using a multi-layer perceptron. This network receives 3 consecutive words as the input and predicts the next word.
<br/>
<img src="images/mlp.png" width="50%" height="50%"/>

---
[Convolutional Neural Network(CNN) from Scratch](/cnn_from_scratch)
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1d2LrO9w4i4dv6gweiRoekQ6baSQJwzbl?usp=sharing)
<br/>
I implemented a convolutional neural network (CNN) architecture from scratch, using Pytorch. Tried different data augmentation and optimization techniques and to boost the performance on CIFAR10 dataset.
<br/>
<img src="images/model9.png" width="25%" height="25%"/>
<img src="images/cifar10.png" width="60%" height="60%"/>
<p style="font-size:8px;">Source for CIFAR10 dataset examples: https://www.cs.toronto.edu/~kriz/cifar.html</p>

---
[Spatio-Temporal Attention for Manipulation Failure Detection (Bachelor's Thesis)](/pdf/poster_corrected.pdf)

---
[Implementing a Variational Auto Encoder(VAE)]
The aim of this project is to implement a VAE, where the encoder is an LSTM network and the decoder is a convolutional network. Training and testing was made on MNIST dataset.
<br/>

<p style="font-size:11px">Page template forked from <a href="https://github.com/evanca/quick-portfolio">evanca</a></p>
<!-- Remove above link if you don't want to attibute -->
{% if site.github.is_project_page %}
<p>This project is maintained by <a href="{{ site.github.owner_url }}">{{ site.github.owner_name }}</a></p>
{% endif %}
<p><small>Hosted on GitHub Pages &mdash; Theme by <a href="https://github.com/orderedlist">orderedlist</a></small></p>
