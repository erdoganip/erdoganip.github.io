## Convolutional Neural Network(CNN) from Scratch

In this project, I implemented a three-layered convolutional neural network (CNN) (that was the constrained) architecture using a deep learning library (PyTorch). For the training and testing phases, I used CIFAR10 dataset.

For the full pipeline: [![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1d2LrO9w4i4dv6gweiRoekQ6baSQJwzbl?usp=sharing)
### 1. Suggest hypotheses about the causes of observed phenomena
I have started with a base model, which has three convolutional layers followed by max-pooling layers and there were 2 fully connected layers at the end. This model made me start with 63% accuracy in both training and testing. Then I tried to change the kernel sizes of these convolutional layers (Model 2) but it didn’t effected results that much. I decided to add Batch Normalization between the convolutional layers and activation functions. This resulted with improvement in training and testing accuracies.

Then I made data augmentation with ”Random Cropping” yet it decreased model’s performance. That’s probably because there were no overfitting, the model haven’t learn well yet. Improving generalization ended up with underfitting. Also, at this point, I have realized that adding max pooling layer after all of the convolutional layers may cause underfitting, too.

### 2. Assess assumptions on which statistical inference will be based

```javascript
if (isAwesome){
  return true
}
```

### 3. Support the selection of appropriate statistical tools and techniques

<img src="images/dummy_thumbnail.jpg?raw=true"/>

### 4. Provide a basis for further data collection through surveys or experiments

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
