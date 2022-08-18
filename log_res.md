<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-TLK47QPQQP"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-TLK47QPQQP');
</script>
## Logistic Regression and Gradient Descent from Scratch
In this project, I implemented a logistic regression model from scratch and update its gradients by following stochastic gradient descent approach.

For the full pipeline: [![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1tppXFSwrH6FGpr4SyDZq87c1143Hchi5?usp=sharing)
### Initialization

The dataset has 2 features. When I checked the training labels, I saw labels were either -1 or 1. So, I decided to use tanh as the scaler function, instead of the σ. So, in this problem:
<img src="images/log1.jpg" width="50%" height="50%"/>
<img src="images/log2.jpg" width="50%" height="50%"/>
<img src="images/log3.jpg" width="50%" height="50%"/>
and my loss function is (since I am using tanh, I choose this loss function. It gaves negative or zero into the logarithm if I use other popular logistic loss function):
<img src="images/log4.jpg" width="80%" height="80%"/>
### Training
For the training part, I started with random initialized weights w1, w2, b. For each epoch, i traversed in the shuffled dataset (I shuffled the indices of the dataset actually, not the dataset itself.) and in each iteration in the epoch, for each data point I randomly chose, I updated the weights. To take the partial derivative of the loss function and update the related weights (partial derivative of loss function according to w1 to update w1, etc.), I used chain rule.
<img src="images/log5.jpg" width="50%" height="50%"/>
for w1, the gradient is
<img src="images/log6.jpg" width="50%" height="50%"/>
for w2, the gradient is
<img src="images/log7.jpg" width="50%" height="50%"/>
and for b, gradient is
<img src="images/log8.jpg" width="50%" height="50%"/>
At the end of the each iteration, I updated the weights and the bias by the following formula:
<img src="images/log9.jpg" width="50%" height="50%"/>
Here, α is the learning rate. I tried my model with different learning rates. I have observed that, when I select learning rate high, the steps get bigger and more rapid changes can be observed from the accuracy graphs.

### 3. Support the selection of appropriate statistical tools and techniques

<img src="images/dummy_thumbnail.jpg?raw=true"/>

### 4. Provide a basis for further data collection through surveys or experiments

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
