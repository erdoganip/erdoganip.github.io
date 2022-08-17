<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-TLK47QPQQP"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-TLK47QPQQP');
</script>
## This can be your internal website page / project page

**Project description:** 
Expectation-Maximization [1] is a clustering algorithm which has an iterative approach. It basically computes maximum likelihood estimates iteratively and updates the distribution parameters (π, Σ, μ) according to this likelihood information. This update can be implemented according to the log-likelihood and log-posterior [2].

In this project, I implemented this solution to cluster a dataset which is a Gaussian Mixture model, includes 3 different Gaussian distribution.

### 1. Finding the best parameters
We are trying to find best parameters theta (π, Σ, μ) to maximize the log-likelihood of each data to appropriate distribution (here we can think distribution as clusters) [3]. There is a paradox here. Since we don’t know the distributions from the beginning, we can not calculate the likelihood. Since we can not calculate the likelihood, we can not maximize it and find the appropriate Gaussian distributions of dataset.

To break this loop and start calculation, we determine random inital parameters for 3 Gaussian distributions: (π0, π1, π2, Σ0, Σ1, Σ2, μ0, μ1, μ2)


```javascript
def random_sigma(n):
    x = np.random.normal(0, 1, size=(n, n))
    return np.dot(x, x.transpose())
def initialize_random_params():
    a = np.random.uniform(0, 1)
    b = np.random.uniform(0, 1)
    c = np.random.uniform(0, 1)
    sum_ = a + b + c
    params = {'phi0': a/sum_,
              'phi1': b/sum_,
              'phi2': c/sum_,
              'mu0': np.random.normal(0, 1, size=(2,)),
              'mu1': np.random.normal(0, 1, size=(2,)),
              'mu2': np.random.normal(0, 1, size=(2,)),
              'sigma0': random_sigma(2),
              'sigma1': random_sigma(2),
              'sigma2': random_sigma(2)}
    return params
```

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
