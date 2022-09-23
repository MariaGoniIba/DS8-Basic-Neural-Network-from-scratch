# Data Science (DS) project 8: Building a basic Neural Network from scratch using NumPy

# Credit
The dataset and proposal of the exercise is from the Udemy course [The Data Science Course 2022: Complete Data Science Bootcamp](https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/). I highly recommend this course for those learning Python and machine learning.

# Some notes about the script
Let's remember that the linear model equation is: 

<p align="center">
$y = xw + b$
</p>

Then, the algorithm will try to find the weights 'w' and biases'b' so the outputs 'y' are closest to the targets. 

Remember that when we perform the gradient descent (figure) we must start from an arbitrary number. 
<p align="center">
    <img width="400" src="https://github.com/MariaGoniIba/DS8-Basic-Neural-Network-from-scratch/blob/main/gradientdescent.png">
</p>
We initialize in this case the variable init_range to 0.1. This way, our initial weights and biases will be picked randomly from the interval [-0.1, 0.1]. Be careful when setting up this value. High initial ranges may prevent the machine learning algorithm from learning.

```
init_range = 0.1
```

Then, we decide a learning rate, known as $\eta$. We choose 0.2. This learning rate useful for this demonstration, but it is always good to play around with it and observe how different $\eta$ affect the speed of optimization. 
This value will affect the number of iterations in the following step. Generally, a lower learning rate would need more iterations, while a higher learning rate would need less iterations. Just bear in mind that a high learning rate may cause the loss to diverge to infinity, instead of converge to 0.

```
learning_rate = 0.02
```

Let's train the model. This is an iterative problem. Then we must create a loop where for each iteration we:

1. Calculate outputs
2. Compare outputs to targets through the loss
3. Print the loss
4. Adjust weights and biases

The deltas are the difference between the outputs and the targets. 

Then we will use the L2-norm loss/2 to get the rule from the gradient descent. We divide between observations (loss/observations) to calculate the average loss. Division by a constant doesn't change the logic of the loss, as it is still lower for higher accuracy. Remember that the L2-norm loss formula is:

<p align="center">
$L2-norm = \sum_{i} (y_{i}-t_{i})^2$
</p>

We print the loss at each step, as we want to keep an eye on whether it is decreasing. 

```
loss = np.sum(deltas ** 2) / 2 / observations
print (loss)
```

Then, we update the weights and biases for the next iteration. For weights, we will follow the gradient descent logic:

<p align="center">
$w_{i+1} = w_{i} - \eta\sum_{i} x_{i}\delta_{i}$ 
</p>

and the biases according to:

<p align="center">
$b_{i+1} = b_{i} - \eta\sum_{i} \delta_{i}$ 
</p>

resulting in

```
weights = weights - learning_rate * np.dot(inputs.T,deltas_scaled)
biases = biases - learning_rate * np.sum(deltas_scaled)
```

# Further modifications
To better understand how different parameters affect this exercise, you can play around trying different:
1. Number of observations
2. Learning rate
3. Number of iterations
4. Initial range for initializing weights and biases
