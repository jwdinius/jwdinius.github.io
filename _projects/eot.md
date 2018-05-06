---
layout: page
title: Extended Object Tracking
description: Tracking an ellipse
img: /assets/img/eot_tn.png
use_math: true
---

[GitHub](https://github.com/jwdinius/extended-object-tracking)

[Original Reference](https://arxiv.org/pdf/1604.00219.pdf)

## Abstract
I built a multithreaded C++ application using the JUCE API to model and track extended objects.  In this page, I will discuss the mathematical details, along with how I approached the implementation.

## Introduction
Late last year, while I was working on a professional problem tracking extended objects, I came across [this paper](https://arxiv.org/pdf/1604.00219.pdf) and [some Matlab code](https://github.com/Fusion-Goettingen/ExtendedObjectTracking/tree/master/MEM_EKF) by the authors.  I ended up pursuing a different approach than the one outlined in the paper, mostly due to differences between the problem the authors had considered and my problem, but I told myself that when I had time, I would dig deep into the paper and understand how the authors' Matlab implementation worked.  In addition to understanding the algorithm, I wanted to use what I had been learning about multithreaded embedded system applications using [JUCE](www.juce.com).

In this project post, I will discuss how I worked through the problem of building a multithreaded application that implemented the extended object tracking algorithm discussed in the previously referenced sources.  I will begin with the final output of the algorithm:

![Imgur](https://i.imgur.com/TPGR8lg.gif)

The setup for the above GIF is:
- A ground-truth elliptical object, in white, moves at near constant velocity except when undergoing turns
- Returns off the surface of the object, shown as white squares, are processed by a modified Kalman filter to generate the state and shape/orientation estimate shown in green.

In the remainder of this post, I will go through the relevant details of the implementation, starting with the mathematics.

## Mathematical Details
For the complete discussion, see the paper.  I will discuss only the major points of the paper in the subequent sections.

### Generating Ground Truth
The ground truth object is an ellipse that travels at constant speed.  The object undergoes 4 counterclockwise turns during the simulation.  The object state and orientation are modeled as explicitly time-dependent:

$$
\dot x = v \cos \theta \\
\dot y = v \sin \theta \\
\theta = \begin{cases} 
      -\frac{\pi}{4} & 0 \leq t \leq t_1 \\
      -\frac{\pi}{4} + \frac{\pi}{4(t_2-t_1)} (t-t_1) & t_1 < t \leq t_2 \\
      0 & t_2 < t \leq t_3 \\
      \frac{\pi}{2(t_4-t_3)} (t-t_3) & t_3 < t \leq t_4 \\
      \frac{\pi}{2} & t_4 < t \leq t_5 \\
      \frac{\pi}{2} + \frac{\pi}{2(t_6-t_5)} (t-t_5) & t_5 < t \leq t_6 \\
      \pi & t > t_6
   \end{cases} \\
\dot v = 0
$$

The object moves with it's velocity along it's major axis.

### Modeling Sensor Measurements
Measurements are modeled as point returns.  At each timestep, $k$, the set of all measurements has the form:

$$
Y_k = \{ \mathbf{y}_{k,i} \}_{i=1}^{n_k},
$$

where the term $n_k$ is the number of measurements at time $k$, which follows a Poisson distribution.  Each measurement $y_{k,i}$ is written as:

$$
\mathbf{y}_{k,i} = \mathbf{m}_k + h_{k,i}^1 l_1 \begin{pmatrix} 
\cos \theta_k \\
\sin \theta_k 
\end{pmatrix} + h_{k,i}^2 l_2 \begin{pmatrix} 
-\sin \theta_k \\
\cos \theta_k
\end{pmatrix} + \mathbf{v}_{k,i}
$$

The noise vectors $$\mathbf{h}_{k,i}$$ and $$\mathbf{v}_{k,i}$$ are clearly seen as multiplicative and additive, respectively.  For the specific noise levels used, check the paper and the subsequent implementation.

In laymen's terms:  the values used for measurements are drawn from a multivariate normal distribution about the object's center.

### Tracking an Extended Object

#### Prediction Step
The kinematic state, $\mathbf{r}$, and covariance, $C^r$, are assumed to propagate according to a nearly-constant velocity model:

$$
\mathbf{r}_{k|k-1} =  A_r \mathbf{r}_{k-1|k-1} \\
A_r = \begin{pmatrix} 
1 \ 0 \ \Delta t \ 0 \\
0 \ 1 \ 0 \ \Delta t \\
0 \ 0 \ 1 \ 0 \\
0 \ 0 \ 0 \ 1
\end{pmatrix} \\
C^r_{k|k-1} = A_r C^r_{k-1|k-1} A_r^T + C^r_w,
$$

where $C^r_w$ is the state process covariance matrix.

For the shape and orientation process updates

$$
\mathbf{p}_{k|k-1} =  A_p \mathbf{p}_{k-1|k-1} \\
A_r = \begin{pmatrix} 
1 \ 0 \ 0 \\
0 \ 1 \ 0 \\
0 \ 0 \ 1
\end{pmatrix} \\
C^p_{k|k-1} = A_p C^p_{k-1|k-1} A_p^T + C^p_w,
$$

where $C^p_w$ is the shape/orientation process covariance matrix.

#### Update Step
The update step revolves around first- and second-order moment estimation using a *quadratic* measurement, a 2-fold Kronecker product, to augment the existing linear measurement.  The choice of this measurement model is due to the fact that there are not enough correlations between the state vector and the measurements, as the paper authors state.  The total measurement vector combining the linear and quadratic measurements will be denoted as $\mathbf{z}_{k,i}$.  Using this measurement model, the resulting update equations are:

$$
\mathbf{r}_{k|k} = \mathbf{r}_{k|k-1} + M^r_k (S^r_k)^{-1} (\mathbf{z}_{k,i} - H \mathbf{r}_{k|k-1}) \\
C^r_{k|k} = C^r_{k|k-1} - M^r_k (S^r_k)^{-1} (M^r_k)^T
$$

where the matrix $H$ is the measurement Jacobian,

$$
H = \begin{pmatrix} 
1 \ 0 \ 0 \ 0 \\
0 \ 1 \ 0 \ 0
\end{pmatrix}
$$

and the covariance matrices $M^r_k$ and $S^r_k$ are the measurement-to-state and measurement-to-measurement covariance matrices.

The update equations for the shape/orientation parameters have the same form, but have different covariance matrices.

## Implementation
The implementation uses two third-party software packages: JUCE and Eigen.

### JUCE
JUCE is a GUI-builder that is usually used for audio applications but it happens to be useful for other embedded system applications as well.  In many ways, JUCE is similar to QT: it has a publish/subscribe messaging protocol for passing data across multiple threads, as well as a graphical front-end that can be used to plot output in real-time.  Both of the aforementioned features were used in completing this project.

JUCE exports a project file that can be used to build an executable.  For this project, I chose Xcode as the export target.  Xcode's built-in debugger was very useful throughout the whole project, though it was a bit difficult at times because the program is multithreaded.

### Eigen
[Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) is an API for building and operating on matrix and vector data types.  Operator-overloading makes matrix operations a breeze.  There is also built-in support for computationally-intensive operations like matrix decompositions, eigenvector computation, and matrix inversion.  Eigen outperforms similar linear algebra packages; check the website for benchmarks.

### Building the Application

#### Modeling measurements
I created a timer thread that executes at a fixed rate.  Once a full cycle of the timer thread executes, a new measurement comprised of a set of $n$ points, where $n$ is drawn from a Poisson distribution.  Each of the $n$ points are drawn from a multivariate Gaussian distribution.  For all random number calls, the [C++ standard library](http://www.cplusplus.com/reference/random/) was used.  After a new measurement set is generated, the sensor thread sends a change message to notify the main process to execute it's `run` method.

#### Kalman filter
The use of the Eigen API made the Kalman filter estimation straight-forward and relatively simple.  Due to the overloading of operators, such as `*` and `+`, matrix multiplication and addition resembles very closely the corresponding operations in Matlab, which the paper authors used for their original implementation.

Only limited a priori knowledge regarding shape of the object is assumed; I initially assume that the object is circular with radius equal to the larger of the two axes of the true extended object.  This makes sense from an operational standpoint; in the absence of more information, filters are initialized using heuristic assumptions about the objects-to-track, e.g. cars in this case.  The position of the object's center is set to the mean of the first set of measurements and the velocity is set to 0, since we cannot observe velocity, directly or indirectly.  After the filter has been initialized, the processing cycle follows the normal schema for Kalman filters: predict to current time, and then correct the predicted estimate with the latest measurement.

#### Putting it all together
I was able to use a pre-existing GUI application: plot window + background + run button.  The plot window uses draw utilities from the JUCE API to plot the measurements, ground truth, and the estimated extended object.  The run button creates pointers to new sensor and Kalman filter objects, which are then executed to update the plot window.

The main loop executes at 1Hz because of the sleep thread that signals when to generate new sensor data.  Consequently, new data is plotted at roughly 1Hz, as well.

### Challenges
The biggest challenge of this project, as in so many estimation tasks, was selecting proper noise terms.  The algorithm is sensitive to the choice of noise terms.  However, when I reviewed [a different reference](https://www.amazon.com/Estimation-Applications-Tracking-Navigation-Bar-Shalom/dp/047141655X), I was able to find a suitable noise model using a single parameter for the acceleration noise.

With regards to the measurement modeling: C++ does not have built-in support for generating multivariate Gaussian random numbers, so I had to write my own utility to do it.  With Eigen, this was no big deal; literally, this was a one-liner.

One final note regarding what I found mildly frustrating: different compilers initialize allocated memory differently!  This has caused me heartburn from time-to-time, but I have become spoiled because my compiler of choice, g++, zeros out the memory when it allocates.  JUCE + Xcode does **not**!  Thankfully, the results I was observing were not repeatable and were consistent with improper initialization.  From now on, I will explicitly zero out memory at initialization to avoid problems in the future.

## Thanks for reading!