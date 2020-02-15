---
layout: post
title:  Point Cloud Registration as Optimization, Part Two
description: Towards robust registration using convex relaxation, continued
date:   2020-2-15 10:00:00
use_math: true
comments: true
---

In a [previous post](https://jwdinius.github.io/blog/2019/point-match/), I presented the "best correspondences" problem encountered in point cloud registration.  I restate the problem here to make the remainder of this post more clear for the reader: 


The "best correspondences" problem can be formulated as an optimization problem:

>Find the vector $$\mathbf{x}^*$$ that maximizes the quadratic objective function:
>$$
>\begin{eqnarray}
>f(\mathbf{x}) = \sum_{ijkl} w_{ijkl} x_{in+j} x_{kn+l}
>\end{eqnarray}
>$$
>
>where
>
>$$
>\begin{eqnarray}
>w_{ijkl} = \exp{[-d_{ijkl}]} \\
>d_{ijkl} = \bigl | ||\mathbf{p}_i - \mathbf{p}_k||_2 - ||\mathbf{p}_j - \mathbf{p}_l||_2 \bigr |.
>\end{eqnarray}
>$$

$$\mathbf{p}_i, \mathbf{p}_k$$ are the 3d cartesian coordinates of points $$i$$ and $$k$$ in the source cloud, whereas $$\mathbf{p}_j, \mathbf{p}_l$$ are the 3d cartesian coordinates of points $$j$$ and $$l$$ in the target cloud.

The $$d$$ value above is interpreted as a _pairwise consistency metric_: the smaller the value of $$d$$, the better the correspondence between the pair $$(i, k)$$ in the source cloud to the pair $$(j, l)$$ in the target cloud.  By adding the $$\exp[\cdot]$$ function, stronger correspondences will be weighted _exponentially_ higher than weaker ones, which will favor better matches during the search for the optimal $$\mathbf{x}^*$$.


There are some additional factors to consider; namely, _what are the constraints on $$\mathbf{x}^*$$_?  For one, only one source point can be matched to a target point, and vice-versa.  Moreover, we want _every_ source point to be matched to a target point.  Mathematically, these constraints can be stated as:

$$
\begin{eqnarray}
\sum_{j=0}^{n-1} x_{in+j} = 1  \ \  \forall i \in [0, m)\\
\sum_{j=0}^{n-1} x_{mn+j} = n - m \\
\sum_{i=0}^{m} x_{in+j} = 1 \ \  \forall j \in [0, n)
\end{eqnarray}
$$

where the slack variables $$x_{mn+j}, 0 \le j < n$$ are added to address the target points that are _not_ matched to a source point.

Okay, now that that's out of the way, we can state the purpose of this post:

>Are the constraints, as stated, reasonable for the problem at-hand?

The constraints presented assume that _every_ point in the source point cloud has a corresponding point in the target point cloud.  In practice, this presents an overly restrictive constraint for our matching problem.  What if there are outliers, for instance?  In this post, I will present a novel modification to the optimization problem above to allow for less restrictive, i.e. more robust, matching between point clouds.

## Relaxing the constraints means augmenting the state vector

Let me first describe, in words, what I'd like to achieve:  I want an optimization problem that allows me to identify the subset, of size $$k<m$$, of source points that is best matched to a corresponding subset, also of size $$k$$, of target points.

In the formulation of the original problem, we required $$n$$ slack variables to account for the unmatched $$n-m$$ points in the target point set.  Now, since we are not matching $$m-k$$ points in the source distribution, we can add additional slack variables, $$m$$ of them, to achieve what we ultimately want.  These additional slack variables allow handling of source points that are _not_ matched to a target point.  Formally, the optimization problem, with constraints, is now:

>Find the vector $$\mathbf{x}^* \in \{0, 1\}^{(m+1)(n+1)}$$ that maximizes the quadratic objective function:
>$$
>\begin{eqnarray}
>f(\mathbf{x}) = \sum_{ijkl} w_{ijkl} x_{i(n+1)+j} x_{k(n+1)+l}
>\end{eqnarray}
>$$
>
>where
>
>$$
>\begin{eqnarray}
>w_{ijkl} = \exp{[-d_{ijkl}]} \\
>d_{ijkl} = \bigl | ||\mathbf{p}_i - \mathbf{p}_k||_2 - ||\mathbf{p}_j - \mathbf{p}_l||_2 \bigr |.
>\end{eqnarray}
>$$
>
>with constraints:
>
>$$
>\begin{eqnarray}
>\sum_{j=0}^{n} x_{i(n+1)+j} = 1 \ \  \forall i \in [0, m) \\
>\sum_{j=0}^{n} x_{m(n+1)+j} = n - k \\
>\sum_{i=0}^{m} x_{i(n+1)+j} = 1 \ \ \forall j \in [0, n) \\
>\sum_{i=0}^{m} x_{i(n+1)+n} = m - k \\
>x_{m(n+1)+n} = 0
>\end{eqnarray}
>$$

Let's explain the contraints, one-by-one:

* $$\sum_{j=0}^{n} x_{i(n+1)+j} = 1 \ \  \forall i \in [0, m)$$ means "every source point must be matched to either an actual or an augmented, e.g. slack, target point".
* $$\sum_{j=0}^{n} x_{m(n+1)+j} = n - k$$ means "$$n-k$$ target points must not be associated to some source point".
* $$\sum_{i=0}^{m} x_{i(n+1)+j} = 1 \ \ \forall j \in [0, n)$$ means "every target point must be matched to either an actual or a slack source point".
* $$\sum_{i=0}^{m} x_{i(n+1)+n} = m - k$$ means "$$m-k$$ source points must not be matched to some target point".
* $$x_{m(n+1)+n} = 0$$ means "the slack source point must not be matched to the slack target point".

Lowering the dimension of the state vector $$\mathbf{x}$$ by 1 would effectively remove this last constraint, however I chose to keep it around because, in my opinion, it makes interpreting the other constraints easier.

## Actual implementation (*finally*)
After two posts setting up the problem, I'm ready to present my solution... in the next post.

Stay tuned, and thanks for reading!
