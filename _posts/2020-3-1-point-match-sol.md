---
layout: post
title:  Point Cloud Registration as Optimization, Code Implementation
description: C++ implementation of point cloud matching using convex relaxation.
date:   2020-3-1 10:00:00
use_math: true
comments: true
---

tl; dr:  Here's a [link](https://github.com/jwdinius/point-registration-with-relaxation) to the GitHub repo.  The `README.md` is pretty descriptive.  Clone it, fork it, whatever.  You should be able to get up-and-running quickly.

In previous posts, [this one](https://jwdinius.github.io/blog/2019/point-match/) and [this one](https://jwdinius.github.io/blog/2019/point-match-cont/), I set up a quadratic optimization problem for finding the best correspondences between two point sets, also called _clouds_.  If you haven't already seen these, I recommend going back and looking at them before going through the this post.  Your call.

## A bit of context
At last year's CVPR, I sat through a really cool talk presenting the [SDRSAC paper](https://arxiv.org/abs/1904.03483).  The results presented seemed really promising and I wanted to see if I could reproduce them.  There were a lot of issues encountered along the way.  In this post, I will highlight and dig deeper into some of these issues.

# Implementation

## Infrastructure
At this point, I use [Docker](www.docker.com) for all of my personal projects.  I just find that it is a more flexible solution than requiring users to install a bunch of dependencies on their machine.  There's an added bonus as well:  _new users can reproduce original results with little-to-no added friction_.

## Language
I chose C++ because performance was a concern.  I wanted to have the performance of a strongly-typed language combined with the large suite of supporting libraries written in it.  To handle automatic resource management, I compiled using the C++-14 standard.  This allowed transfer of ownership of resources to smart pointers through the use of the `std::make_unique` and `std::make_shared` functions introduced by the C++-14 standard.

### Linear algebra library
Despite [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)'s popularity over the last decade, I decided to go with [Armadillo](http://arma.sourceforge.net/).  The reasons for my choice include:

* Documentation: _the wiki is great_
* [Speed](http://nghiaho.com/?p=1726)
* Matlab-like syntax: _though 0-based indexing of C++ is still used_
* Functionality - _reshaping, resampling, and operations like singular-value decomposition come for free_

### Optimization

#### Nonlinear optimization framework
I had originally wanted to use a semidefinite solver, SDP,  like the original SDRSAC work, but finding such a solver proved to be a major roadblock.  My requirements for the solver were:

* It had to be free - _I wanted to be able to share this work with everyone_
* It had to be well-supported - _I didn't want to spend a lot of time debugging an external_
* It had to have a simple interface - _I wanted to be able to define the optimization objective in a clear, intuitive format_

Some libraries considered were [CSDP](https://github.com/coin-or/Csdp/wiki), [SDPA](http://sdpa.sourceforge.net/), and [Ensmallen](http://ensmallen.org/), however _none_ of these libraries, when evaluated, met the three criteria above.

The choice of semidefinite solver was driven primarily by the structure of the optimization objective, however when looking into some comparisons, like in Section 5.4 of this [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.140.910&rep=rep1&type=pdf), I convinced myself that a more general optimization framework could be effective, so I decided to go with [IPOPT](https://coin-or.github.io/Ipopt/).  IPOPT meets all of the above requirements and, as a bonus, I have used it before in other projects; see [this](https://jwdinius.github.io/blog/2018/udacity_sdcnd/).

The translation of the optimization constraints was also _much_ easier for the IPOPT formulation when compared to the SDP formulation.  Don't take my word for it, though:  compare the constraints presented in the posts referenced above to the ones in the SDRSAC paper.

#### Linear correction
As in the SDRSAC paper, depending upon the convergence criteria imposed on the nonlinear optimizer, the resulting solution to the optimization objective _may not be a valid member of the desired solution space_!  To fix this, I needed to find an implementation of something akin to the [simplex algorithm](http://fourier.eng.hmc.edu/e176/lectures/NM/node32.html) for projecting the solution to the nonlinear problem onto the solution space: 0 for non-matches, 1 for matches.  I was able to find an implementation in Google's [ORTools](https://developers.google.com/optimization/lp/lp) which meets the requirements I outlined above for the nonlinear optimizer above.

### Data creation and performance analysis
I knew that I wanted to be able to easily create datasets, run the optimization, and quickly analyze the results graphically.  The native C++ support for such capabilities is not very flexible, so I decided to wrap the main function call in a Pythonic wrapper using [`Boost::Python`](https://www.boost.org/doc/libs/1_70_0/libs/python/doc/html/index.html).  This allowed me to use the [`numpy`](https://numpy.org/) suite of tools for creating point clouds to be passed to the algorithm and the plotting capabilities of [`matplotlib`](https://matplotlib.org/) to visualize the output.  I found that, by writing the wrapper, it was a lot easier to identify system-level issues while developing the algorithm.

## Testing
Testing is a big part of my approach to application development.  Though I didn't follow the full principles of TDD, I did try to capture test cases for all function calls under both nominal and off-nominal conditions.  This approach gave me the confidence to try out things quickly and verify if the expected behavior was observed.  As a follow-on, it allowed me to catch bugs and identify corner-cases much more quickly than traditional approaches; i.e. top-down development with little to no unit testing.

To automatically execute and generate a test report, I built my unit tests using [Boost's unit test](https://www.boost.org/doc/libs/1_45_0/libs/test/doc/html/utf.html).  After building the shared library, the unit tests can be built with `make test` and the pass/fail data is reported automatically. 

## Code quality and standards-checking
I have included a wrapper script in the GitHub repo that does a quick static code check using [`cpplint`](https://github.com/cpplint/cpplint), which verifies that the code meets common style conventions for C++.  This helps to keep the repo's implementation consistent should additional features be added down the road.

# Summary and future work
In this post, I presented my design choices for the computational solution to the problem of point cloud matching, which I developed in pair of previous posts.  I have made the work available for others to use and contribute to, should they wish to do so.  Some things that I would like to add to the repo would be in no particular order:

* Continuous Integration, CI - _this way I can automatically check that all tests pass before merging new commits_
* Code coverage tests - _I'd really like to make sure that there are no corner cases that I am neglecting in my test-suite_
* Adding [maximum clique algorithm](https://en.wikipedia.org/wiki/Clique_problem#Finding_maximum_cliques_in_arbitrary_graphs) correspondence solver - _more on this to come in a future post!_

Thanks for reading!  Check out the link at the top of this post for the GitHub repo.
