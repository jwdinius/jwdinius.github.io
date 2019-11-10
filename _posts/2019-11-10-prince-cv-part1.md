---
layout: post
title:  Review of Prince's CV Book, Part One
description: Review of Part I of "Computer Vision. Models, Learning, and Inference"
date:   2019-11-9 08:00:00
use_math: true
comments: true
---

Over the past few months, I have been reading through Prince's awesome book [_Computer Vision: Models, Learning, and Inference_](http://www.computervisionmodels.com/).  The book takes a probabilistic approach to computer vision with a remarkable level of mathematical rigor: not unapproachable, but requiring commitment to work through the details.  For me, Prince's book strikes a much-needed balance between the academic and practical concerns for practitioners of computer vision.

> You can find solutions to end-of-chapter exercises and algorithm implementations at my [GitHub repo](https://github.com/jwdinius/prince-computer-vision).  At the time of writing this post, work through Chapter 8 has been mostly completed.  I will continue adding solutions and algorithms as I review more parts, so stay tuned!

In this first of several posts, I will go through Part I, which is titled _Probability_.  Subsequent posts will tackle each part of the book in turn.  This way, I can keep the posts focused and readable in about 10 minutes' time.

The first thing to note about this book comes before Part I even begins: _the visual aids are highly informative_.  Even the graphic, Figure 1.2 on pg. 4, showing organization of the book's parts is eye-catching and immediately makes plain the author's intent.  The book's visuals, including graphs and tables, are laid out in such a way that complex relationships are demonstrated clearly and intuitively.  The book is worth a glance for the visuals alone, to say nothing of the rest of its content.  Now, let's move on to Part I.

## Part I: Probability

Part I begins with a background on probability theory that you could get from any undergrad textbook.  For example, my undergrad textbook was [_Probability and Statistics_ by DeGroot and Schervish](https://www.amazon.com/Probability-Statistics-4th-Morris-DeGroot/dp/0321500466), if you are looking for a decent reference.  The ideas about conditional probability, marginalization, expectation, and Bayes' rule are presented as motivation for later chapters, but there is not anything really unexpected here.  The next chapter, however, starts to get interesting in a hurry!

In Chapter 3, Prince presents the idea of _conjugacy_ of probability distributions in the context of fitting models to data.  Let's say you have training data,

$$X = \{x_i | 1 \le i \le I\}$$

and you want to find a suitable probability distribution to fit the data.  Well, usually the distribution will be chosen based upon observations about the data; namely, are the data _continuous_ or _discrete_.  Whichever model is chosen, there will be a parameter vector, call it $$\mathbf{\theta}$$, that needs to be chosen to _best_ represent the data.  There are two big questions/concerns that emerge at this point:

* How do we find the _best_ parameters?  What is meant by _best_?
* What can be said about the _uncertainty_ in our chosen model parameters given the data?

Prince addresses the second question directly on pg. 18:

> ... when we fit probability models to data, we need to know how uncertain we are about the fit.  This uncertainty is represented as a probability distribution over the parameters of the fitted model.

The probability distribution over the parameters of the fitted model referred to in this quote is the _conjugate_ distribution to the original model distribution used for fitting.  This is a powerful idea: _Given data, not only can we identify and fit a particular model to these data, we can also quantify the uncertainty of our fit!_  With well-worked examples, Prince further elaborates on this point.

Back to the first question now:  _How do we_ actually_ accomplish the fit?_.  There are three ways presented:  _Maximum Likelihood Estimation, Maximum a Posteriori, and Bayesian_.

### The Major Model-Fitting Techniques

#### Maximum Likelihood Estimation, or MLE

Maximum Likelihood Estimation, unsurprisingly, finds the parameters that maximize the likelihood of the data observed _in the absence of any prior knowledge about the underlying distribution of the model parameters_.  This last part is important, as you'll see in the next section on the Maximum a Posteriori approach.  Mathematically, MLE seeks to evaluate the following optimization objective:

$$ \hat \theta = \text{argmax}_\theta \bigl [ Pr(X | \theta) \bigr ].$$

That is, we are trying to find the $$\hat \theta$$ that maximizes the expression above, which assumes no prior on $$\theta$$.  I don't wish for the post to get caught up in details at this point, so I'll refer the interested reader to Chapter 4 in the book for reference.  Before moving on, though, I want to briefly mention that the independence assumption of the individual data points in $$X$$ is important in making the expression above solvable in closed-form for typical distributions; e.g. normal or Gaussian distributions.

#### Maximum a Posteriori, or MAP
The Maximum a Posteriori approach is structurally quite similar to MLE, only the objective is now:

$$ \hat \theta = \text{argmax}_\theta \bigl [ Pr(X | \theta) Pr(\theta) \bigr ].$$

Note the inclusion of the prior term $$Pr(\theta)$$.  If we know something about the distribution before doing the fit, we can use that prior knowledge.  In the case of an uninformative prior, like $$\theta$$ is drawn from a uniform distribution over its outcome space, the MAP approach _is exactly the same as MLE_!

#### Bayesian
I think that the Bayesian approach is my favorite.  Whereas the two approaches above use the data to find a _single_ parameter vector $\theta$ to fit the data, the Bayesian approach does something fundamentally different.  You can observe that, given any set of data and a desired model to fit it, _there are infinitely many parameter vectors that_ could _explain the data_!  This may require some thought, but it is particularly true in the case datasets with only a few representative samples.  What the Bayesian approach seeks to find is an appropriate weighting for this continuum of parameter choices for model fitting parameters.  This flexibility comes at a cost when it comes time to predict the probability of a new data point given the training data, though.  This is because the probability must be computed as a weighted average _over all likely models_, where the weight comes from the fit probability $$Pr(\theta | X)$$.  The fit probability comes from, as you may have guessed, Bayes' Rule:

$$Pr(\theta | X) = \frac{Pr(X | \theta) Pr(\theta)}{Pr(X)}.$$

These three approaches are cornerstones of statistical model fitting, and Prince presents them succinctly but thoroughly.  Plots, such as Figures 4.6 and 4.8 on pages 35 and 37, respectively, highlight the similarities and differences between the different approaches very clearly.  Specifically, the following key ideas are made plain:

* The MLE and MAP approaches can be overconfident in their predictions.  Because they only seek to find a _single_ parameter vector to explain the data, they may generalize poorly to data unseen during training.
* The Bayesian approach may be too cautious in its predictions.  This is because predicted probabilities are an average over all _consistent_ parameter vectors.

### Summary
Part I of Prince's book presents the fundamentals needed for the remainder of the book.  Each chapter is presented in 20 pages or less, which makes going through it a breeze.  Oftentimes, authors will include extraneous details that derail the reader's focus away from the core thesis, but not Prince.  He only provides what is necessary for contextualizing future discussions related to the book's core topic: computer vision.  I have taken probability courses at the undergraduate and graduate levels and I wish that I would have had this book while going through them.  Some things that I struggled with then would have been made plain by Prince's insights.

As far as the content, itself, I enjoyed writing the algorithms from pseudocode and experimenting with them given representative datasets that I created.  I have made the work available on my [GitHub](https://github.com/jwdinius/prince-computer-vision), so feel free to clone and play around with the repo.  I have done the experiments in Jupyter notebooks so that interested readers can edit parameters on-the-fly in their own browsers.  Hopefully then they will be able to both evaluate model performance and gain insights with regards to the different models' practical concerns.

The next post will be more computationally-focused, as Part II is concerned with fitting of actual data and robustness to outliers.  Hopefully you will read through that post as well.

Thanks for reading! 