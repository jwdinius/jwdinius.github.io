---
layout: post
title:  Some thoughts on control theory
description: I worked through Friedland's "Control System Design" book and have captured a few thoughts about control theory, in general.
date:   2019-5-17 08:00:00
use_math: true
comments: true
---

A few months ago, I worked through Friedland's really excellent graduate-level controls book [Control System Design](https://www.amazon.com/Control-System-Design-Introduction-State-Space/dp/0486442780).  I have been working in the control systems-domain for most of my professional career, but I didn't have a solid foundation in state-space methods, so I decided to pick up the book and work through each chapter meticulously; including working most exercises to the best of my ability.  I went into the exercise with the notion that I would discover that state-space methods were superior to those of the frequency domain.  What I actually learned was quite different.  I have tried to pair down the big takeaways from this exercise to three main points, each of which are summarized below.

### State-space control methods are best applied to observer design
What I found most striking about state space control methods for linear time-invariant systems were that, given controllability and observability criteria, closed-loop system response could be chosen _arbitrarily_!  This means that the poles/eigenvalues of a closed-loop system, with linear feedback, can be placed anywhere.  This is truly amazing!  But, there is a caveat: _such methods are only useful when full-state information is available for feedback_.  This is a major bummer, because usually only a subspace of the total phase space of an LTI system is measurable.  Not to worry, the same state-space methods can be used to design _observers_ to estimate the full state from limited measurements.

A quick summary of objectives for state-space control is as follows:

* _For controller design_: Find a gain matrix to drive system state towards a desired reference in a desirable way; e.g. with error between reference and actual state decaying exponentially with desired rate.  The rate is arbitrary, but should consider things such as disturbance rejection, noise attenuation, and the variation in time of the reference signal
* _For observer design_: Find a gain matrix to drive observation error to zero in a desirable way; e.g. exponential decay with desired rate.  The rate is arbitrary, but should consider things such as measurement noise and model uncertainty.

Observers are themselves LTI systems that seek to find a residual gain, $$K$$, to be applied to the difference between the _actual_ and _predicted_ measurements. A predicted measurement is what the observer predicts the measurement to be given the current state estimate. Observers are truly remarkable.  Before they were introduced, there was no mathematically precise way to use estimated state information in full-state feedback control.  An example of a very popular observer is the [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter), which has spawned many variants and is used in most systems where smooth state estimation is required.

So why exactly do I say that state-space control methods are best applied to observer design?  Here are the major points:

* _State-space methods are really all there is for linear observer design_: Take the Kalman filter for example:  using state space methods, and making some simple assumptions about system dynamics and the structure of noise present in the system, a provably-optimal observer can be designed using state space methods.  Nothing comparable to this exists for observers in the frequency domain, to my knowledge.
* _State-space control is brittle_: In order that system response actually match designed response, uncertainty in the system must not be present.  This is really an unreasonable ask and, in my opinion, renders state-space methods for controller design dead-on-arrival.  Assuming the system evolves perfectly according to the LTI model, there is still the matter of needing full- or nearly full-state feedback to place the poles.  In practice, this is done with an observer, and the observer introduces it's own error dynamics that would be unaccounted for in the original system model.  Chapter 8 deals with compensator design for this problem, but it's pretty complicated and doesn't deal with measurement noise or model uncertainty of any kind.

I address the second bullet point in the next section.

### Frequency domain methods can't so easily be discarded
This came as quite a blow to me, but upon reflection, it seems to make perfect sense for the following reason:

> State space methods decompose the system into two parts: controllability and observability.  Each of these are evaluated in separate stages.  The frequency domain approach, however, considers both observability and controllability simultaneously when reproducing the transfer function between input and output.

There is no good way to deal with uncertainties in the state-space methodologies.  Sure, there are perturbation techniques that can be used to do some sensitivity analysis, but there is no substitute for the rigorous stability assessment techniques of the frequency-domain approach.  Via singular value analysis or Bode plots, system feedback can be designed with frequency attenuation in mind.  True, frequency domain methods are not as mathematically-pretty as state-space methods, but I don't believe that one can deny they are more powerful.

> State-space methods provide no direct way of addressing disturbance rejection and robustness!

Friedland presents nice historical discussions of both methods in the context of the material he is presenting.  He continually refernces a large divide between groups of both practitioners in terms of uses and limitations of their methods.  After finishing the book, I don't really see such a sharp divide.  To me, both methods are useful and have domains where they are most readily applied.  Neither method should be discarded in the overall considerations of control system design.

### Working through textbooks is crucial to becoming a subject matter expert
Many may disagree with me on this point, especially in the wake of massively online open courses delivered by [Coursera](www.coursera.org) and [Udacity](www.udacity.com) that deliver content in 5-10 minute digested chunks, but I believe that attaining true mastery of a topic requires a higher level of commitment.  Online courses can teach practical skills, however textbooks present ways of thinking about focused topics _in general_.  This latter skillset is crucial for building the general-purpose skills required for tackling unsolved problems.  I don't advocate discarding one learning method in favor of another; both are important.  I merely want to highlight the point that knowledge of practical skills alone is insufficient to understanding their application in novel contexts.

If you would like to see my solutions to the exercises, check out this [repo](https://github.com/jwdinius/friedland-csd-solutions).
