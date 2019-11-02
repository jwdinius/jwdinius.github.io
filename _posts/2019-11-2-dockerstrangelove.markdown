---
layout: post
title:  Docker Strangelove
description: How I learned to stop worrying and love containers
date:   2019-11-2 08:20:00
comments: true
---

_Caveat: I run Ubuntu, and so the contents of this post are specific to this context.  Although I believe much of the content would generalize, I have not verified.  Regardless, hopefully there will be something of interest to you non-Ubuntu's out there!_

[Docker](https://www.docker.com/) is all-the-rage nowadays because of it's ability to isolate application environments for development and testing.  They are super-popular in web contexts, because you can setup lightweight servers and mocks for isolated testing, but they are only just beginning to catch on in robotics.  Better late than never, I guess.  In this post, I'm going to talk about some of the motivating cases for using Docker in robotics development.  I use [ROS](https://www.ros.org/) as the starting point, but the context carries over to other networking protocols and frameworks.  The topics discussed herein should be sufficiently general to get the point across.  Anyways, let's get on with it.

To all the roboticists out there:  How many times have you come across a software package that you thought looked promising only to discover that the dependencies clash with your global dev environment?  In such a case, you could locally install the clashing dependencies and then point the package to the local install path while building the package.  While this certainly works, you are still modifying your workspace and creating potential issues for future development.  A _better_ solution would be to create an isolated dev environment, one where you could install the needed dependencies and use them only in the context of development/testing of the new package.  This is where containers, particularly [docker]() containers, come in.  Enough preamble: How about some practical use cases?

For the uninitiated, ROS is a popular framework for robotics development.  For the initiated, ROS can be endlessly frustrating because of the highly distributed and fluid nature of package development.  I will go through a couple of use-cases of how you can use There are different ROS releases, and these releases only have debian packages for certain Ubuntu releases; for example, ROS kinetic is not supported for Ubuntu 18.04.  If you are running Ubuntu 18.04 and you find an interesting ROS package built with kinetic, you're hosed right?  _Wrong!_  This leads to the first practical use-case:

## Containers alleviate release/versioning issues

With Docker, you can build a base image for your package based on the version of Ubuntu you _want_, rather than the version of Ubuntu you are _running_.  Let's say you are running Bionic \(18.04\) on your machine, but one of your colleagues shows you a cool demo they made using kinetic and you want to recreate it.  ROS kinetic is not supported for Ubuntu 18.04, so how would you do this?  You could create a virtual machine based on Xenial, but virtual machines are unwieldy and resource-intensive.  You could buy a new machine, flash it with Xenial and install, but this is not desirable either for financial reasons.  Or, you could install ROS kinetic, and all of the necessary packages, from source on your machine, but this might clash with your current dev environment and would be quite time-intensive.

All of these options suck, but there is another one:  _build a Docker image using a Xenial-based image as the starting point_.  [Dockerhub](https://hub.docker.com/) provides a great starting point for finding base images that can be easily extended for multiple contexts.  When you do `apt-get` to install debian packages, you will be installing them only within the image context: _they won't affect your global dev context!_  For the example, at hand, you can install ROS kinetic and all of the package dependencies in a docker image's context and not worry about polluting your global workspace.

This example covers the use-case of single applications, but robotic systems are composed of many networked _stacks_, _i.e. combination of multiple software functions for a common purpose_, that communicate with each other in tightly-coupled ways; sensors, algorithms, and motors all must robustly and efficiently communicate with each other for the robot to operate successfully.  What if we discover a major bug in one of our stacks, and that fixing this bug could have negative impact on the other stacks?  Here comes use-case number two:

## Containers isolate dependencies

Because Docker was designed to work in a web context, which is by its nature networking-friendly, we can use Docker for the context presented in the previous paragraph.  Each stack can be encapsulated in its own docker image and all images can be run concurrently during operation of the robot.  This approach is much easier to maintain; each stack can be tested in its own isolated context with greater ease.

I have a lot more to say on this, but the words are escaping me at the moment so I think that I will leave other thoughts for a later post.  I decided recently to try to write for one hour consistently each week and the best way to continue with that, I believe, is to be vigilant about writing for this time and, to start with, only for this time.  I don't want to derail my efforts by making this writing time stressful or otherwise unpleasant.  Over time, it is my hope, that the quality of these posts will get more succinct and cogent.  Thanks for reading!