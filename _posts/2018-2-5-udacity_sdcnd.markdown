---
layout: post
title:  Self-Driving Car Nanodegree Summary
description: opinions and reflections on my 9-month learning expedition with Udacity
date:   2018-2-5 00:10:00
comments: true
---

<p align="center"> 
<img src="/assets/img/nd013.png">
</p>

### Summary

* High value, High cost
* The first term was much more rigorous than the subsequent two
* Entire technology stack, from high-level planning to low-level controls, was covered in adequate depth
* Culminated in programming Carla, Udacity's self-driving car

### Introduction

I had been working for Ford for a about a year when Udacity launched the Self-Driving Car Nanodegree, now rebranded as the _Self-Driving Car Engineer Nanodegree_.  I was on the object detection and tracking team which, admittedly, I did not find as fascinating as the other technology areas adjacent; these included path planning, motion controls, and computer vision.  I wanted to learn more about these other topics so that, one day, I could move to another group doing work within Ford in one or more of these areas.

Udacity has the deserved reputation of being more expensive than competitors when comparing similar offerings.  I took this into account before considering enrolling in the program.  The nanodegree was broken out into 3 terms, each costing a whopping $800.  Wow...  $2400 for an online education credential?!  And one that is through an unaccredited institution?  To Udacity's credit: this program has no competition among the other big MOOC platforms.  Coursera and edX have nothing comparable.  The affiliation with industry partners Nvidia and Mercedes-Benz made this program very appealing.  I ultimately decided to commit the time and capital resources to the program.  I estimated that each term would take about 3 months to complete.

I enrolled in the second-ever cohort, experienced some of the growing pains of early adoption, and had a blast working through the material.  I will go through each term, presenting some interesting results and my opinions along the way.

### Term 1:

[My GitHub repo](https://github.com/jwdinius/CarND-Term1){:target="_blank"}

The first term was dedicated to computer vision, a field in which I was quite inexperienced.  I had some experience throughout the years working with imaging, e.g. IR cameras, and non-imaging, e.g. radar, sensors, however I had not done much work with image processing.  I was really looking forward to this term because it gave me a chance to learn methods of image processing, like convolutional neural networks and gradient-based methods, and apply them to real-world problems encountered in self-driving cars.

The video below shows the output of the project where I used OpenCV filtering and processing methods to do lane finding on a video stream captured from a moving car.

[![lane_finding](https://img.youtube.com/vi/EHplRv18Brw/0.jpg)](https://www.youtube.com/watch?v=EHplRv18Brw){:target="_blank"}



I hadn't thought about it before, but image processing could be used to develop steering commands for a car!  Using a convolutional neural net architecture, I was able to train a model to steer a simulated car around a test track.  I trained the model by driving the car around the test track many different times and many different ways: counterclockwise, clockwise, weaving about the track, etc...  The data captured provided the neural net with the appropriate steering commands given a picture of the world at any given time.  This was super cool.  Check out the video below:

[![cnn_steering](https://img.youtube.com/vi/DxcIq6H5sWk/0.jpg)](https://www.youtube.com/watch?v=DxcIq6H5sWk){:target="_blank"}


This term, aside from the the really cool projects, had a high-level of academic rigor.  The project output was only a part of the criteria for successful completion.  The other part, which really impressed me, was a detailed analysis of the methods used in the context of the problem.  I think that it is important to do complete analyses when learning new material.  Otherwise, with emphases placed only on production, engineers are turned into technicians capable of nothing more than reiteration of previous results.


In this term, I completed projects that classified traffic signs from a database of images, detected vehicles in a video stream, drove a simulated car around a test track, and identified lanes from a video stream in real-time.  Below, you'll see the output of the vehicle detection project:

[![detection](https://img.youtube.com/vi/VysM74ktGTE/0.jpg)](https://www.youtube.com/watch?v=VysM74ktGTE){:target="_blank"}


I liked the vehicle detection project so much that I [extended it](https://github.com/jwdinius/CarND-Term1/blob/master/CarND-Vehicle-Detection/pipeline.pdf) it with a tracker to smooth out the measurement transients.  I used this work for a few job interview presentations with strong favorable feedback.

This term was awesome!  In a period of only a few months, I completed five projects, each relevant to real-world perception problems in self-driving car and, most importantly, using real data.

### Term 2:

[My GitHub repo](https://github.com/jwdinius/CarND-Term2){:target="_blank"}

Term 1 set the bar very high, so I was really looking forward to Term 2.  The scope of this term was image processing and control; with emphases placed on tracking, localization, and control.  The first few projects didn't seem that interesting to me; they were concerned with Kalman filtering, and I had had much experience with implementing and tuning these in practice.  Even with my experience with Kalman filtering, I still found some useful tips and tricks.  Below, you'll find the link to the unscented Kalman filter project results:

[![ukf](https://img.youtube.com/vi/rQWKwz2ewJM/0.jpg)](https://www.youtube.com/watch?v=rQWKwz2ewJM){:target="_blank"}


One departure from the first term I observed quickly: this term's projects did not require the same level of effort as those of the first.  Granted, the projects were done in C++, as opposed to Python, and there isn't an integrated platform for code and analysis like Jupyter notebooks.  I was a little disappointed that there was no longer an analytical component to the projects.

There really were some interesting projects here.  My favorite was, perhaps, the particle filter landmark-based localization project:

[![particle](https://img.youtube.com/vi/HBIvq_eb5rE/0.jpg)](https://www.youtube.com/watch?v=HBIvq_eb5rE){:target="_blank"}



The term ended with a comparison of PID and model-predictive control methods for error-correction and stability.  You can see the MPC results in the video below:

[![mpc](https://img.youtube.com/vi/63_C3s6U8AU/0.jpg)](https://www.youtube.com/watch?v=63_C3s6U8AU){:target="_blank"}


Although much of the material was not new to me, I found some interesting aspects that I had overlooked previously.  The simulation platform for testing algorithms was very good and allowed for rapid algorithm development and testing.

### Term 3

[My GitHub repo](https://github.com/jwdinius/CarND-Term2){:target="_blank"}

The third term was a culmination of the previous two.  Techniques presented previously were further explored with greater depth.  The first project was a high-level path planner that fused sensor data with vehicle telemetry to move about in traffic:

[![path_planning](https://img.youtube.com/vi/AILA8kxl56Y/0.jpg)](https://www.youtube.com/watch?v=AILA8kxl56Y){:target="_blank"}


The second project was interesting; there was an choice between two options.  I chose the option for advanced deep learning applied to semantic segmentation of camera images.  Semantic segmentation means a pixel-by-pixel classification of content within images by training a convolutional neural network to identify which pixels belong to certain objects.  The project had only two objects: roads and non-roads, but the work could be easily extended to add more classes.  The image below shows a sample segmented image, with the road pixels highlighted in green:

<p align="center"> 
<img src="/assets/img/um_000019.png">
</p>

The final project was incredible.  We got to organize teams and write code that ran on Carla, Udacity's self-driving car.  I wrote up a more in-depth [project report](/projects/carla/).  This experience was very rewarding, and also very challenging.


### Conclusion

The problems I encountered throughout all three terms were not academic; they were _real_ industry problems along with a discussion of how industry approaches them.  I believe that Udacity's approach is unique among other platforms is this regard; neither edX nor Coursera have much regard for industry in their courses or specializations.  While the cost was relatively high, the return on investment was much more tangible.  I learned a lot in a relatively short period of time, and the knowledge was directly applicable to robotics, in general, and self-driving cars, in particular.


