---
layout: page
title: Self-Driving Car
description: ROS - Perception & Control
img: /assets/img/carla.jpeg
category: autonomy
importance: 4
use_math: true
---

## Abstract
I recently had the chance to write code to run on Carla, Udacity's self-driving car.  The car's perception, planning, and controls modules were written on top of an existing ROS framework.  To test out the efficacy of the written code, simulation experiments were performed using all-digital simulations and pre-recorded ROS bagfiles.  This project write-up describes the project, the approaches considered, and some results.  The end product was the following [code repository](https://github.com/jwdinius/CarND-Term3/tree/master/CarND-Capstone-NPLH).

## Outline
Udacity has [Carla, the self-driving car](https://medium.com/udacity/how-the-udacity-self-driving-car-works-575365270a40), and, in a very unique offering, they allow students taking the third term of the [Self-Driving Car Engineering Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) to write code that will run on it.  This write-up is a walkthrough of the approach that my team took to write the perception and control logic for safely driving the car.

The project began with team formation.  Team formation was voluntary and tracked via a Google Docs spreadsheet.  I led the team *No Place Like Home*, composed of four other members and myself.  All of us were running a bit behind schedule and we had a mutual incentive to complete the project on a particular timeline.  Team formation proved to be a major factor in the project; more on this later.

The following system diagram shows the interconnections between different subsystems on Carla, and it provides a nice pictorial introduction to the logic that needed to be written:

![system_diagram](/assets/img/carla/final-project-ros-graph-v2.png)

This project is broken into the following steps:

* **Perception** 
* **Planning**
* **Control**

The **Perception** step involved obstacle avoidance and traffic light detection.  The obstacle avoidance piece was not covered in this project.  The goal here was to create a working traffic light detector that would tell the planner when to stop for red lights.

The **Planning** step was to create a way of handling car behavior at pre-defined waypoints along specific tracks.  There are two tracks

* Simulation - used for basic logic testing and debugging
* Site - the actual test track that Carla drives around

The **Control** step handled the drive-by-wire, DBW, logic for smooth steering and throttle/brake commands to send to the actuators.

Each step was worked on by different team members in parallel.  My individual responsibilities were the **Planning** module, administration and team planning, and system integration.  I will present here all of the work that the team accomplished to get the code up and running on Carla.

## Preliminaries
The team was provided with a [basic repo](https://github.com/udacity/CarND-Capstone) with communication between nodes plumbed out.  The interprocess communication layers were built using [ROS](http://www.ros.org/), an ever-more popular starting point for robotics projects because of its large user base and many success stories.  ROS also has many debugging tools, such as [rviz](http://wiki.ros.org/rviz), good logging capabilities, and plotting tools.  All of these tools come with a nice python interface.

## Perception
Since the obstacle avoidance part of the perception layer was out-of-scope for this project, the only remaining task was to build a traffic light detector capable of both detecting traffic lights and classifying the color observed.  The approach our team took was inspired by [this blog post](https://becominghuman.ai/traffic-light-detection-tensorflow-api-c75fdbadac62).  A quick synopsis is that we built an integrated bounding-box detector and classifier using the *Resnet 101* model.  Two separate inference graphs were generated: one for the simulation and another for implementation on the real car.  We were able to implement these two separate graphs within a single ROS codebase with the addition of a simple `sim` parameter defined in the launchfiles.

Typical standalone perception results are shown below:

[![site_test_bag](https://img.youtube.com/vi/7tjPTK19POg/0.jpg)](https://www.youtube.com/watch?v=7tjPTK19POg){:target="_blank"}

The video shows classification accuracy in the terminal window at the left.  The camera video playback and the ros bag output are also displayed.  The classification results match the observed light colors from the video playback, with accuracies well above 90%.

The classification accuracies are impressive, but in practice the classification pipeline using this particular neural network architecture was quite slow, and this led to integration issues later on.  I will discuss this more later.

## Planning
The planning step is the one I was most directly involved in with respect to code architecture and implementation.  We were provided a set of waypoints via csv files for both the simulation and the site tracks.  Each waypoint had a pose and an associated velocity.  The goal of this step was to set desired velocities incorporating output from the traffic light detector and the waypoint loader.  These desired velocities were then passed to the control layer to generate actuator commands.

The particular strategy I employed was simple: through subscription to the traffic light publisher, the waypoint index and subsequent location of red lights are determined.  If a red light was detected at an upcoming waypoint, a linear ramp down to 0 of the longitudinal velocity is commanded.

The output of this step is a set of waypoints ahead of the current one, along with a desired velocity at each waypoint.

## Control
The control step generated drive-by-wire steering and throttle commands by passing the outputs of the planner through a low-pass filter.  The filter was provided as part of the base repository, as was the PID controller logic that was used to zero out errors between the desired and achieved vehicle responses.

## Integration
Bringing all of these pieces together was difficult...

Aside from team members scattered throughout the world with various personal commitments, the technical piece alone was quite difficult.  Getting all of the pieces to work on both the simulation and pre-recorded bag files required many hours of effort; iteration-after-iteration was performed to get better performance.  Simulation results are shown below:

[![submission_sim](https://img.youtube.com/vi/VbBsrrDvEf0/0.jpg)](https://www.youtube.com/watch?v=VbBsrrDvEf0){:target="_blank"}

The simulated car stops at red lights, indicating the classifier is working, and stays in the lane, indicating the DBW logic is functioning correctly too.  We thought we were ready to run on the real car at this point, so we submitted our code for running on Carla.  

Fast-forward two weeks...  That's right, it was two weeks to get feedback.  We got the ros bagfile and feedback on our submission.  The video below shows the reviewer's response to our submission.  Aside from the jerky startup at the initiation of drive-by-wire control, the vehicle appeared to drive quite smoothly.  The reviewer even commented with "Nice Job".

[![site_feedback](https://img.youtube.com/vi/lNJQDRigO9g/0.jpg)](https://www.youtube.com/watch?v=lNJQDRigO9g){:target="_blank"}

We were feeling pretty good about our submission at this point, but deeper inspection of the bagfile left us feeling not so great, see the video below:

[![site_bag](https://img.youtube.com/vi/ByDRA8uefmQ/0.jpg)](https://www.youtube.com/watch?v=ByDRA8uefmQ){:target="_blank"}

The car appears to go through a red light!  Digging through the logfiles showed that the traffic light detector had failed to come up in time to detect the red light.  By the time the node had come up and was functioning, the car had already driven through the red light.  This was disheartening, but the team seemed interested in fixing and resubmitting.

"Seemed" was perhaps the opportune word in the above sentence.  Of the four other team members, only one other member put in any additional effort, and even that was very limited.  The criteria for graduation from the program was submission *only* of project code.  Ultimately, it is up to each individual team to decide when they are satisfied with the results.  Apparently, the rest of our team was satisfied with the results, but I wasn't.  I then put together another submission that addressed the issues of the first submission, namely the following:

* Abrupt transition when engaging the drive-by-wire control
* Traffic light detection engages too late

After digging through the code, I found that the filtering strategy being used was to blame for the abrupt transition; acceleration commands at initiation were spiking.  The second issue was more subtle, and I had a lot of difficulty addressing this.  Ultimately, my strategy was to give a little more time for the traffic light detection node to initialize by slowly increasing the waypoint speed to it's maximum value when the ros processes first come up.  I made these changes, and resubmitted.

[![sim_resubmit](https://img.youtube.com/vi/hIMgtaHPAqY/0.jpg)](https://www.youtube.com/watch?v=hIMgtaHPAqY){:target="_blank"}

The above simulation results show the desired effects of the changes.  The vehicle slowly increases speed upon initiation.  The first light is resolved with plenty of distance to stop.  Not shown: the filtered steering and throttle commands no longer spike at instantiation.

The output from Carla was not as expected. The log seems to indicate that the traffic light detection node failed to spin up in time again.  On the plus side, the drive-by-wire behavior was much smoother.  At this point, simulation behavior was dificult to correlate to actual vehicle behavior.  Despite my desire to get the code really working on Carla, the time to feedback and the inability to debug issues in real-time made me realize that further efforts would not be fruitful.

## Challenges

* The biggest challenge faced was the submission-feedback timeline.  Our first submission took over a week to get feedback, at which time there was a technical issue on Udacity's side that forced us to resubmit and wait another week for results.  For the two submissions made, the *average* time to feedback was over two weeks.  With everything else going on in my life, it was difficult to stay focused on this project.
* The test track at the site only had one traffic light at the very beginning, and therefore debugging the system under steady-state conditions was next to impossible.
* Vehicle behavior between the test track and the simulator were qualitatively different.  Simulation results showed expected output: stopping at red lights, maintaining constant speed between lights, and lane-keeping.  The track behavior, on the other hand, showed an inability to stop at the traffic light.
* Team members had widely varying levels of engagement with the project.  Some team members were active contributors, while others actually made more work for the rest of us.

## Concluding Remarks
This project was a lot of fun to work on.  I got a chance to do some project management during execution of a complicated project with multiple participants.  I also got a chance to work with an exciting platform for robotics development, ROS, applied to a real self-driving car.  I would have liked there to have been a better matching between simulation and real vehicle behaviors, as this would have made debugging much easier.  All-in-all, I had a great time working on this and although I'm a bit disappointed with the end result, it was a great experience.
