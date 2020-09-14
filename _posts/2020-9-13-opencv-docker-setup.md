---
layout: post
title:  Building a Docker image with OpenCV
description: building your own Nvidia-powered instance of OpenCV deployed within a Docker container
date:   2020-9-13 11:30:00
comments: true
---

![Imgur]({{ site.baseurl }}{% link /assets/img/facial_landmark_det.gif %})
_Sample demonstration showing the development environment discussed below in action:  A facial landmark detector is shown attempting to keep up with my face while I move it around and change orientation.  The noisy red dots show the detector with no smoothing while the blue dots show the results of applying optical flow to smooth out the noise._

### Background 

Earlier this year, I completed the first in a [series AI courses from OpenCV](https://www.kickstarter.com/projects/satyamallick/ai-courses-by-opencvorg).  Most of the course assignments were completed using Jupyter notebooks; all other assignments, including projects, were completed on my host machine.  I was given the following two options for satisfying all of the dependencies for completing the course assignments on my host machine:

* Install OpenCV and its dependencies natively on my machine
* Pull a [Docker image](https://hub.docker.com/r/vishwesh5/opencv/tags) from dockerhub

The first option was not desirable for several reasons; not least of which is the potential for conflict with other versions of dependencies already installed on my machine.  Option 2 was significantly better, and I have used Docker a lot over the last year-and-a-half, so this was the option I chose.  Completion of all of the non-notebook assignments went well; primarily because all input data was read from a file.

I recently enrolled in the second course in the series, which is focused on applications, and I wanted to see if I could create an environment - built with Docker, of course - that would be optimal for my hardware configuration: 
_workstation with a single 6-core CPU and a GTX-1080i Ti Founder's Edition graphics card, running Ubuntu 18.04 as the OS and a Logitech C270 USB Webcam._

### Setting up Docker

The first desirable optimization would be to get GPU acceleration for OpenCV inside of my container instances.  My environment was already setup for this, but I'll mention briefly here the steps I followed

* [Install Nvidia driver (> 430)](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-18-04-bionic-beaver-linux)
* [Install Docker](https://docs.docker.com/engine/install/).  _I also followed the_ [Post-installation steps for Linux](https://docs.docker.com/engine/install/linux-postinstall/)_._
* [Setup X-forwarding for GUI apps](https://iamhow.com/How_To/Docker_How_To.html)
* [Install Nvidia runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
* Enable Nvidia runtime by default: add the following line to `/etc/docker/daemon.json` file
```bash
"default-runtime": "nvidia"
```
_make sure the resulting_ `daemon.json` _file is valid json, otherwise docker will fail upon attempting to restart!_

Now, most of the infrastructure is in place for building our image.  After identifying that dependencies for OpenCV - and OpenCV, itself - would result in intermediate containers that exceed the default Docker base device size while building my image, I followed [this guidance](https://www.projectatomic.io/blog/2016/03/daemon_option_basedevicesize/) for increasing the base device size.  _In practice, I found that a base device size of 30GB was sufficient for building the desired image._

### Building the Docker Image

I start from a base image from [here](https://hub.docker.com/r/nvidia/cudagl/).  The CUDA runtime library, OpenGL implementation, and other dependencies are enabled immediately, which makes setting up the remainder of the image easier.  [CuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) is not present, but is desirable for neural network inference.  Before attempting to build the Docker image, download the CuDNN runtime and dev libraries - as debian packages - from the Nvidia developer site following [these steps](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn_751/cudnn-install/index.html#download) and move/copy them into the same folder as the [Dockerfile]({{ site.baseurl }}{% link /assets/txt/Dockerfile %}).  Now, you are setup to build the docker image:

```bash
cd {dir-with-Dockerfile}
docker build {--network=host} -t {name-of-image} .
```
_The_ `--network=host` _option allows using the host machine's network interfaces directly. I usually disable the Docker bridge network and just use host networking for all of my containers._

This will take awhile to build...

In the meantime, you can consider the following things about the Docker image being built:

* Steps discussed [here](https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/) were used as the basis for building OpenCV, with two exceptions:
   * CUDA acceleration flags are enabled for this environment
   * No Python virtualenv is setup - _the Docker environment is already sufficiently isolated._
* A user with passwordless login and sudo privileges is created.  This allows for easily attaching additional terminals to a running container instance as well as adding desirable additional packages not included in the original image build.
* A user-defined entrypoint script

```bash
#!/bin/bash
set -e

# start jackd server to avoid webcam crash with guvcview
jackd -d dummy &
exec "$@"
```
is included to enable webcam streaming within the container _after correctly setting up the host environment_.

### The Host Environment

After digging into an issue with my webcam not properly streaming, it seemed I had a [permissions issue](https://askubuntu.com/questions/457983/how-can-i-get-my-webcam-to-work-with-ubuntu-14-04) on `/dev/video*` in my host machine.  This was easy enough to fix with a udev rule executed at startup:

* Create a file `/etc/udev/rules.d/99-webcam-rules`
* Add the following line to the file:  `KERNEL=="video[0-9]*",MODE="0666"`  _assuming your webcam is discovered as /dev/video[0-9]_
* Restart the host machine

Non-root users -including our newly created Docker user - will have read-write access to the webcam now.  Everything should now be in place to run and test the container.

### Launching a Container Instance

We want our container to be able to do the following:

* Display GUI windows - from a webcam streaming app like `guvcview` or from OpenCV-based applications
* Ability to read from the webcam
* Enable non-volatile storage for intermediate work products - e.g. source code under development

We can achieve all of these goals with the following run command:

```bash
docker run --rm -it \
    --name opencv-course-c \
    --net host \
    --privileged \
    --ipc host \
    --device /dev/video0 \
    --device /dev/video1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/course-materials:/home/opencv/course-materials \
    -e DISPLAY=$DISPLAY \
    opencv-nvidia \
    /bin/bash

```

This command, option-by-option, does the following:

* Tells the Docker runtime to cleanup the container environment when the user triggers exit - `--rm`
* Creates an interactive container - `-it`
* Gives the container instance the name `opencv-course-c`
* Uses host networking - `--net host`
* Gives the container privileged access - _required for x11 forwarding, apparently_
* Uses host shared memory for interprocess communication - `--ipc host`
* Gives access to `/dev/video*` devices
* Sets up X11 forwarding from host
* Mounts `./course-materials` folder as read-write volume inside of container at `/home/opencv/course-materials`.  _This is the non-volatile storage_
* Uses host display
* Uses `opencv-nvidia` image as container base
* Launches a bash shell for the user to interact with

Now, you should be ready to experiment with this; add sample OpenCV source code, compile and run it, and see what happens.  The gif of facial landmark tracking I share at the beginning of this blog post was generated using this environment, so I'm pretty confident it'll work.  I would share the facial landmark tracking app, but the code comes from the second OpenCV course, which is behind a paywall :disappointed:

I've only just begun to use this environment, and I'm really looking forward to pushing further and doing more with it.  I hope you'll find this post and materials referenced useful in your own learning journey.

Thanks for reading!
