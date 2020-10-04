---
layout: post
title:  Virtual Makeup Project Writeup
description: OpenCV Computer Vision II Applications Project #1
date:   2020-10-2 11:30:00
comments: true
---

### Introduction
I am currently enrolled in the OpenCV course "Computer Vision II: Applications (C++)".  For the first project, I had to implement two personal "improvement" features to be applied to a test image (shown below):

![test-image]({{ site.baseurl }}{% link /assets/img/girl-no-makeup.jpg %})

In this blog post, I present the writeup portion of this project. 

### Writeup
For my two features, I chose to implement virtual lipstick application and ???.  I will present a walkthrough of the rationale and solution for each feature in subsections below.  In each subsection, I will discuss the problem statement, background research I conducted before attempting a solution, and the final solution.

#### Feature 1: Virtual Lipstick
The point of this feature is to implement a _natural-looking_ overlay of a user-defined lip color applied to the test image shown above.  "Natural-looking" is the important part of the previous sentence: the variation in luminance in the new image with lipstick applied should be consistent with the original image.

As a start, I did a bit of research to find starting reference material to help me think about the problem.  The first reference I found was [this](https://www.learnopencv.com/cv4faces-best-project-award-2018/).  In this post, I found a brief description of the winning project, _AutoRetouch_, which introduced a lipstick application approach with a very good example of the application showing a near-seamless application of a different shade of lipstick applied to a test image, with both before and after photos for comparison.  The description of the application stated:

>This algorithm automatically created a lip mask based on the facial landmark detected using Dlib’s model. Naively blending color on top of the lips does not produce good results, and so he manipulated the chroma components of color while keeping luminance constant.

So, the key takeaways from this for me were:
* Use Dlib's facial landmark model to build a lip mask
* Use colorspace conversion to transform problem to a colorspace where luminance and chroma components of color can be manipulated instead of BGR colors

Next, I thought about what the right rule for transforming lip pixels from the test image would be to ensure that the transformation looked natural.  In other words, I wanted to find a reference that could help to explain to me how to preserve the underlying variation in the original test image despite doing color modifications in the output image.  While searching, I found this [article](https://www.pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/) from PyImageSearch about color transfer between images.  This article is great because it outlines a sequence of steps for doing the color conversion (to L\*a\*b\* colorspace).  Particularly useful were Steps 6 and 7:

>Step 6: Scale the `target` channels by the ratio of the standard deviation of the `target` divided by the standard deviation of the `source`, multiplied by the `target` channels.
>Step 7: Add in the means of the L\*a\*b\* channels for the `source`.

I found this approach to be really interesting and useful in the current context because it discusses an approach for taking the desired target channel intensities at each pixel and scaling/translating them to better match the source channel intensities.  The application described is different than my desired application, however I was able to pull out the following takeaways for my desired solution:

* I only care about changing the pixel values corresponding to the lips in the target image, so after creating a lip mask, I will compute the statistics (mean and standard deviation) on only the lip pixels in the test image (converted to an alternate colorspace with luminance channel)
* The color I want to apply is constant (just some BGR value corresponding to lipstick color) - so the standard deviation in my target will be 0.  _I only need to consider variation in the test image, not in the shade of lipstick applied over the lip mask._

From these takeaways, I was able to put together a solution that created the following output image (from the initial test image shown above):

![test-image]({{ site.baseurl }}{% link /assets/img/girl-lipstick.jpg %})

I'll now discuss the solution in more detail.

*Solution*

The process flow I followed was quite simple:

* Construct a binary image mask using Dlib's facial landmark detector and the `fillPoly` method from OpenCV
* Blur the mask to soften the transition from pixels adjacent to the lips to the lips, themselves
* Convert test image to alternate colorspace, compute statistics on lip pixels, and then use those statistics to transform the desired lipstick color to a natural embedding for the test image

I'll discuss each of these approaches in more detail:

*Identify lip pixels*

There were several examples in the first two weeks of the course that showed how to setup a Dlib facial landmark detector, so I reused some of that code as a starting point.  After running the facial landmark detector on the test image, I was able to pass those landmark points, along with the original image, to create a mask denoting the lip pixels in the target image with the following source code:

```cpp
void computeLipMask(cv::Mat const &im, full_object_detection const &landmarks, cv::Mat &mask) {
  std::vector <cv::Point> points;
  for (int i = 0; i <= landmarks.num_parts(); ++i)
  {
      points.push_back(cv::Point(landmarks.part(i).x(), landmarks.part(i).y()));
  }
  // top
  std::list<size_t> indicesTop = {64, 63, 62, 61, 60, 48, 49, 50, 51, 52, 53, 54}; 
  std::vector<cv::Point> pointsTop;
  for (auto const &i : indicesTop) {
    pointsTop.emplace_back(points[i]);
  }

  // bottom
  std::list<size_t> indicesBottom = {54, 55, 56, 57, 58, 59, 48, 60, 67, 66, 65, 64}; 
  std::vector<cv::Point> pointsBottom;
  for (auto const &i : indicesBottom) {
    pointsBottom.emplace_back(points[i]);
  }
  std::vector< std::vector<cv::Point> > polys = { pointsTop, pointsBottom };
  cv::fillPoly( mask, polys, 255);
  return;
}
```

In the source code above, I created two masks: one for the top lip and one for the bottom.  These two polygons were created by tracing the points _clockwise_ around each lip contour.  `fillPoly` takes in a `std::vector` of a `std::vector` of `cv::Point` objects and the resulting output is the desired mask.

*Soften the transition from lip to non-lip (and vice-versa)*

The binary mask created in the last step will most likely lead to a very dramatic transition around the lip boundary.  Because a major goal of this application is natural-looking output, I want to blur the mask to allow non-zero contribution of the desired lip color to pixels just outside the lip boundary.  Moreover, I would like to have the transition to inside the lip region from outside to be smooth across the boundary.  I decided the best way to achieve this would be to apply a Gaussian blur to the binary mask found in the previous step.  Here's the code:

```cpp
void blurLipMask(cv::Mat &mask, cv::Size const &size = cv::Size(0, 0)) {
  if (size.height == 0 || size.width == 0) return;
  auto maskCopy = mask.clone();
  cv::Mat blurMask;
  cv::GaussianBlur(mask, blurMask, size, 0);
  blurMask.convertTo(mask, CV_32F, 1./255);
}
```

This code is _very_ simple.  The default value for the blur kernel size is `(0, 0)`, which means that no blur will be applied; i.e. the "blurred" mask will be the same as the original binary mask.  However, if the user passes a non-trivial kernel size, a blurred mask, scaled to a floating point value in the interval `[0, 1]`, is returned by reference via the `mask` variable.  Now, we have a non-binary mask that can be used to smooth the transitions.

*Apply lipstick*

There are a few steps that I followed for applying the lipstick.  I will present the source code implementation, which has these steps outlined in commented blocks.  I will explain each step block-by-block.

```cpp
void applyLipstick(cv::Mat &im, cv::Mat const &mask, cv::Scalar const & color) {
  // "color" is the BGR value of the desired lipstick shade
  // STEP 1: get YCrCb decomposition of input image and desired lip color
  // STEP 1.1: input image
  cv::Mat imClr;
  cv::cvtColor(im, imClr, cv::COLOR_BGR2YCrCb);
  cv::Mat imClrFlt;
  imClr.convertTo(imClrFlt, CV_32FC3, 1. / 255);

  // STEP 1.2: desired lip color
  auto convertColor = [&]() {
    cv::Mat YCrCb;
    cv::Mat BGR(1, 1, CV_8UC3, color);
    cv::cvtColor(BGR, YCrCb, cv::COLOR_BGR2YCrCb);
    cv::Mat YCrCbflt;
    YCrCb.convertTo(YCrCbflt, CV_32FC3, 1./255);
    return YCrCbflt.at<cv::Vec3f>(0, 0);
  };
  auto lipstickYCrCb = convertColor();

  // STEP 2: compute weighted mean of lip pixels in YCrCb colorspace
  auto m = maskedMean(imClrFlt, mask);

  // STEP 3: apply alpha blending to pixels with non-zero value in the mask
  // this alpha blending utilizes the following scheme:
  // - apply weight (equal to the mask pixel value) to the desired YCrCb lip color + a translation that incorporates variation in the source image
  // - apply weight (equal to 1 - mask pixel value) to the original pixel in the source image transformed to YCrCb colorspace
  for (size_t i = 0; i < imClrFlt.rows; ++i) {
    for (size_t j = 0; j < imClrFlt.cols; ++j) {
      auto mpxl = mask.at<float>(i, j);
      if (mpxl > 0) {
        auto &srcPxlClr = imClrFlt.at<cv::Vec3f>(i, j);
        for (size_t idx = 0; idx < 3; ++idx) {
          srcPxlClr[idx] = mpxl * ( lipstickYCrCb[idx] + (srcPxlClr[idx] - m[idx]) ) + (1. - mpxl) * srcPxlClr[idx];
        }
      }
    }
  }

  // STEP 4: convert transformed image back to BGR and change depth to 8-bit unsigned int
  cv::Mat imFlt;
  cv::cvtColor(imClrFlt, imFlt, cv::COLOR_YCrCb2BGR);
  imFlt.convertTo(im, CV_8UC3, 255);
  return;
}
```

Step 1 is pretty trivial: using OpenCV's built-in colorspace conversion routines, I was able to efficiently convert the source image and the target lip color to Y\*Cr\*Cb\*.  The only thing worth noting here is the `convertColor` lambda created to convert a single `cv::Scalar`.  From some StackOverflow searches, this seemed the best way to accomplish the desired conversion, but it still seems a little inefficient having to create a 1x1 image, transform it to the desired colorspace, and then return the first pixel channel values for the transformed color.  _Why choose Y\*Cr\*Cb\* you might ask?_  Simple: the hint from the first reference I found suggested that the AutoRetouch approach used this colorspace to achieve its impressive results.

Step 2 requires some discussion.  I wanted to match the underlying distribution of lip pixels in the test image _exactly_ in the output image: i.e. if a pixel is 2 standard deviations from the mean value for lip pixels in the source image, I want that same pixel in the output image to be 2 standard deviations from the now _new_ mean value (determined by the desired lip color).  As you'll see above, there is no standard deviation being calculated or used.  _Don't we need the standard deviation?_  The answer is "no": _because I am targeting the same standard deviation over lip pixels in the output image as for the test image, the standard deviation is not used at-all by this approach._  For some justification about why this should work, you can check out Steps 6 and 7 from [here](https://www.pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/) that I quoted above.  Because the test and output image standard deviations are the same, the scaling to be applied is unity, so the standard deviation does not need to be computed.  What about the mean?  Here's the function definition for `maskedMean`:

```cpp
auto maskedMean = [](cv::Mat const &img, cv::Mat const &mask) {
  cv::Vec3f out(0, 0, 0);
  float N = 0;
  for (size_t i = 0; i < img.rows; ++i) {
    for (size_t j = 0; j < img.cols; ++j) {
      auto mpxl = mask.at<float>(i, j);
      if (mpxl > 0) {
        N += mpxl;
        cv::Vec3f pxl = img.at<cv::Vec3f>(i, j);
        for (size_t idx = 0; idx < 3; ++idx) {
          out[idx] += mpxl * pxl[idx];
        }
      }
    }
  }
  return out /= N;
};
```

Looking at the implementation, you can see that this is not a standard mean calculation, but rather a [_weighted_ mean](https://stats.stackexchange.com/a/6536).  A weighted mean is appropriate in this context because of the choice of blurred mask to smoothen the transition from non-lip to lip pixels.

Step 3 is the crucial step as I actually transform the pixels here.  I apply a simple alpha blending scheme with a twist in the way that compute the alpha-part.  The `alpha` part in the update comes from the mask weight times the mean-shifted test image lip pixel translated by the desired lip color.  This is a bit of a mouthful, I know, but the gist is that I am trying to maintain the statistical distributions of all of the Y\*Cr\*Cb\* channels in the test image, but just translated to the new desired lipstick color.  The `1-alpha` part in the alpha-blending comes from original test image pixel, itself.  Note that when the mask gives weight near 1, nearly all of the pixel's channel values will come from the translated lipstick channel values, whereas when the weight is near 0, nearly all of the values will come directly from the unaltered test image pixel.

Step 4 just does transforms back to the BGR colorspace and rescales the image to a 3-channel image with unsigned 8-bit depth in each channel.

To see how this approach works in practice, just look at the image shared just before the solution discussion.  In the image shown, a lipstick shade of "(B,G,R) = (0, 0, 155)" was chosen.  The solution loks very natural: lower luminance values from the original image and preserved in the output image (see the creases in the lips, for one, as well as the corners).  The transition from non-lip to lip looks very smooth: there is no hard line present.

What I found while tuning the blur kernel size and evaluating the effect of the weighted vs. non-weighted mean approach, I found that both were critical to making the final image look as natural as possible.  Once I had the blur, the weighted mean allowed me to smooth the transition in a way that resulted in a nearly-seamless application of lipstick to the original image.  In my opinion, compared to AutoRetouch, I think that my results compare quite favorably and I am pleased with the result.

#### Feature 2: ???

Thanks for reading!
