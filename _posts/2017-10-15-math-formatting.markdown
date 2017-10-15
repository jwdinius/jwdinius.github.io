---
layout: post
title:  A Post about Math Formatting
date:   2017-10-15 07:53:16
description: trial and (mostly) error
use_math: true

---
So, for some reason I've been having a rough go at getting LaTeX expressions implemented.  [KaTeX](https://khan.github.io/KaTeX) came integrated with the al-folio theme.  It looked like it might foot the bill, so I tried it out.  The documentation showed it rendered faster than [MathJax](https://www.mathjax.org), so that'd be great; a number of things popped out pretty quickly:
* It is not as widely used as MathJax.
* Rendering matrices was not simple.  I stumbled onto [this thread](https://github.com/Khan/KaTeX/issues/674).  Funny, this thread references many things, such as ```doctype```, with no definition to or reference regarding what the hell these things were.
* Inline vs. display modes were not easily distinguished or implemented.

So, it may be slow, but MathJax has a wider usage and seems to render the types of objects I care about so, following [this](http://haixing-hu.github.io/programming/2013/09/20/how-to-use-mathjax-in-jekyll-generated-github-pages/), I updated my theme to include MathJax support.  Let's try it out, shall we?

Let's try something simple, like an inline argument ```$x = 1$```: $x = 1$. :ballot_box_with_check:

How about something fancy in a block, like a fancy triangle inequality variant ``` $ (\mathbf{x} \cdot \mathbf{x} + \mathbf{y} \cdot \mathbf{y}) \leq (\mathbf{x} \cdot \mathbf{x})^2 + (\mathbf{y} \cdot \mathbf{y})^2 $```:
$ (\mathbf{x} \cdot \mathbf{x} + \mathbf{y} \cdot \mathbf{y})^2 \leq (\mathbf{x} \cdot \mathbf{x})^2 + (\mathbf{y} \cdot \mathbf{y})^2 $ :ballot_box_with_check:

What about in display mode?
```
<center>
$$ (\mathbf{x} \cdot \mathbf{x} + \mathbf{y} \cdot \mathbf{y})^2 \leq (\mathbf{x} \cdot \mathbf{x})^2 + (\mathbf{y} \cdot \mathbf{y})^2 $$
</center>
```
<center>
$$ (\mathbf{x} \cdot \mathbf{x} + \mathbf{y} \cdot \mathbf{y})^2 \leq (\mathbf{x} \cdot \mathbf{x})^2 + (\mathbf{y} \cdot \mathbf{y})^2 $$
</center> :ballot_box_with_check:

Looks good.  What I care more about is rendering matrices.   ```bmatrix``` and ```pmatrix```.  Let's try it out, shall we?

<center>
$$ \begin{pmatrix} 
a & b \\ 
c & d 
\end{pmatrix} $$
</center> :ballot_box_with_check: