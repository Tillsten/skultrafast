What is skultrafast?
====================

Skultrafast stands for scikit.ultrafast and is an
python package. It aims to include everthing 
needed to analyze ultrafast data. At the moment,
it is mostly used used to analyse transient absorption
data, so this at the moment the only fully functional
part. In also includes a not well tested FROG-algorithm.

The big advantage of python is, that is quite easy 
to talk to other software. Using different optimizers
to fit data is really easy, also writing a gui in python
is much easier than most other languages.

Aims of the project
-------------------
I would include any kind of alogrithm or data structure which
comes up in ultrafast physics. Also, data expoloration
should be made easy (a.k.a. has a gui for these parts).

Users
-----
At the moment it is mostly me. I would be happy
if anyone would like to join the project! 

Published results
-----------------
Poster on the femto 10.
First papers using the software should follow soon.


Software prerequisites
=======================
To use the software, are working Python enverioment 
with numpy and scipy is necessery. The software
is written for Python 2.7, but porting to Python 3.3 
shoul be easy for the most parts.

Using the GUI introduces a lot of dependencies: It is made
with traitsui and chaco, which both have its own stack fo
depenencies. 

I am also recommending to python packages lmfit and cma-es.


Why not TIMP or the Ultrafast Spectroscopy Modelling Toolbox?
=============================================================

Mostly because i love the scientific ecosystem of python
and building my own data-analyis was quite educational.
I also think it is faster than the other packages and
most importently, the code is easier.
But i'll try to address both packages directly:

Why not TIMP/Glotaran?
----------------------
First: i think TIMP and Glotaran is awesome! 
Try it out, if you don't have yet.

But it has some warts:
I don't like R too much. Also some parts are slower
than neccersery. I also dislike that the corresponding
GUI is written is java, making changes unessercery 
complicated. 

Why not the Ultrafast Spectroscopy Modelling Toolbox?
-----------------------------------------------------
Mostly because it needs matlab and many of its toolboxes.
I could not try it out because of that. I only 
have access to matlab without toolboxes. Also i can't enjoy 
programming in Matlab after using Python for a long time.


