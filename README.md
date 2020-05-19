# Project-GOLFIE-Advances-in-Robot-Learning

ABSTRACT—

Despite the evolution in robotics, learning robot manipulation task is still an extant challenge in the ﬁeld of robotics. 
This is mainly due to the complexity involved in encoding and computing multiple degrees of freedom in the learning process. 
In this project, we implement a simple golf playing robot which hits a golf ball into a target hole. 
The idea is to learn the amount of torque to be applied on the golf club based on the distance between the golf ball and 
the target hole such that the ball enters the hole. The distance between the golf ball and the hole is estimated using object detection 
and contour area techniques.

METHODOLOGY-
For the sake of simplicity, in this project, we set up our environment in such a way that the golf ball, the golf club and the target 
hole are all linearly arranged. Therefore, we establish that when the golf club nudges/hits the ball, 
it rolls down linearly in the direction of the hole. We achieved this by restricting the motion of the club with a single degree of freedom. 
We have three components in our project, 
a) Computer vision component: to estimate distances, 
b) Learning component: to learn the amount of torque to be applied and the success/failure of a hit and 
c) Robot Manipulation component: to execute the process of hitting the golf ball into the hole.

RUN THE GOLFIE.PY FILE
