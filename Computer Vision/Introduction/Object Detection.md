# Single Object

(Classification + Localization)

![objectdetectionsingleobject1](../../img/CV/objectdetectionsingleobject1.png)

Update:

![objectdetectionsingleobject2](../../img/CV/objectdetectionsingleobject2.png)



Final:

![objectdetectionsingleobject3](../../img/CV/objectdetectionsingleobject3.png)



#  Multiple Objects

![objectdetectionmultipleobject1](../../img/CV/objectdetectionmultipleobject1.png)

**Region Proposals:** Selective Search

 ● Find “blobby” image regions that are likely to contain objects 

● Relatively fast to run; e.g. Selective Search gives 2000 region proposals in a few seconds on CPU

# Beyond 2D Object Detection

## Object Detection + Captioning = Dense Captioning

![Dense Captioning](../../img/CV/Dense Captioning.png)

## Dense Video Captioning

![Dense Captioning](../../img/CV/ Dense Video Captioning.png)

## Objects + Relationships = Scene Graphs

![Scene Graphs](../../img/CV/Scene Graphs.png)

## Scene Graph Prediction

![Scene Graph Prediction](../../img/CV/Scene Graph Prediction.png)

## 3D Object Detection

2D Object Detection: 2D bounding box

(x, y, w, h)

3D Object Detection:
 3D oriented bounding box

(x, y, z, w, h, l, r, p, y) Simplified bbox: no roll & pitch

Much harder problem than 2D object detection!

![3D Object Detection](../../img/CV/3D Object Detection.png)

## 3D Object Detection: Simple Camera Model

A point on the image plane corresponds to a **ray** in the 3D space

A 2D bounding box on an image is a **frustrum** in the 3D space

Localize an object in 3D:
 The object can be anywhere in the **camera viewing frustrum**!



























# Reference 

[cs231 **Detection and Segmentation** Semantic segmentation Object detection Instance segmentation](http://cs231n.stanford.edu/slides/2020/lecture_12.pdf)

