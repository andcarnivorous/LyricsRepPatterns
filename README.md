
# Table of Contents

1.  [Model](#org8b62313)
2.  [Dataset structure](#orgb64e91f)



<a id="org8b62313"></a>

# Model

This is a model made with Keras and based on the VGG-16 architecture for binary classification of image data. The \*Matrices folders contain the image data to feed to the model. The weights folder contains some of the best performing weights for different classification pairs.


<a id="orgb64e91f"></a>

# Dataset structure

In order to feed data to the model, copy-paste two of the **Matrices directories of your choice in the \*training** folder. The function **flow\_from\_directory** from keras in the model.py file will take care of the rest. If you want to do a multiclass classification then you should also change some parameters in the model.py file.

