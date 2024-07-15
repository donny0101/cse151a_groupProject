# cse151a_groupProject
summer session 1  2024

## Abstract:
The ISIC 2024 Skin Cancer Detection with 3D Total Body Photos (TBP) dataset aims to help clinicians in determining if patients are developing or have skin cancer to help with early detection of the disease. The dataset includes images of skin lesions which vary in range of severity from benign to malignant. The goal of our project is to create a model which can determine the probability (between 0 and 1) that the pictured skin lesion is malignant. To do so, we plan to explore pre-trained CNN models such as ResNet or ImageNet to train on the given train and test images. We will also explore building our own CNN model from scratch. We also plan to employ ensemble learning, combining multiple models together, to yield better performance for predicting probability that the case is malignant. 

## Data Exploration: 
Our initial data exploration was more focused on the metadata associated with the images than the images themselves. To begin with, distributions of features deemed most critical were plotted in order to understand existing trends or biases. An immediate trend noticed was that there were significantly more males than females present in the dataset. 

<img src="male_female.png" width="700"> <br>

We can also see that most of the samples in the data are from regions outside the head. The head is underrepresented in the dataset and might be difficult to predict from our model. 

<img src="region.png" width="700"> <br>

Additionally, the correlation matrix and corresponding dendrogram showed us that we will have to construct some form of multivariate "index" variables to reasonable analyze the data. A lot of features are strongly related and measure similar biological characteristics. The orange cluster in the dendrogram contains multiple clear examples of this such as tbp_lv_x, tpb_lv_y, and tbp_lv_z. PCA might be a very useful tool for this analysis. 

<img src="corr_matrix.png" width="700"> <br>
<img src="dendrogram.png" width="700"> <br>

The pairplot showcases a number of potential correlations, including some that seem directly linear. These features are worth analyzing and might even be worth preserving as is. 

<img src="pairplot_high_res.png" width="700"> <br>

## For preprocessing we plan to implement the following:

Through EDA we have seen that there is a great class imbalance, so in turn we plan to apply affine transformations to our images using the keras ImageDataGenerator which makes batches of  tensor image data with real-time data augmentation. More simply it will create variation of data. We will implement transformations such as rotation, brightness, shear, zoom in, and reflection. <br>

<img src="https://github.com/donny0101/cse151a_groupProject/blob/main/image%20augmentation.png" width="700"> <br>
Continuing there are some preprocessing techniques that are standard for skin cancer data sets such as the one we are using. Most notably we plan to implement enhancement and hair removal methods (time permitting). More specifically for enhancement we plan to use histogram equalization for contrast optimisation through OpenCV.
As for hair removal we will implement a common hair removal technique called generalized grayscale morphological closure operation . This process happens by performing a dilation followed by an erosion on a grayscale image. We found it in published research that summarized preprocessing techniques for skin cancer data (IEEE, 2019).<br>
### Here are the steps: <br> 
#### Grayscale Morphological Closure: <br>
- Convert the image to grayscale.
- Apply a morphological closure operation to enhance dark regions, typically corresponding to hair.
#### Identifying Thin and Long Structures:
- Create a binary image by thresholding the closed image.
- Use connected component analysis to identify regions.
- Filter regions based on their aspect ratio to identify thin and long structures.
#### Bilinear Interpolation:
- Create a mask for identified hair pixels.
- Use surrounding non-hair pixels to interpolate and replace hair pixels with smoothed values. <br>

<img src="https://github.com/donny0101/cse151a_groupProject/blob/main/hair_removal.png" width="500"> <br>
Hair removal can significantly help our model classification accuracy as they act as physical noise. 

Finally depending on the model we decide to make we may try segmentation techniques, however like hair removal it will be time permitting. 

All these methods should set up our data very well as long as we are mindful not to over augment the images. 

## Reference 

Image pre-processing in Computer Vision Systems for melanoma detection | IEEE conference publication | IEEE xplore. (n.d.-a). https://ieeexplore.ieee.org/document/8621507/ 
