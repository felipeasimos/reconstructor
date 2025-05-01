## 3d Reconstruction of an image

### 1. Edge Detection

#### Whare are image gradients?

It is a directional change in the intensity or color in an image.
In the edge detection context, borders show a sudden change in these attributes between close pixels.
Intuition: imagine the image as a heightmap, gradients then map directly to the idea of derivatives as the inclination of a 2d function

#### Edge detection roadmap

1. edge detection
    1. grayscale
    2. canny - in the output every edge has a 1 pixel thickness
        1. gaussian blur
            * reduces noise
        2. compute gradients magnitude and direction using sobel
            * the two sobel filters compute change in the x and y directions
            * combined we can get the actual direction of change (gradient direction and magnitude)
        3. non-maximum suppresion
        4. double thresholding and edge tracking
    3. hough

### 2. Vanishing Points
