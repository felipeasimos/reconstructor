## 3D Reconstruction of an image (Manhattan-like)

How to run:

```
pip3 install -r requirements.txt
python3 main.py IMAGE_PATH
```

### 1. Edge Detection

#### Whare are image gradients?

It is a directional change in the intensity or color in an image.
In the edge detection context, borders show a sudden change in these attributes between close pixels.
Intuition: imagine the image as a heightmap, gradients then map directly to the idea of derivatives as the inclination of a 2d function

#### Roadmap

1. Grayscale
2. Canny - in the output every edge has a 1 pixel thickness
    1. Gaussian blur
        * Reduces noise
    2. Compute gradients magnitude and direction using sobel
        * Two Sobel filters compute change in the x and y directions
        * Combined we can get the actual direction of change (gradient direction and magnitude)
    3. Non-maximum suppresion
    4. Double thresholding and edge tracking
3. Hough

### 2. Vanishing Points


1. Get hough line intersection points
2. Cluster intersection points to get the dominant vanishing points
3. Find 3 orthogonal vanishing points

### 3. Calibration

1. Calculate omega (absolute conic)
2. Calculate K
