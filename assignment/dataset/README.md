# Sport video analysis for billiard matches


## Overview
The benchmark dataset consists of 10 different video clips of individual billiard shots extracted from 4 different match videos on YouTube. For each video clip, two frames were selected corresponding to the initial situation before the shot (i.e., “frame_first.png”) and the situation after the shot (i.e., “frame_last.png”).

## Dataset structure

The dataset is organized as the following. A folder is provided for each video clip, containing:

- a clip video of a billiard shots (`gameX_clipY.mp4`);
- a `frames` folder containing the frames extracted from the video clip for the initial status of the game (`frame_first.png`) and the final status of the game (`frame_last.png`);
- a `bounding_boxes` folder containing the bounding box annotations of each image;
- a `masks` folder containing the segmentation mask annotations of each image.


## Annotation labels

The benchmark dataset has been annotated according the following categories. Each category is assigned to a unique ID:

0. Background
1. white "cue ball"
2. black "8-ball"
3. ball with solid color
4. ball with stripes
5. playing field (table)


The frames corresponding to the initial and final status of the game have been annotated with bounding boxes and segmentation masks:

- Bounding boxes are defined for every ball in the image as a rectangle defined by 5 parameters [x, y, width, height, ball category ID], where (x,y) are the top-left corner coordinates and width and height are the bounding box main dimensions; the 5th parameter is the label ID representing the type of the ball; such parameters are listed in a row, one ball per row;

- Segmentation masks are provided as grayscale mask where each pixel is assigned the corresponding category ID  (background, white cu ball, black 8-ball, solid color, striped and playing field).


Note that segmentation masks can be easily visualized as color images highlighting the different categories by mapping each category ID in a segmentation mask to a RGB color, for example:

0: (128,128,128)
1: (255, 255, 255)
2: (0,0,0)
3: (0,0,255)
4: (255,0,0,)
5: (0,255,0)
