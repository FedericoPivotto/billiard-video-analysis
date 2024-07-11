# Sport video analysis for billiard matches
Final project - June + July 2024.

# Authors
- Federico Pivotto, 2121720
- Fabrizio Genilotti, 2119281
- Leonardo Egidati, 2106370

# TODO
- [ ] Federico: object detection
- [ ] Fabrizio: 2D top-view minimap
- [ ] Leonardo: metrics

# Steps
The project will be divided in steps according to the following schedule.

## 1 - Object detection
This step consists in identifying the bounding boxes of all the billiard balls divided in the classes.

The information extracted from this process will be used in the Step 3.

**Input**: first video frame.

**Output**: frame text file (as frame_number_bbox.txt), object detection for the first video frame without the table borders.

**Procedure the first frame**:
1. Create in a proper directory a text file for the first frame
2. Identify each ball using Viola and Jones approach
3. For each bounding box, save the identified portion of the image
4. For each portion:
   1. Make the background black
   2. Identify the ball class:
      - 1: White ball - white is the predominant color
      - 2: Black ball - black is the predominant color
      - 3: Solid ball - color, except white and black, is the predominant color
      - 4: Stripe ball - both white and color are predominant colors 
   3. Append a row ```[x, y, width, height, ball category ID]``` to the frame text file
5. Close the frame text file

**For the first video frame**:
1. Open the filled frame text file for first frame, copy the first frame and superimpose on it all the identified bounding boxes

## 2 - Border identification
This step consists in detecting the table borders.

The information extracted from this process will be used in the Step 3 and Step 4.

**Input**: object detection for the first video frame without the table borders.

**Output**: object detection for the first video frame with the table borders.

**Procedure for the first frame**:
1. Apply to the original frame the Canny Edge Detector
2. For each table edge, properly tune and apply the Linear Hough Transform
3. Identify the four table corners as intersection of the four edges
4. Superimpose the table borders to the given object detection frame
4. Save output image

## 3 - Segmentation of playing field
This step consist in two main sub-steps: first, divide the playing field only into 2 types of regions (ball and non-ball, i.e. playing field), second, assign to each pixel of the segmented playing field the relative class (i.e. the relative color) by means of the output of Step 1 (i.e. ball bounding boxes + relative class).

**Input**: frame text file (as frame_number_bbox.txt) of the first video frame.

**Output**: segmented frame (table and balls) with table borders

**Procedure**:
1. Superimpose the table borders on a copy of the first video frame
2. Open the frame text file
3. For each bounding box (row):
   1. Identify its center and ray
   2. Color the circle identified according to class
4. Color table pixels within the table borders except those with ball class colors assigned in the previous step
5. Save output image

## 4 - Object tracking and 2D map-view
This step consists in generating a 2D Map-view starting from the borders of the playing field, in particular by considering its relative corners (computed using output of Step 2). Each Map-view is obtained by the current video-frame on which is applied the colored mask obtained from Step 3. In order to track in the current Map-view an eventual change of the balls position, in the original frame track each ball in the playing field by checking whether its position (i.e. bounding box) has been updated (according to a tracker of opencv) w.r.t the previous frame. If the ball position has been updated w.r.t. previous frame, then print the output of the tracker in the correct position in the image.

**Input**: frame text file (as frame_number_bbox.txt).

**Output**: 2D map-view

**Procedure**: in the mind of Fabri.

## 5 - Metrics
