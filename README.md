# Sport video analysis for billiard matches
Final project - June + July 2024.

# Authors
- Federico Pivotto, 2121720
- Fabrizio Genilotti, 2119281
- Leonardo Egidati, 2106370

# Instructions

## Compilation
To compile the computer vision system you have to:
1. Create the `/build` directory in the root directory of the repository;
2. Enter in the `/build` directory;
3. Run the command `cmake ..` from terminal;
4. Build the project running the command `make`.

## Execution
To run the built computer vision system you have to:
1. Enter in the `/build` directory;
2. Run the command `./system` from terminal.

## Output
At the end of the execution of the computer vision system, the generated outputs are saved in the `/result` directory.

For each video sub-directory `/gameX_clipY` included in `/dataset` is generated an output directory `/gameX_clipY` in the `/result` directory, containing the following sub-directories:
- `/frames`: first and last frame of the video;
- `/edge_detection`: first and last frame with border detection of the playing field;
- `/object_detection`: first and last frame with ball detection and classification;
- `/bounding_boxes`: text file with list of ball bounding boxes detected in the first and last frame;
- `/segmentation`: first and last frame with segmentation of the playing field;
- `/masks`: first and last frame with mask segmentation of the playing field and surrounding environment;
- `/output`: first and last frame given by the union of edge detection and segmentation;
- `/minimap`: first and last frame with 2D top-view visualization map;
- `/metrics`: mAP and mIoU metrics measured on the first and last frame.
- Video with 2D-top-view visualization map.