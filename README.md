## Overview
The chess corner prediction model is trained with a dataset contains of 5000 chessboard images.After that, yolov8 object detection model applied in order to find pieces locations and their labels(classes).This informations combined with chess corner prediction model and 2d chessboard is estimated.

## Steps
1. Detection of chessboard contour with modified yolov1 model architecture.
2. Calculation of square corners.
3. Piece detection with yolov8.
4. Combination of pieces and corresponding squares.
5. Generation FEN notation of chessboard with determined chessborad state.
6. Saving result.


## Contour Detection
Architecture:
![Screenshot](https://github.com/ilguneray/Chessboard_digitizer/tree/main/models/model_architecture.png)

In order to predict chessboard contour corners ,I have modified the fully 
connected layer of yolov1 architecture such as;
7x7x1024 --> 512 --> 8 instead of 7x7x1024 --> 4096 --> 7x7x30.
These 8 output classes represent each corner coordinate of chessboard.

https://github.com/ilguneray/Chessboard_digitizer/tree/main/models

Output can be seen below:
![Screenshot](https://github.com/ilguneray/Chessboard_digitizer/blob/main/outputs/square_centers_and_corners_1.png)



