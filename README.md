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
![Screenshot](https://github.com/ilguneray/Chessboard_digitizer/blob/main/models/model_architecture.png)

In order to predict chessboard contour corners ,I have modified the fully 
connected layer of yolov1 architecture such as;
7x7x1024 --> 512 --> 8 instead of 7x7x1024 --> 4096 --> 7x7x30.
These 8 output classes represent each corner coordinate of chessboard.


Output can be seen below:
![Screenshot](https://github.com/ilguneray/Chessboard_digitizer/blob/main/outputs/square_centers_and_corners_1.png)


## Piece Detection
In this section , yolov8 object detection model is used for detection.
Training    -> @0.5mAP = %99.5
Validation  -> @0.5mAP = %96

Example after detection can be seen below:
![Screenshot](https://github.com/ilguneray/Chessboard_digitizer/blob/main/outputs/obj_detection_1.png)


## Placement of Pieces to Corresponding Squares
To be able to calculate chess pieces location weighted center calculation is used with 3/4 rate closet to their bottom pixel values.Then these coordinates are compared with all squares centers coordinates according to MSE error.Chess pieces are placed in the square with the least error rate.As a result current state is represented in array:
```
[['--' '--' '--' 'bq' '--' '--' 'bk' '--']
 ['--' '--' '--' '--' '--' 'bp' '--' 'bp']
 ['wq' 'bp' 'wb' '--' 'bp' '--' 'bp' '--']
 ['--' '--' '--' '--' '--' '--' '--' '--']
 ['wp' '--' '--' '--' '--' 'wp' '--' '--']
 ['--' '--' 'wp' '--' '--' '--' 'wp' 'wp']
 ['--' '--' 'wp' '--' '--' '--' '--' 'wk']
 ['--' '--' '--' 'bn' '--' '--' '--' '--']]
```

 ## Fen Conversation
 Chessboard state in array format is converted to FEN format.
```
Fen: 3q2k1/5p1p/QpB1p1p1/8/P4P2/2P3PP/2P4K/3n4
```

## Saving Result
Chessboard is converted to png format and saved.
Output can be seen below:

![Screenshot](https://github.com/ilguneray/Chessboard_digitizer/blob/main/outputs/chessboard_1.png)

