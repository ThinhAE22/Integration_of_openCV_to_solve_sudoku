import cv2
import numpy as np
from tensorflow.keras.models import load_model

def initializePredectionModel():
    model = load_model('./SudokuImage.jpg')
    return model

#1 Pre Process the image
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11,2)
    return imgThreshold

#3 reorder pts
def reorder(myPoints):
    # Reshape the points into a 4x2 array (4 points with x and y coordinates)
    myPoints = myPoints.reshape((4, 2))
    
    # Create an empty array for reordered points
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    
    # Calculate sum of points to find the top-left and bottom-right
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]  # Top-left
    myPointsNew[3] = myPoints[np.argmax(add)]  # Bottom-right
    
    # Calculate difference to find top-right and bottom-left
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]  # Top-right
    myPointsNew[2] = myPoints[np.argmax(diff)]  # Bottom-left
    
    return myPointsNew

#3 Finding the biggest contour 
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    
    for i in contours:
        area = cv2.contourArea(i)  # ✅ Corrected parameter
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:  # ✅ Fixed typo
                biggest = approx
                max_area = area
    
    return biggest, max_area  # ✅ Moved outside loop

#4 Splitbox
def splitBoxes(img):
    # Split the image into 9 rows
    rows = np.vsplit(img, 9)
    boxes = []
    
    for r in rows:
        # Split each row into 9 columns
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    
    return boxes

def initializePredictionModel():
    model = load_model('./myModel.h5')
    return model

# getPrediction
def getPrediction(boxes, model):
    result = []
    for image in boxes:
        # Convert the image to a numpy array
        img = np.asarray(image)

        # Remove borders (optional, depending on your image preprocessing)
        img = img[4:img.shape[0]-4, 4:img.shape[1]-4]

        # Resize the image to 28x28 pixels for MNIST model
        img = cv2.resize(img, (28, 28))

        # Normalize the image to range [0, 1]
        img = img / 255.0

        # Reshape the image for the model (batch size of 1, 28x28x1 for grayscale)
        img = img.reshape(1, 28, 28, 1)

        # Get model prediction
        predictions = model.predict(img)

        # Get the index of the class with the highest probability
        classIndex = np.argmax(predictions, axis=-1)
        probabilityValue = np.amax(predictions)

        # Print prediction details for debugging
        print(f"Predicted Class: {classIndex[0]}, Probability: {probabilityValue}")

        # Save to result if probability is higher than threshold (e.g., 0.8)
        if probabilityValue > 0.8:
            result.append(classIndex[0])
        else:
            result.append(0)  # 0 can represent a failed prediction (empty box)

    return result

#4&5 Display Number
def displayNumbers(img, numbers, color=(0, 255, 0)):
    secW = int(img.shape[1] / 9)
    secH = int(img.shape[0] / 9)
    
    for x in range(0, 9):
        for y in range(0, 9):
            if numbers[(y * 9) + x] != 0:  # Fix indexing to properly access the numbers list
                # Position to display the number
                cv2.putText(img, str(numbers[(y * 9) + x]),
                            (x * secW + int(secW / 2) - 10, int((y + 0.8) * secH)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, color, 1, cv2.LINE_AA)  # Fixed typo here

    return img


#### 6 - DRAW GRID TO SEE THE WARP PRESPECTIVE EFFICENCY (OPTIONAL)
def drawGrid(img):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for i in range (0,9):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0),2)
        cv2.line(img, pt3, pt4, (255, 255, 0),2)
    return img
    
#6 Stack the image in 1 window

def stackImages(imgArray, scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range(0,rows):
            for y in range(0,cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:  
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)

        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows

        for x in range(rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)

    else:
        for x in range(rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    return ver