# BOXER_auto_annotation_tool
Real time automatic creation of bounding boxes as well as salience maps around detected objects for fixed backgrounds

No model training required as it only uses difference in feature maps between the image of the object and the image of the background.

Requirements:
- cv2
- tensorflow

Notes:
- Salience maps are optional (can be toggled)
- Images with bounding box around the object as well as a text file for the annotation can be saved
- Salience maps works best with light colored, stable backgrounds
