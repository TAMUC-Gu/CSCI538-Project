import cv2
import dlib
import imutils

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the glasses image
glasses = cv2.imread("glasses.png", -1)

# Define the function for face detection and landmark detection
def detect_face_landmarks(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image
    faces = detector(gray, 0)
    # Loop over the faces
    for face in faces:
        # Detect landmarks in the face
        landmarks = predictor(gray, face)
        # Extract the coordinates of the eyes
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        # Calculate the width of the glasses
        glasses_width = int(1.5 * abs(right_eye[0] - left_eye[0]))
        # Resize the glasses image
        resized_glasses = imutils.resize(glasses, width=glasses_width)
        # Calculate the position of the glasses
        glasses_x = left_eye[0] - int(0.25 * glasses_width)
        glasses_y = left_eye[1] - int(0.4 * glasses_width)
        # Add the glasses to the image
        overlay_image_alpha(image, resized_glasses[:, :, 0:3], (glasses_x, glasses_y), resized_glasses[:, :, 3] / 255.0)
    return image

# Define the function to overlay the glasses onto the image
def overlay_image_alpha(image, overlay, position, alpha_mask):
    x, y = position
    h, w = overlay.shape[:2]
    alpha = alpha_mask.reshape(h, w, 1)
    foreground = alpha * overlay[:, :, :3]
    background = (1 - alpha) * image[y:y+h, x:x+w]
    image[y:y+h, x:x+w] = cv2.add(foreground, background)

# Load the input image
image = cv2.imread("face.jpg")

# Apply face detection and landmark detection
image_with_glasses = detect_face_landmarks(image)

# Show the result
cv2.imshow("Face with glasses", image_with_glasses)
cv2.waitKey(0)
cv2.destroyAllWindows()