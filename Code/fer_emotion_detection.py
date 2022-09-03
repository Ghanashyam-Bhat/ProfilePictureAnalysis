import fer 
import matplotlib.pyplot as plt 
#%matplotlib inline
#https://towardsdatascience.com/the-ultimate-guide-to-emotion-recognition-from-facial-expressions-using-python-64e58d4324ff

path = "/home/gb/CDSAML/Images/1.jpg"

# Import the Images module from pillow
from PIL import Image
# Open the image by specifying the image path.
image_file = Image.open(path)
# the default
image_file.save("/home/gb/CDSAML/Images/edited.jpg", quality=25)

path2 = "/home/gb/CDSAML/Images/edited.jpg"
test_image_one = plt.imread(path2)
emo_detector = FER(mtcnn=True)
# Capture all the emotions on the image
captured_emotions = emo_detector.detect_emotions(test_image_one)
# Print all captured emotions with the image
print(captured_emotions)
plt.imshow(test_image_one)

# Use the top Emotion() function to call for the dominant emotion in the image
dominant_emotion, emotion_score = emo_detector.top_emotion(test_image_one)
print(dominant_emotion, emotion_score)