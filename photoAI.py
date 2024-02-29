import sys, os
try:import cv2
except:
    os.system('pip install opencv-python')
    try:import cv2
    except:print('Error! Please Install cv2 Library!');sys.exit()
try:import numpy as np
except:
    os.system('pip install numpy')
    try:import numpy as np
    except:print('Error! Please Install numpy Library!');sys.exit()
try:from PIL import Image 
except:
    os.system('pip install pillow')
    try:from PIL import Image 
    except:print('Error! Please Install pillow Library!');sys.exit()
def help():
    print("""
Coded By NICOLA (Telegram: @black_nicola)

          
open_video(video)
save_video(filename,video)
          
open_photo(photo)
show_photo(title,photo)
save_photo(filename,photo)
          
webcam.Get_webcam_frame()
          
photo.Adjust_brightness_and_contrast(image,brightness=1,contrast=35)
photo.Adjust_grain(image)
photo.Remove_noise(image,pow=11)
photo.Increase_color(image, factor)
photo.Setting_size(image,size1,size2)
photo.Setting_size_PIL(image,size1,size2)
photo.Rotate_image(image, angle)
photo.Enhance_image(image)
photo.Reconstruct_image(image)
photo.Setting_quality(image,quality)
photo.Get_texts_in_image(image)
photo.Remove_text(image)
          
filters.Dark_filter(image)
filters.Blurred_filter(image)
filters.Old_filter(image)
filters.Modern_quality(image,quality=1000)
""")
def version():
    return('1.0.0')
def open_video(video):
    return(cv2.VideoCapture(video))
def save_video(filename,video):
    fps = int(video.get(cv2.CAP_PROP_FPS));width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH));height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT));out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        out.write(frame)
    video.release();out.release();cv2.destroyAllWindows()
def open_photo(photo):
    try:return(cv2.imread(photo))
    except:return('Photo not find!')
def show_photo(title,photo):
    cv2.imshow(title,photo);cv2.waitKey(0);cv2.destroyAllWindows()
def save_photo(filename,photo):
    cv2.imwrite(filename,photo)
class webcam:
    def Get_webcam_frame():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open the webcam");exit()
        ret, frame = cap.read()
        if not ret:
            print("Could not read a frame from the webcam");exit()
        return (frame)
class photo:
    def Adjust_brightness_and_contrast(image,brightness=1,contrast=35):
        return(cv2.convertScaleAbs(image, alpha=brightness, beta=contrast))
    def Adjust_grain(image):
        sharpening_filter = np.array([[-1, -1, -1],[-1, 9, -1],[-1, -1, -1]])
        return(cv2.filter2D(image, -1, sharpening_filter))
    def Remove_noise(image,pow=11):
        return(cv2.medianBlur(image,pow))
    def Increase_color(image, factor):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_hsv[:, :, 1] = image_hsv[:, :, 1] * factor
        image_hsv[:, :, 1] = np.clip(image_hsv[:, :, 1], 0, 255)
        return(cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR))
    def Setting_size(image,size1,size2):
        return(cv2.resize(image, (int(size1), int(size2))))
    def Setting_size_PIL(image,size1,size2):
        save_photo('tmp.png',image);image_file = Image.open('tmp.png') ;image_file.resize((int(size1), int(size2)));image_file.save("tmp2.png");image=open_photo('tmp2.png');os.remove('tmp.png');os.remove('tmp2.png');return image
    def Rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2);rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0);rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR);return rotated_image
    def Enhance_image(image):
        sharpening_filter = np.array([[-1, -1, -1],[-1, 9, -1],[-1, -1, -1]]);enhanced_image = cv2.filter2D(image, -1, sharpening_filter);return enhanced_image
    def Reconstruct_image(image):
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0);laplacian_filter = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]]);reconstructed_image = cv2.filter2D(blurred_image, -1, laplacian_filter);return reconstructed_image
    def Setting_quality(image,quality):
        save_photo('tmp.png',image);image_file = Image.open('tmp.png');image_file.save("tmp2.png", quality=quality);image=open_photo('tmp2.png');os.remove('tmp.png');os.remove('tmp2.png');return image
    def Get_texts_in_image(image):
        try:import pytesseract
        except:
            os.system('pip install pytesseract')
            try:import pytesseract
            except:print('Error! Please Install pytesseract Library!');sys.exit()
        return(pytesseract.image_to_string(image))
    def Remove_text(image):
        try:import pytesseract
        except:
            os.system('pip install pytesseract')
            try:import pytesseract
            except:print('Error! Please Install pytesseract Library!');sys.exit()
        text = pytesseract.image_to_string(image)
        if text:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
            return(binary_image)
        else:
            return("No text detected in the image")
class filters:
    def Dark_filter(image):
        enhanced_image = cv2.detailEnhance(image);contrast_image = cv2.convertScaleAbs(enhanced_image, alpha=1.5, beta=0);adjusted_image = cv2.fastNlMeansDenoisingColored(contrast_image, None, 10, 10, 7, 21);return adjusted_image
    def Blurred_filter(image):
        contrast_image = cv2.convertScaleAbs(image, alpha=1.5, beta=0);blurred_image = cv2.GaussianBlur(contrast_image, (5, 5), 0);adjusted_image = cv2.fastNlMeansDenoisingColored(blurred_image, None, 10, 10, 7, 21);grain_image = cv2.addWeighted(image, 0.5, cv2.GaussianBlur(adjusted_image, (0, 0), 10), 0.5, 0);return grain_image
    def Old_filter(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);return(gray_image)
    def Modern_quality(image,quality=1000):
        image = photo.Setting_quality(image,quality=quality);image = photo.Adjust_brightness_and_contrast(image,contrast=10,brightness=2);return(photo.Enhance_image(image))
