M='Error! Please Install pytesseract Library!'
L='pip install pytesseract'
G='tmp2.png'
F='tmp.png'
D=int
C=print
import sys as H,os as B
try:import cv2 as A
except:
	B.system('pip install opencv-python')
	try:import cv2 as A
	except:C('Error! Please Install cv2 Library!');H.exit()
try:import numpy as E
except:
	B.system('pip install numpy')
	try:import numpy as E
	except:C('Error! Please Install numpy Library!');H.exit()
try:from PIL import Image
except:
	B.system('pip install pillow')
	try:from PIL import Image
	except:C('Error! Please Install pillow Library!');H.exit()
def help():C('\nCoded By NICOLA (Telegram: @black_nicola)\n\n          \nopen_video(video)\nsave_video(filename,video)\n          \nopen_photo(photo)\nshow_photo(title,photo)\nsave_photo(filename,photo)\n          \nwebcam.Get_webcam_frame()\n          \nphoto.Adjust_brightness_and_contrast(image,brightness=1,contrast=35)\nphoto.Adjust_grain(image)\nphoto.Remove_noise(image,pow=11)\nphoto.Increase_color(image, factor)\nphoto.Setting_size(image,size1,size2)\nphoto.Setting_size_PIL(image,size1,size2)\nphoto.Rotate_image(image, angle)\nphoto.Enhance_image(image)\nphoto.Reconstruct_image(image)\nphoto.Setting_quality(image,quality)\nphoto.Get_texts_in_image(image)\nphoto.Remove_text(image)\n          \nfilters.Dark_filter(image)\nfilters.Blurred_filter(image)\nfilters.Old_filter(image)\nfilters.Modern_quality(image,quality=1000)\n')
def N():return'1.0.0'
def O(video):return A.VideoCapture(video)
def P(filename,video):
	B=video;E=D(B.get(A.CAP_PROP_FPS));F=D(B.get(A.CAP_PROP_FRAME_WIDTH));G=D(B.get(A.CAP_PROP_FRAME_HEIGHT));C=A.VideoWriter(filename,A.VideoWriter_fourcc(*'XVID'),E,(F,G))
	while B.isOpened():
		H,I=B.read()
		if not H:break
		C.write(I)
	B.release();C.release();A.destroyAllWindows()
def J(photo):
	try:return A.imread(photo)
	except:return'Photo not find!'
def Q(title,photo):A.imshow(title,photo);A.waitKey(0);A.destroyAllWindows()
def K(filename,photo):A.imwrite(filename,photo)
class R:
	def Get_webcam_frame():
		B=A.VideoCapture(0)
		if not B.isOpened():C('Could not open the webcam');exit()
		D,E=B.read()
		if not D:C('Could not read a frame from the webcam');exit()
		return E
class I:
	def Adjust_brightness_and_contrast(B,brightness=1,contrast=35):return A.convertScaleAbs(B,alpha=brightness,beta=contrast)
	def Adjust_grain(B):C=E.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]);return A.filter2D(B,-1,C)
	def Remove_noise(B,pow=11):return A.medianBlur(B,pow)
	def Increase_color(C,factor):B=A.cvtColor(C,A.COLOR_BGR2HSV);B[:,:,1]=B[:,:,1]*factor;B[:,:,1]=E.clip(B[:,:,1],0,255);return A.cvtColor(B,A.COLOR_HSV2BGR)
	def Setting_size(B,size1,size2):return A.resize(B,(D(size1),D(size2)))
	def Setting_size_PIL(A,size1,size2):K(F,A);C=Image.open(F);C.resize((D(size1),D(size2)));C.save(G);A=J(G);B.remove(F);B.remove(G);return A
	def Rotate_image(B,angle):C=tuple(E.array(B.shape[1::-1])/2);D=A.getRotationMatrix2D(C,angle,1.);F=A.warpAffine(B,D,B.shape[1::-1],flags=A.INTER_LINEAR);return F
	def Enhance_image(B):C=E.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]);D=A.filter2D(B,-1,C);return D
	def Reconstruct_image(B):C=A.GaussianBlur(B,(5,5),0);D=E.array([[0,1,0],[1,-4,1],[0,1,0]]);F=A.filter2D(C,-1,D);return F
	def Setting_quality(A,quality):K(F,A);C=Image.open(F);C.save(G,quality=quality);A=J(G);B.remove(F);B.remove(G);return A
	def Get_texts_in_image(D):
		try:import pytesseract as A
		except:
			B.system(L)
			try:import pytesseract as A
			except:C(M);H.exit()
		return A.image_to_string(D)
	def Remove_text(D):
		try:import pytesseract as E
		except:
			B.system(L)
			try:import pytesseract as E
			except:C(M);H.exit()
		F=E.image_to_string(D)
		if F:D=A.cvtColor(D,A.COLOR_BGR2GRAY);I,G=A.threshold(D,128,255,A.THRESH_BINARY_INV);return G
		else:return'No text detected in the image'
class S:
	def Dark_filter(B):C=A.detailEnhance(B);D=A.convertScaleAbs(C,alpha=1.5,beta=0);E=A.fastNlMeansDenoisingColored(D,None,10,10,7,21);return E
	def Blurred_filter(B):C=A.convertScaleAbs(B,alpha=1.5,beta=0);D=A.GaussianBlur(C,(5,5),0);E=A.fastNlMeansDenoisingColored(D,None,10,10,7,21);F=A.addWeighted(B,.5,A.GaussianBlur(E,(0,0),10),.5,0);return F
	def Old_filter(B):C=A.cvtColor(B,A.COLOR_BGR2GRAY);return C
	def Modern_quality(A,quality=1000):A=I.Setting_quality(A,quality=quality);A=I.Adjust_brightness_and_contrast(A,contrast=10,brightness=2);return I.Enhance_image(A)
