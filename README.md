# Importing In Python Project
```
E='ALIOL.py';B=exit;A=print;from os import system as F,listdir as G
if not E in G():
	try:from requests import get
	except:
		F('pip install requests')
		try:from requests import get
		except:A('Error! Please Install Requests Library');B()
	try:
		C=get('https://mr-r0ot.github.io/photoAI_python_library/photoAI.py')
		if C.status_code!=200:A('GiHub Error!');B()
	except:A('NetWork Error!');B()
	D=open(E,'w+');D.write(C.text);D.close()
import photoAI
```


# Commands
```
photoAI.open_video(video)
photoAI.save_video(filename,video)
          
photoAI.open_photo(photo)
photoAI.show_photo(title,photo)
photoAI.save_photo(filename,photo)
          
photoAI.webcam.Get_webcam_frame()
          
photoAI.photo.Adjust_brightness_and_contrast(image,brightness=1,contrast=35)
photoAI.photo.Adjust_grain(image)
photoAI.photo.Remove_noise(image,pow=11)
photoAI.photo.Increase_color(image, factor)
photoAI.photo.Setting_size(image,size1,size2)
photoAI.photo.Setting_size_PIL(image,size1,size2)
photoAI.photo.Rotate_image(image, angle)
photoAI.photo.Enhance_image(image)
photoAI.photo.Reconstruct_image(image)
photoAI.photo.Setting_quality(image,quality)
photoAI.photo.Get_texts_in_image(image)
photoAI.photo.Remove_text(image)
          
photoAI.filters.Dark_filter(image)
photoAI.filters.Blurred_filter(image)
photoAI.filters.Old_filter(image)
photoAI.filters.Modern_quality(image,quality=1000)
```


# Exam
```
image = photoAI.webcam.Get_webcam_frame()
photoAI.show_photo('camera',image)

image = photoAI.photo.Setting_size(image,size1=900,size2=700)
image = photoAI.photo.Setting_quality(image,quality=1000)
image = photoAI.photo.Adjust_brightness_and_contrast(image,contrast=10,brightness=2)
image = photoAI.photo.Enhance_image(image)

photoAI.show_photo('output',image)
photoAI.save_photo('webcam.png',image)
```


Coded By NICOLA (Telegram: @black_nicola)
https://mr-r0ot.github.io/photoAI_python_library/
