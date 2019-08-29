from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageChops
import pytesseract
img=Image.open("/home/azima/Desktop/DOCs/bill1.jpg")
text=pytesseract.image_to_string(img,lang='eng')
print(text)
