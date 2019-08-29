import requests, io, pytesseract
from PIL import Image
import pandas as pd

img = Image.open("/home/azima/Desktop/DOCs/bill1.jpg")
img = img.resize([img.width*3, img.height*3], Image.ANTIALIAS)
img = img.convert('L')
content = pd.DataFrame()
imagetext = pytesseract.image_to_string(img)
temp = pd.DataFrame({'Words':[imagetext]})
content.append(temp)
content.head()
print(temp)

writer = pd.ExcelWriter('/home/azima/Desktop/DOCs/bill2.xlsx')
content.to_excel(writer,'Sheet1')
