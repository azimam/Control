from wand.image import Image as wi

# Converting first page into JPG
#with Image(filename="/home/azima/Desktop/DOCs/Statement3_from_CHEM_MARK_OF_SAN_ANTONIO_INC17808.pdf[0]") as img:
#    img.save(filename="/home/azima/Desktop/DOCs/test.jpg")

pdf=wi(filename="/home/azima/Desktop/DOCs/Statement3_from_CHEM_MARK_OF_SAN_ANTONIO_INC17808.pdf",resolution=700)

pdfImage=pdf.convert("jpeg")

i=1
for img in pdfImage.sequence:
    page=wi(image=img)
    page.save(filename="/home/azima/Desktop/DOCs/bill"+str(i)+".jpg")
    i=+1
