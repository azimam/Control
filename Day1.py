# importing required modules 
import PyPDF2 
# creating a pdf file object 
pdfFileObj = open('/home/azima/Desktop/DOCs/Statement3_from_CHEM_MARK_OF_SAN_ANTONIO_INC17808.pdf', 'rb')
# creating a pdf reader object 
pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 
# printing number of pages in pdf file 
print(pdfReader.numPages) 
# creating a page object 
pageObj = pdfReader.getPage(0) 
# extracting text from page 
print(pageObj.extractText()) 
# closing the pdf file object 
pdfFileObj.close() 
