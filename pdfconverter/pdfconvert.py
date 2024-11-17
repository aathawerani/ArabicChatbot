from multilingual_pdf2text.pdf2text import PDF2Text
from multilingual_pdf2text.models.document_model.document import Document
import logging
logging.basicConfig(level=logging.INFO)

def main():
    ## create document for extraction with configurations
    print("calling pdf_document")
    pdf_document = Document(
        document_path='D:/aaht14/freelance/Taqi/pdfs/arabic-manasek-1444.pdf',
        language='ara'
        )
    print("calling pdf2text")
    pdf2text = PDF2Text(document=pdf_document)
    print("calling pdf2text extract")
    content = pdf2text.extract()
    #print(content)
    print("opening file")
    f = open("demofile2.txt", "w", encoding="utf-8")
    print("writing file")
    for details in content:
        for key, value in details.items():  
            f.write('%s:%s\n' % (key, value))
    print("closing file")
    f.close()

if __name__ == "__main__":
    main()