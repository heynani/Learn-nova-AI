import pymupdf

pdf_path = input("Enter the PDF file path: ")

document = pymupdf.open(pdf_path)

for page_number, page in enumerate(document, start=1):
    text = page.get_text()

    print(f"\n--- Page {page_number} ---")
    print(text)

document.close()