

# Run on Google Colab

python main.py https://digitalcollections.nypl.org/items/4f7822b0-c00d-0136-5411-11376e3c248e?canvasIndex=0 --batch-size 8 --workers 4 --nypl-csv --download --surya-ocr

# Run locally
python main.py https://digitalcollections.nypl.org/items/4f7822b0-c00d-0136-5411-11376e3c248e?canvasIndex=0 --select-pages 



python main.py https://digitalcollections.nypl.org/items/4f7822b0-c00d-0136-5411-11376e3c248e?canvasIndex=0  --generate-prompts 

python main.py https://digitalcollections.nypl.org/items/4f7822b0-c00d-0136-5411-11376e3c248e?canvasIndex=0 --gemini-ocr 


python main.py https://digitalcollections.nypl.org/items/4f7822b0-c00d-0136-5411-11376e3c248e?canvasIndex=0 --batch-size 8 --workers 4 --nypl-csv --align-ocr --force


python main.py https://digitalcollections.nypl.org/items/4f7822b0-c00d-0136-5411-11376e3c248e?canvasIndex=0 --gemini-ocr --review-alignment --extract-entries --geocode --map

python pipeline/export_entry_boxes.py https://digitalcollections.nypl.org/items/4f7822b0-c00d-0136-5411-11376e3c248e?canvasIndex=0 \
    --base-url https://hadro.github.io/traveler-book-iiif-test \
    --update-manifest \
    --force

pipeline/export_annotations.py  https://digitalcollections.nypl.org/items/4f7822b0-c00d-0136-5411-11376e3c248e?canvasIndex=0