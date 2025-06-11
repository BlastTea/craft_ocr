import cv2
import pytesseract
import numpy as np
import craft_text_detector.craft_utils as craft_utils

# Konfigurasi Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def safe_adjust(polys, ratio_w, ratio_h, ratio_net=2):
    adjusted = []
    for poly in polys:
        if poly is not None:
            adjusted.append(np.array(poly) * (ratio_w * ratio_net, ratio_h * ratio_net))
    return adjusted

craft_utils.adjustResultCoordinates = safe_adjust

from craft_text_detector import Craft

# Muat CRAFT
craft = Craft(output_dir=None, crop_type="poly", cuda=False)

# Baca gambar
orig = cv2.imread("ch4_training_images/img_59.jpg")

# Deteksi teks
prediction = craft.detect_text(orig)
boxes = prediction["boxes"]

# Lepas model dari memori
craft.unload_craftnet_model()
craft.unload_refinenet_model()

boxes_list = []
for poly in prediction["boxes"]:
    arr = np.array(poly)       # shape (4,2)
    xs, ys = arr[:,0], arr[:,1]
    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max(), ys.max()
    # Cast ke Python int
    boxes_list.append((int(x1), int(y1), int(x2), int(y2)))

# OCR per-ROI
results = []
for (sX, sY, eX, eY) in boxes_list:
    roi = orig[sY:eY, sX:eX]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh, config="--oem 1 --psm 7")
    results.append(((sX, sY, eX, eY), text.strip()))

# Cetak teks
for ((sX, sY, eX, eY), text) in results:
    print(f"Box: {(sX, sY, eX, eY)} -> {text}")
    
# Tampilkan hasil kotak
output = orig.copy()
for ((sX, sY, eX, eY), text) in results:
    cv2.rectangle(output, (sX, sY), (eX, eY), (0, 255, 0), 2)
cv2.imshow("CRAFT Detected Regions", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

