import base64
from io import BytesIO

def encode_image(img, fmt):
    if fmt.upper() == "JPG":
        fmt = "JPEG"
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

