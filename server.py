import numpy as np
import cv2
from flask import Flask, request, jsonify
from roast_engine import RoastEngine
import io
import re
import json
import pytesseract
import requests
import os
import urllib.request
from PIL import Image
from pyzbar.pyzbar import decode as zbar_decode

app = Flask(__name__, static_folder="static", static_url_path="")

# Preload OpenCV's detector; fall back to pyzbar if needed
barcode_detector = cv2.barcode_BarcodeDetector()

# Lazily download and initialize a DB text detector for nutrition labels
DB_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/text_detection_db/db_resnet50.onnx"
DB_PATH = os.path.join("models", "db_text_detection.onnx")
text_detector = None


def get_text_detector():
    """Return a singleton DB text detector, downloading the model if needed."""
    global text_detector
    if text_detector is not None:
        return text_detector
    if not os.path.exists(DB_PATH):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        try:
            urllib.request.urlretrieve(DB_URL, DB_PATH)
        except Exception:
            return None
    try:
        td = cv2.dnn_TextDetectionModel_DB(DB_PATH)
        td.setBinaryThreshold(0.3)
        td.setPolygonThreshold(0.5)
        td.setUnclipRatio(1.7)
        td.setInputParams(
            scale=1.0 / 255,
            size=(736, 736),
            mean=(122.67891434, 116.66876762, 104.00698793),
            swapRB=True,
            crop=False,
        )
        text_detector = td
    except Exception:
        text_detector = None
    return text_detector


def detect_barcode(bgr):
    """Detect a UPC/EAN barcode returning {bbox:[x,y,w,h], upc:str?, conf:float}."""

    # 1) Try OpenCV's built-in detector
    try:
        ok, decoded, _, pts = barcode_detector.detectAndDecode(bgr)
        if ok and pts is not None and len(pts):
            pts = pts[0].reshape(-1, 2)
            x, y, bw, bh = cv2.boundingRect(pts.astype(np.float32))
            code = re.sub(r"\D", "", decoded[0] if decoded else "")
            return {"bbox": [int(x), int(y), int(bw), int(bh)], "upc": code or None, "conf": 0.9}
    except Exception:
        pass

    # 2) Morphological fallback to locate vertical bar patterns
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        grad = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1)
        grad = cv2.convertScaleAbs(grad)
        grad = cv2.morphologyEx(
            grad, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
        _, bw = cv2.threshold(
            grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        bw = cv2.morphologyEx(
            bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7)))
        cnts, _ = cv2.findContours(
            bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            if w > h * 2 and w * h > 1000:
                roi = bgr[y:y + h, x:x + w]
                code = decode_upc(roi)
                return {"bbox": [int(x), int(y), int(w), int(h)], "upc": code, "conf": 0.6}
    except Exception:
        pass

    # 3) Fallback to zbar which is slower but more forgiving
    try:
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        for obj in zbar_decode(pil):
            x, y, w, h = obj.rect
            code = re.sub(r"\D", "", (obj.data or b"").decode(
                "utf-8", errors="ignore"))
            return {"bbox": [int(x), int(y), int(w), int(h)], "upc": code or None, "conf": 0.5}
    except Exception:
        pass
    return None


def detect_nutrition_label(bgr):
    """Use a DB text detector to find dense text clusters."""
    td = get_text_detector()
    if td is None:
        return None
    try:
        polys, confs = td.detect(bgr)
    except Exception:
        return None
    if polys is None or len(polys) == 0:
        return None

    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for (poly, conf) in zip(polys, confs):
        if conf < 0.5:
            continue
        pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
        cv2.fillPoly(mask, [pts], 255)
    if cv2.countNonZero(mask) == 0:
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    mask = cv2.dilate(mask, kernel, iterations=2)
    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    if bw * bh < 0.02 * h * w:
        return None
    return {"bbox": [int(x), int(y), int(bw), int(bh)], "conf": 0.7}

# ---------- ROUTES ----------


@app.route("/")
def index():
    return app.send_static_file("index.html")

# Camera/JS reports a barcode; we fetch OFoods + roast


@app.route("/api/roast", methods=["POST"])
def roast_by_upc():
    data = request.get_json(silent=True) or {}
    upc = (data.get("upc") or "").strip()
    if not upc:
        return jsonify({"error": "UPC is required"}), 400

    product, nu_100g, nu_serv, serv = fetch_openfoodfacts(upc)
    nutrition = {"per_100g": nu_100g, "per_serving": nu_serv}
    food = build_food(product, upc, serv, nutrition, {}, None, None, None)
    roast = RoastEngine.choose_roast(RoastEngine.tags_from_nutrition(
        normalize_for_roast(food)), product.get("name"))
    food["roast"] = roast
    return jsonify(food)

# Client sends a full video frame; we detect barcode + nutrition label


@app.route("/api/detect", methods=["POST"])
def detect_frame():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file"}), 400
    pil = Image.open(io.BytesIO(f.read())).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    barcode = detect_barcode(bgr)

    label = None
    try:
        label = detect_nutrition_label(bgr)
    except Exception:
        label = None

    return jsonify({"barcode": barcode, "label": label})


# Browser uploads a cropped ROI (label or barcode frame)
@app.route("/api/analyze", methods=["POST"])
def analyze_roi():
    f = request.files.get("file")
    det_type = (request.form.get("type") or "").lower()  # 'label' or 'barcode'
    upc_hint = (request.form.get("upc") or "").strip() or None
    bbox = parse_bbox(request.form.get("bbox"))

    if not f:
        return jsonify({"error": "No file"}), 400
    pil = Image.open(io.BytesIO(f.read())).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    product = {"name": "Unknown Item", "brand": ""}
    nu_100g, nu_serv, serv = {}, {}, {"size_g": None,
                                      "size_text": None, "per_container": None}

    # 1) If barcode ROI, zbar decode + fallback OFoods
    upc = upc_hint
    if det_type == "barcode" and not upc:
        upc = decode_upc(bgr)

    if upc:
        p, n100, nserv, s = fetch_openfoodfacts(upc)
        if p:
            product = p
        if n100:
            nu_100g = n100
        if nserv:
            nu_serv = nserv
        if s:
            serv = s

    # 2) OCR label ROI regardless (to fill gaps or override)
    text = ocr_label(bgr)
    parsed = parse_label_text(text)
    # merge per_serving if found
    nu_serv = {**nu_serv, **{k: v for k,
                             v in parsed.get("per_serving", {}).items() if v is not None}}
    # infer 100g from serving if possible
    nu_100g = {**nu_100g, **{k: v for k,
                             v in parsed.get("per_100g", {}).items() if v is not None}}
    # serving info
    serv = {**serv, **parsed.get("serving", {})}
    ingredients = parsed.get("ingredients_text", "")

    labels = {
        "kind": det_type,
        "bbox": bbox,                 # [x,y,w,h] in frame coords if provided
        "confidence": parsed.get("confidence", 0.5)
    }
    food = build_food(product, upc, serv, {"per_100g": nu_100g, "per_serving": nu_serv},
                      {"ingredients_text": ingredients}, labels, text, None)

    roast = RoastEngine.choose_roast(RoastEngine.tags_from_nutrition(
        normalize_for_roast(food)), product.get("name"))
    food["roast"] = roast
    return jsonify(food)

# ---------- HELPERS ----------


def parse_bbox(s):
    try:
        j = json.loads(s or "null")
        if isinstance(j, list) and len(j) == 4:
            return [int(j[0]), int(j[1]), int(j[2]), int(j[3])]
    except Exception:
        pass
    return None


def decode_upc(bgr):
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    for obj in zbar_decode(pil):
        code = (obj.data or b"").decode("utf-8", errors="ignore")
        typ = obj.type or ""
        if typ in {"EAN13", "EAN8", "UPCA", "UPCE", "CODE128"}:
            return re.sub(r"\D", "", code)
    return None


def ocr_label(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if max(gray.shape) < 1200:
        scale = 1200.0 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    cfg = "--oem 1 --psm 6 -l eng"
    return pytesseract.image_to_string(gray, config=cfg)


def num(s):
    try:
        return float(s)
    except:
        return None


N = r"([0-9]+(?:\.[0-9]+)?)"


def parse_label_text(t):
    tl = t.lower()
    out = {"per_serving": {}, "per_100g": {}, "serving": {},
           "ingredients_text": "", "confidence": 0.5}

    # serving size + per container
    ms = re.search(r"serving size[^0-9]*"+N+r"\s*g", tl)
    if ms:
        g = num(ms.group(1))
        out["serving"]["size_g"] = g
        out["serving"]["size_text"] = ms.group(0)

    mc = re.search(r"servings? per container[^0-9]*"+N, tl)
    if mc:
        out["serving"]["per_container"] = num(mc.group(1))

    # per-serving common items
    def g(unit, key, *aliases):
        pat = r"(?:%s)[^0-9]*%s\s*%s" % ("|".join(aliases), N, unit)
        m = re.search(pat, tl)
        return num(m.group(1)) if m else None

    ps = out["per_serving"]
    ps["calories_kcal"] = num(re.search(
        r"calories[^0-9]*"+N, tl).group(1)) if re.search(r"calories[^0-9]*"+N, tl) else None
    ps["sodium_mg"] = g("mg", "sodium", "sodium")
    ps["protein_g"] = g("g", "protein", "proteins")
    ps["total_fat_g"] = g("g", "total fat", "totalfat", "fat")
    ps["saturated_fat_g"] = g("g", "saturated fat", "sat fat", "saturatedfat")
    ps["trans_fat_g"] = g("g", "trans fat", "transfat")
    ps["total_carb_g"] = g("g", "total carbohydrate", "carbohydrate", "carb")
    ps["sugars_g"] = g("g", "sugars", "sugar", "total sugars")
    ps["added_sugars_g"] = g(
        "g", "added sugars", "incl. added sugars", "includes added sugars")
    ps["fiber_g"] = g("g", "fiber", "dietary fiber")

    # ingredients text (rough)
    mi = re.search(r"ingredients?:\s*(.+)", tl)
    if mi:
        out["ingredients_text"] = mi.group(1)

    # crude confidence
    hits = sum(1 for k, v in ps.items() if v is not None)
    out["confidence"] = min(0.95, 0.2 + 0.12*hits)
    return out


def fetch_openfoodfacts(upc):
    url = f"https://world.openfoodfacts.org/api/v0/product/{upc}.json"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return ({}, {}, {}, {})
        j = r.json()
        if j.get("status") != 1:
            return ({}, {}, {}, {})
        p = j.get("product", {})
        n = p.get("nutriments", {}) or {}

        def f(k):
            try:
                return float(n.get(k, 0) or 0)
            except:
                return 0.0

        sodium_mg = 0.0
        try:
            if "sodium_serving" in n:
                sodium_mg = float(n["sodium_serving"])*1000 if float(
                    n["sodium_serving"]) < 10 else float(n["sodium_serving"])
            elif "salt_serving" in n:
                sodium_mg = float(n["salt_serving"])*400
        except:
            pass

        nu_serv = {
            "calories_kcal": f("energy-kcal_serving"),
            "protein_g": f("proteins_serving"),
            "total_fat_g": f("fat_serving"),
            "saturated_fat_g": f("saturated-fat_serving"),
            "trans_fat_g": f("trans-fat_serving"),
            "total_carb_g": f("carbohydrates_serving"),
            "sugars_g": f("sugars_serving"),
            "added_sugars_g": f("added-sugars_serving"),
            "fiber_g": f("fiber_serving"),
            "sodium_mg": sodium_mg or f("sodium_serving"),
            "cholesterol_mg": f("cholesterol_serving"),
        }
        # per 100g
        s_mg_100 = 0.0
        try:
            if "sodium_100g" in n:
                s_mg_100 = float(n["sodium_100g"])*1000
            elif "salt_100g" in n:
                s_mg_100 = float(n["salt_100g"])*400
        except:
            pass
        nu_100g = {
            "calories_kcal": f("energy-kcal_100g"),
            "protein_g": f("proteins_100g"),
            "total_fat_g": f("fat_100g"),
            "saturated_fat_g": f("saturated-fat_100g"),
            "trans_fat_g": f("trans-fat_100g"),
            "total_carb_g": f("carbohydrates_100g"),
            "sugars_g": f("sugars_100g"),
            "added_sugars_g": f("added-sugars_100g"),
            "fiber_g": f("fiber_100g"),
            "sodium_mg": s_mg_100,
            "cholesterol_mg": f("cholesterol_100g"),
        }

        serv = {"size_g": None, "size_text": p.get(
            "serving_size"), "per_container": None}
        return (
            {"name": p.get("product_name") or "Unknown Item",
             "brand": p.get("brands") or ""},
            nu_100g, nu_serv, serv
        )
    except Exception:
        return ({}, {}, {}, {})


def build_food(product, upc, serving, nutrition, extras, labels, ocr_text, barcode_bbox):
    return {
        "product": {"upc": upc, "name": product.get("name", "Unknown Item"), "brand": product.get("brand", "")},
        "serving": {
            "size_g": serving.get("size_g"),
            "size_text": serving.get("size_text"),
            "per_container": serving.get("per_container")
        },
        "nutrition": {
            "per_serving": nutrition.get("per_serving", {}),
            "per_100g": nutrition.get("per_100g", {})
        },
        "ingredients_text": extras.get("ingredients_text", ""),
        "labels": labels or {},
        "ocr_debug": ocr_text
    }


def normalize_for_roast(food):
    # Pull per_serving first; fall back to 100g
    ns = food["nutrition"].get("per_serving", {})
    n1 = food["nutrition"].get("per_100g", {})
    def pick(k): return ns.get(k) or n1.get(k) or 0
    return {
        "calories": pick("calories_kcal") or 0,
        "sugar_g": pick("sugars_g") or 0,
        "sodium_mg": pick("sodium_mg") or 0,
        "protein_g": pick("protein_g") or 0,
        "fiber_g": pick("fiber_g") or 0,
        "ingredients_count":  len([w for w in (food.get("ingredients_text") or "").split(",") if w.strip()])
    }


# === Simple object detection wrapper ===

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    detections = []

    bc = detect_barcode(img)
    if bc:
        x, y, w, h = bc["bbox"]
        detections.append({"bbox": [x, y, x + w, y + h],
                           "score": bc.get("conf", 0.0),
                           "class": "barcode"})

    lb = None
    try:
        lb = detect_nutrition_label(img)
    except Exception:
        lb = None
    if lb:
        x, y, w, h = lb["bbox"]
        detections.append({"bbox": [x, y, x + w, y + h],
                           "score": lb.get("conf", 0.0),
                           "class": "label"})

    boxes = [d["bbox"] for d in detections]
    scores = [d["score"] for d in detections]
    classes = [d["class"] for d in detections]
    return jsonify({"boxes": boxes, "scores": scores, "classes": classes})
# === End object detection wrapper ===


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6969)
