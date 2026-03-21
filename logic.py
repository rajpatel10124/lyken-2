import os
import re
import hashlib
import numpy as np
import PyPDF2
import pytesseract
import gc
from PIL import Image, ImageOps, ImageFilter
import faiss

from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from rapidfuzz.fuzz import ratio
from nltk.tokenize import sent_tokenize


# -------------------------------------------------
# OPTIONAL: TESSERACT PATH (WINDOWS ONLY)
# -------------------------------------------------
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".png", ".jpg", ".jpeg")


# -------------------------------------------------
# LOAD SEMANTIC MODEL (LAZY LOADING)
# -------------------------------------------------
_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        print("[INFO] Loading semantic model (This may take a moment)...")
        _model_instance = SentenceTransformer("all-MiniLM-L6-v2")
        print("[INFO] Model loaded.")
    return _model_instance


# -------------------------------------------------
# TEXT CLEANING (OCR SAFE)
# -------------------------------------------------
def clean_text(text):
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9., ]", " ", text)

    return text.strip()


# -------------------------------------------------
# HASH GENERATION
# -------------------------------------------------
def generate_hash(content):
    return hashlib.sha256(content).hexdigest()


# -------------------------------------------------
# IMAGE OCR
# -------------------------------------------------
def extract_image_text(path):
    try:
        img = Image.open(path)
        
        # --- Performance: Resize large images to reduce RAM and speed up OCR ---
        # Resize if any dimension is > 2000px
        max_dim = 2000
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # --- Preprocessing for blurred/handheld images ---
        # 1. Convert to grayscale
        img = ImageOps.grayscale(img)
        # 2. Enhance contrast
        img = ImageOps.autocontrast(img)
        # 3. Apply a slight sharpen filter
        img = img.filter(ImageFilter.SHARPEN)
        
        # Use better OCR config for handwriting: --psm 3 (Auto Page Segmentation)
        # --oem 1 (LSTM only) is generally faster and better for handwriting
        text = pytesseract.image_to_string(img, config='--psm 3 --oem 1')
        
        # Optional: try sparse text PSM 11 if PSM 3 yields too little
        if len(text.strip()) < 10:
             text = pytesseract.image_to_string(img, config='--psm 11 --oem 1')

        return clean_text(text)

    except Exception as e:
        print(f"[ERROR] OCR failed: {path} -> {e}")
        return ""
    finally:
        if 'img' in locals():
            del img
        gc.collect()


# -------------------------------------------------
# PDF TEXT EXTRACTION + OCR FALLBACK
# -------------------------------------------------
def extract_pdf_text(path):
    text = ""

    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)

            # Limit to first 10 pages for performance
            max_pages = min(10, len(reader.pages))
            for i in range(max_pages):
                page = reader.pages[i]
                t = page.extract_text()
                if t:
                    text += t + " "

        # OCR fallback only if no text and file is small
        if not text.strip() and os.path.getsize(path) < 100 * 1024 * 1024:
            print("[INFO] No text found. Using OCR (Page-by-page to save RAM)...")
            
            # Process pages one-by-one to save RAM
            # Using 150 DPI for a good speed/accuracy trade-off
            for page_num in range(1, 6): # First 5 pages
                try:
                    images = convert_from_path(path, first_page=page_num, last_page=page_num, dpi=150)
                    if images and len(images) > 0:
                        img = images[0]
                        # Use same logic as extract_image_text but on the PIL object
                        img = ImageOps.grayscale(img)
                        img = ImageOps.autocontrast(img)
                        text += pytesseract.image_to_string(img, config='--psm 3 --oem 1') + " "
                        del img
                        del images
                        gc.collect()
                except Exception as e_inner:
                    print(f"[WARNING] OCR failed on page {page_num}: {e_inner}")
                    break

    except Exception as e:
        print("[ERROR] PDF extraction:", e)

    return clean_text(text)


# -------------------------------------------------
# MAIN TEXT EXTRACTION
# -------------------------------------------------
def extract_text(file_path):

    if not os.path.exists(file_path):
        return "", None, None

    with open(file_path, "rb") as f:
        content = f.read()

    file_hash = generate_hash(content)
    text = ""

    name = file_path.lower()

    try:
        if name.endswith(".txt"):
            text = open(file_path, encoding="utf-8", errors="ignore").read()

        elif name.endswith(".pdf"):
            text = extract_pdf_text(file_path)

        elif name.endswith((".png", ".jpg", ".jpeg")):
            text = extract_image_text(file_path)

    except Exception as e:
        print("[ERROR] Extraction failed:", e)

    return clean_text(text), content, file_hash


# -------------------------------------------------
# CHUNKING (PARAGRAPH SPLIT)
# -------------------------------------------------
def split_into_chunks(text, max_len=300):

    sentences = sent_tokenize(text)

    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) < max_len:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current:
        chunks.append(current.strip())

    return chunks


# -------------------------------------------------
# GLOBAL VECTOR DATABASE
# -------------------------------------------------
faiss_index = None
stored_chunks = []
chunk_to_doc = []
document_texts = []


# -------------------------------------------------
# BUILD FAISS INDEX
# -------------------------------------------------
def build_index(all_documents):

    global faiss_index, stored_chunks, chunk_to_doc, document_texts

    print("[INFO] Building semantic index...")

    stored_chunks = []
    chunk_to_doc = []
    document_texts = all_documents

    for doc_id, text in enumerate(all_documents):

        chunks = split_into_chunks(text)

        for ch in chunks:
            stored_chunks.append(ch)
            chunk_to_doc.append(doc_id)

    if not stored_chunks:
        print("[WARNING] No text available for indexing.")
        return

    embeddings = get_model().encode(
        stored_chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)

    print(f"[INFO] Indexed {len(stored_chunks)} chunks.")


# -------------------------------------------------
# SEMANTIC SEARCH
# -------------------------------------------------
def search(query, top_k=5):

    global faiss_index

    if faiss_index is None:
        return []

    query = clean_text(query)

    q_embed = get_model().encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_embed)

    scores, indices = faiss_index.search(q_embed, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append({
                "matched_chunk": stored_chunks[idx],
                "score": float(score),
                "document_id": chunk_to_doc[idx]
            })

    return results


# -------------------------------------------------
# HYBRID SIMILARITY (REAL PLAGIARISM SCORE)
# -------------------------------------------------
def hybrid_similarity(text1, text2):

    if not text1 or not text2:
        return 0.0

    text1 = clean_text(text1)
    text2 = clean_text(text2)

    # semantic similarity
    emb = get_model().encode([text1, text2],
                       convert_to_numpy=True).astype("float32")

    faiss.normalize_L2(emb)

    semantic_score = float(np.dot(emb[0], emb[1]))

    # fuzzy OCR tolerant similarity
    fuzzy_score = ratio(text1, text2) / 100.0

    # hybrid weighted score
    final_score = (0.7 * semantic_score) + (0.3 * fuzzy_score)

    return round(final_score, 4)


# -------------------------------------------------
# FULL DOCUMENT COMPARISON
# -------------------------------------------------
def compare_documents(text, threshold=0.6):

    results = []

    matches = search(text, top_k=10)

    seen_docs = set()

    for m in matches:

        doc_id = m["document_id"]

        if doc_id in seen_docs:
            continue

        seen_docs.add(doc_id)

        score = hybrid_similarity(text, document_texts[doc_id])

        if score >= threshold:
            results.append({
                "document_id": doc_id,
                "similarity": score
            })

    return sorted(results, key=lambda x: x["similarity"], reverse=True)

# import os
# import re
# import hashlib
# import numpy as np
# import PyPDF2
# import pytesseract
# import gc
# from PIL import Image, ImageOps, ImageFilter
# import faiss

# from pdf2image import convert_from_path
# from sentence_transformers import SentenceTransformer
# from rapidfuzz.fuzz import ratio
# from nltk.tokenize import sent_tokenize
# from rapidfuzz.fuzz import ratio

# SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".png", ".jpg", ".jpeg")


# # -------------------------------------------------
# # LOAD SEMANTIC MODEL (UNCHANGED)
# # -------------------------------------------------
# _model_instance = None

# def get_model():
#     global _model_instance
#     if _model_instance is None:
#         print("[INFO] Loading semantic model...")
#         _model_instance = SentenceTransformer("all-MiniLM-L6-v2")
#         print("[INFO] Model loaded.")
#     return _model_instance


# # -------------------------------------------------
# # TEXT CLEANING (IMPROVED)
# # -------------------------------------------------
# def clean_text(text):
#     if not text:
#         return ""

#     text = text.lower()
#     text = re.sub(r"\n+", " ", text)
#     text = re.sub(r"\s+", " ", text)
#     text = re.sub(r"[^a-z0-9., ]", " ", text)

#     return text.strip()


# # -------------------------------------------------
# # OCR ERROR FIXING (NEW)
# # -------------------------------------------------
# def fix_ocr_errors(text):
#     replacements = {
#         "0": "o",
#         "1": "l",
#         "|": "l",
#         "5": "s"
#     }
#     for k, v in replacements.items():
#         text = text.replace(k, v)
#     return text


# # -------------------------------------------------
# # HASH GENERATION (UNCHANGED)
# # -------------------------------------------------
# def generate_hash(content):
#     return hashlib.sha256(content).hexdigest()


# # -------------------------------------------------
# # IMAGE OCR (OPTIMIZED)
# # -------------------------------------------------
# def extract_image_text(path):
#     try:
#         img = Image.open(path)

#         # 🔥 Adaptive resize (better speed + low RAM)
#         max_dim = 1400
#         if max(img.size) > max_dim:
#             ratio = max_dim / max(img.size)
#             new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
#             img = img.resize(new_size, Image.Resampling.BILINEAR)

#         # 🔥 Advanced preprocessing (important)
#         img = ImageOps.grayscale(img)
#         img = ImageOps.autocontrast(img, cutoff=2)
#         img = img.filter(ImageFilter.MedianFilter(size=3))

#         # 🔥 Binarization (key for handwriting)
#         img = img.point(lambda x: 0 if x < 140 else 255, '1')

#         # 🔥 Better OCR config
#         text = pytesseract.image_to_string(img, config='--oem 1 --psm 6')

#         # Retry if weak text
#         if len(text.strip()) < 20:
#             text = pytesseract.image_to_string(img, config='--oem 1 --psm 11')

#         text = fix_ocr_errors(clean_text(text))
#         return text

#     except Exception as e:
#         print(f"[ERROR] OCR failed: {path} -> {e}")
#         return ""
#     finally:
#         if 'img' in locals():
#             del img
#         gc.collect()


# # -------------------------------------------------
# # PDF TEXT EXTRACTION + OCR (OPTIMIZED)
# # -------------------------------------------------
# def extract_pdf_text(path):
#     text = ""

#     try:
#         with open(path, "rb") as f:
#             reader = PyPDF2.PdfReader(f)

#             max_pages = min(10, len(reader.pages))
#             for i in range(max_pages):
#                 page = reader.pages[i]
#                 t = page.extract_text()
#                 if t:
#                     text += t + " "

#         # 🔥 OCR fallback only if needed
#         if not text.strip() and os.path.getsize(path) < 100 * 1024 * 1024:
#             print("[INFO] Using OCR fallback (optimized)...")

#             for page_num in range(1, 6):
#                 try:
#                     images = convert_from_path(
#                         path,
#                         first_page=page_num,
#                         last_page=page_num,
#                         dpi=120   # 🔥 reduced DPI (faster)
#                     )

#                     if images:
#                         img = images[0]

#                         # Same preprocessing as image OCR
#                         img = ImageOps.grayscale(img)
#                         img = ImageOps.autocontrast(img, cutoff=2)
#                         img = img.filter(ImageFilter.MedianFilter(size=3))
#                         img = img.point(lambda x: 0 if x < 140 else 255, '1')

#                         t = pytesseract.image_to_string(img, config='--oem 1 --psm 6')
#                         text += t + " "

#                         del img
#                         del images
#                         gc.collect()

#                 except Exception as e_inner:
#                     print(f"[WARNING] OCR failed on page {page_num}: {e_inner}")
#                     break

#     except Exception as e:
#         print("[ERROR] PDF extraction:", e)

#     return fix_ocr_errors(clean_text(text))


# # -------------------------------------------------
# # MAIN TEXT EXTRACTION (UNCHANGED LOGIC)
# # -------------------------------------------------
# def extract_text(file_path):

#     if not os.path.exists(file_path):
#         return "", None, None

#     with open(file_path, "rb") as f:
#         content = f.read()

#     file_hash = generate_hash(content)
#     text = ""

#     name = file_path.lower()

#     try:
#         if name.endswith(".txt"):
#             text = open(file_path, encoding="utf-8", errors="ignore").read()

#         elif name.endswith(".pdf"):
#             text = extract_pdf_text(file_path)

#         elif name.endswith((".png", ".jpg", ".jpeg")):
#             text = extract_image_text(file_path)

#     except Exception as e:
#         print(f"[ERROR] File processing failed: {file_path} -> {e}")

#     return text, file_hash, file_path

# def build_index(texts):
#     print("[INFO] Building FAISS index...")

#     model = get_model()

#     embeddings = model.encode(texts)

#     dimension = embeddings.shape[1]

#     index = faiss.IndexFlatL2(dimension)
#     index.add(np.array(embeddings).astype("float32"))

#     # Save globally or return
#     global faiss_index
#     faiss_index = index

#     print(f"[INFO] Indexed {len(texts)} documents.")

# def hybrid_similarity(text1, text2):
#     """
#     Hybrid similarity:
#     - Semantic similarity (AI embeddings)
#     - Fuzzy similarity (string match)
#     """

#     if not text1 or not text2:
#         return 0.0

#     model = get_model()

#     # -------- Semantic Similarity --------
#     emb1 = model.encode([text1])[0]
#     emb2 = model.encode([text2])[0]

#     # Cosine similarity
#     sim_semantic = np.dot(emb1, emb2) / (
#         np.linalg.norm(emb1) * np.linalg.norm(emb2)
#     )

#     # -------- Fuzzy Similarity --------
#     sim_fuzzy = ratio(text1, text2) / 100.0

#     # -------- Hybrid Score --------
#     # You can tune weights later
#     final_score = (0.7 * sim_semantic) + (0.3 * sim_fuzzy)

#     return float(final_score)