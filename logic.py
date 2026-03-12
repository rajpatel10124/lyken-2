import os
import re
import hashlib
import numpy as np
import PyPDF2
import pytesseract
from PIL import Image
import faiss
import nltk

from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from rapidfuzz.fuzz import ratio, token_sort_ratio
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Download required NLTK data silently
for _pkg in ('stopwords', 'punkt', 'punkt_tab'):
    try:
        nltk.data.find(f'tokenizers/{_pkg}' if 'punkt' in _pkg else f'corpora/{_pkg}')
    except LookupError:
        nltk.download(_pkg, quiet=True)

_STOPWORDS = set(stopwords.words('english'))


# -------------------------------------------------
# OPTIONAL: TESSERACT PATH (WINDOWS ONLY)
# -------------------------------------------------
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".png", ".jpg", ".jpeg")


# -------------------------------------------------
# LOAD SEMANTIC MODEL (AI BRAIN)
# -------------------------------------------------
print("[INFO] Loading semantic model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("[INFO] Model loaded.")


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
        text = pytesseract.image_to_string(img)
        return clean_text(text)

    except Exception as e:
        print(f"[ERROR] OCR failed: {path} -> {e}")
        return ""


# -------------------------------------------------
# PDF TEXT EXTRACTION + OCR FALLBACK
# -------------------------------------------------
def extract_pdf_text(path):
    text = ""

    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)

            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + " "

        # OCR fallback
        if not text.strip():
            print("[INFO] No text found. Using OCR...")
            images = convert_from_path(path)

            for img in images:
                text += pytesseract.image_to_string(img) + " "

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

    embeddings = model.encode(
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

    q_embed = model.encode([query], convert_to_numpy=True).astype("float32")
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
# CONTENT WORD EXTRACTION (strips stopwords)
# -------------------------------------------------
def _content_words(text):
    """Extract meaningful words only — strips stopwords and short tokens.
    These words survive OCR noise better than character-level comparison."""
    return {w for w in text.split() if w not in _STOPWORDS and len(w) > 2}


# -------------------------------------------------
# N-GRAM OVERLAP
# -------------------------------------------------
def _ngram_overlap(words1, words2, n=2):
    """Word-level n-gram (bigram/trigram) Jaccard overlap.
    Captures shared PHRASES — a strong signal for copied content."""
    words_list1 = list(words1)
    words_list2 = list(words2)
    ng1 = set(zip(*[words_list1[i:] for i in range(n)]))
    ng2 = set(zip(*[words_list2[i:] for i in range(n)]))
    if not ng1 and not ng2:
        return 0.0
    union = ng1 | ng2
    return len(ng1 & ng2) / len(union) if union else 0.0


# -------------------------------------------------
# SENTENCE-LEVEL SEMANTIC MATCHING
# -------------------------------------------------
def _sentence_level_similarity(text1, text2):
    """
    Splits both texts into sentences and computes cross-sentence semantic
    similarity. Returns the average top-match score for each sentence in
    text1 against all sentences in text2.

    This is OCR-noise resistant: even if some sentences are garbled,
    sentences with the same meaning will still match semantically.
    """
    sents1 = [s.strip() for s in re.split(r'[.!?]', text1) if len(s.strip()) > 15]
    sents2 = [s.strip() for s in re.split(r'[.!?]', text2) if len(s.strip()) > 15]

    if not sents1 or not sents2:
        return 0.0

    # Encode all sentences at once (efficient batch)
    all_sents = sents1 + sents2
    embs = model.encode(all_sents, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embs)

    embs1 = embs[:len(sents1)]
    embs2 = embs[len(sents1):]

    # For each sentence in text1, find best match in text2
    sim_matrix = np.dot(embs1, embs2.T)  # shape: (len1, len2)
    top_matches = sim_matrix.max(axis=1)  # best match per sentence

    # Only count matches above 0.6 to exclude unrelated sentences
    strong_matches = top_matches[top_matches > 0.6]
    if len(strong_matches) == 0:
        return 0.0

    # Proportion of sentences with a strong match
    coverage = len(strong_matches) / len(sents1)
    avg_strength = float(strong_matches.mean())

    return round(coverage * avg_strength, 4)


# -------------------------------------------------
# HYBRID SIMILARITY (HANDWRITING-ROBUST PLAGIARISM SCORE)
# -------------------------------------------------
def hybrid_similarity(text1, text2):
    """
    Robust plagiarism score for handwritten OCR documents.

    Problem with previous approach: text_evidence was weighted 55%.
    Real handwriting OCR produces very different character sequences even
    for identical content → text_evidence crashes to 0 → bad final score.

    Fix: ADAPTIVE WEIGHTING based on semantic confidence.
    - When semantic is HIGH (>=0.80): semantic dominates (70%).
      OCR noise is expected, so we trust the AI meaning score most.
    - When semantic is MEDIUM: balanced blend.
    - Sentence-level matching provides OCR-resistant secondary evidence.

    Scenario                       | semantic | sent_match | verdict
    -------------------------------|----------|------------|--------
    Same content, diff handwriting | 0.85+    | HIGH       | PLAGIARISM
    Same topic, diff answer        | 0.60-0.75| LOW        | OK
    Exact copy / typed             | 0.95+    | VERY HIGH  | PLAGIARISM
    """
    if not text1 or not text2:
        return 0.0

    text1 = clean_text(text1)
    text2 = clean_text(text2)

    if not text1 or not text2:
        return 0.0

    # ---- 1. FULL-DOCUMENT SEMANTIC SIMILARITY ----
    # Most reliable signal for handwriting — OCR noise doesn't affect meaning
    emb = model.encode([text1, text2], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(emb)
    semantic_score = float(np.dot(emb[0], emb[1]))

    # ---- 2. SENTENCE-LEVEL SEMANTIC MATCHING ----
    # Compares sentences individually — much more robust than full-doc for
    # handwriting because even garbled sentences retain meaning
    sent_score = _sentence_level_similarity(text1, text2)

    # ---- 3. CONTENT-WORD JACCARD (stopword-filtered) ----
    cw1 = _content_words(text1)
    cw2 = _content_words(text2)
    union_cw = cw1 | cw2
    content_jaccard = len(cw1 & cw2) / len(union_cw) if union_cw else 0.0

    # ---- 4. TOKEN-SORT FUZZY (order-independent, OCR-tolerant) ----
    fuzzy_score = token_sort_ratio(text1, text2) / 100.0

    # ---- 5. ADAPTIVE WEIGHTED SCORE ----
    # When semantic is HIGH → trust semantic + sentence match more.
    # When semantic is MEDIUM → use balanced blend to avoid false positives.
    if semantic_score >= 0.80:
        # High semantic confidence: same content very likely
        # Semantic + sentence matching dominate; text signals support
        final_score = (0.55 * semantic_score) + (0.30 * sent_score) + \
                      (0.10 * content_jaccard) + (0.05 * fuzzy_score)
    elif semantic_score >= 0.65:
        # Medium confidence: balance all signals
        final_score = (0.40 * semantic_score) + (0.30 * sent_score) + \
                      (0.20 * content_jaccard) + (0.10 * fuzzy_score)
    else:
        # Low semantic: require strong text evidence to flag
        final_score = (0.25 * semantic_score) + (0.25 * sent_score) + \
                      (0.30 * content_jaccard) + (0.20 * fuzzy_score)

    return round(final_score, 4)


# -------------------------------------------------
# FULL DOCUMENT COMPARISON
# -------------------------------------------------
def compare_documents(text, threshold=0.3):

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