# --- Set CUDA allocator config before importing torch ---
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# main.py
from dotenv import load_dotenv
from fastapi import FastAPI, Response, UploadFile, File, APIRouter
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline, AutoModelForCausalLM
)
from langdetect import detect
from functools import lru_cache
import torch
import json
import re
import requests
import random
import polyline
from typing import Optional, Dict, List, Any
import collections
import string
import time
from gtts import gTTS
import base64
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tempfile
import subprocess
import asyncio
import datetime
import difflib

# --- Import the Google Generative AI library ---
import google.generativeai as genai

# --- Import Google Search API ---
try:
    from googleapiclient.discovery import build
    GOOGLE_SEARCH_API_AVAILABLE = True
    print("✅ Google Search API available")
except ImportError:
    GOOGLE_SEARCH_API_AVAILABLE = False
    print("⚠️ Google Search API not available. Install with: pip install google-api-python-client")

# Import the new modules
from navigation_utils import geocode_place, get_route_details
from state_manager import SQLiteConversationStateStore
from analytics_logger import analytics_logger
from map_renderer import generate_static_map_html
from text_normalizer import normalize_transcript_with_gemma, normalize_text

# --- Add at the top, after other imports ---
from stt_fast import transcribe_audio
from nlu_parallel import run_nlu_in_parallel
from tts_fast import synthesize_reply
from text_normalizer import normalize_text as normalize_text_llm
from rag_utils import retrieve_relevant_passages
# --- Add at the top, after other imports ---
from gtts import gTTS
from faster_whisper import WhisperModel

# --- Whisper tiny model for STT (lazy load) ---
stt_model = None

def transcribe_audio(audio_path: str) -> str:
    global stt_model
    if stt_model is None:
        stt_model = WhisperModel("tiny", compute_type="float16")
    segments, _ = stt_model.transcribe(audio_path, beam_size=1)
    return " ".join([seg.text.strip() for seg in segments if seg.text.strip()])

# --- gTTS for TTS (single function) ---
def synthesize_reply(text: str, output_path: str, lang="hi"):
    tts = gTTS(text, lang=lang)
    tts.save(output_path)

# --- FIXED: Consolidated and corrected startup configuration ---
load_dotenv()  # Load .env file at the very beginning

app = FastAPI(title="NLU Pipeline with DST + TomTom + Gemini + Web Search", version="4.0-web-search-enabled")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config and API Key Setup ---
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY") or "225fbf96b8f00c921a416ca176bb42bd"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyBZc68Yb3q4czM7fIdPZNK_THhK-OZ_mrw"

# Configure Gemini API
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY environment variable not set. Gemini fallback will be disabled.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Google API Key configured for Gemini and Search")


TOMTOM_TRAFFIC_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json"
TAJ_MAHAL_COORDS = (27.1751, 78.0421) # Default user location if not provided
INTENT_MODEL_PATH = "trained_models/intent_classifier/final_model"
EMOTION_MODEL_PATH = "trained_models/emotion_classifier/final_model"
NER_MODEL_PATH = "trained_models/ner_model"
GAZETTEER_PATH = "expanded_gazetteer.json"
DISTRESS_FILE_PATH = "distress_signals.json"

# --- [NEW] CONFIDENCE THRESHOLD to handle ambiguous intents ---
CONFIDENCE_THRESHOLD = 0.30  # Lowered from 0.70 to see what models actually predict

# --- Web Search Functions ---
def is_factoid_query(query: str) -> bool:
    factoid_starts = [
        'who is', 'what is', 'when is', 'how many', 'current', 'president', 'prime minister', 'score', 'weather', 'time', 'date', 'capital of', 'ceo of', 'founder of', 'population of', 'height of', 'age of', 'meaning of', 'definition of'
    ]
    q = query.lower().strip()
    return any(q.startswith(start) or start in q for start in factoid_starts)

def is_future_event_query(query: str) -> bool:
    # Simple check for year in query and compare to current year
    import re
    match = re.search(r'(\d{4})', query)
    if match:
        year = int(match.group(1))
        now = datetime.datetime.now().year
        if year > now:
            return True
    return False

# --- Google Knowledge Graph API Helper ---
def search_knowledge_graph(query: str, api_key: str) -> Optional[str]:
    """Search Google Knowledge Graph for a direct answer."""
    url = "https://kgsearch.googleapis.com/v1/entities:search"
    params = {
        "query": query,
        "key": api_key,
        "limit": 1,
        "indent": True,
    }
    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if "itemListElement" in data and data["itemListElement"]:
            element = data["itemListElement"][0].get("result", {})
            # Try to get the description or detailedDescription
            if "detailedDescription" in element and "articleBody" in element["detailedDescription"]:
                return element["detailedDescription"]["articleBody"]
            elif "description" in element:
                return element["description"]
        return None
    except Exception as e:
        print(f"KG API error: {e}")
        return None

def search_web(query: str, num_results: int = 5, location: Optional[str] = None) -> List[Dict]:
    """Search the web using Google Custom Search API, optionally location-aware, with improved BeautifulSoup fallback for direct answers, tables, 'People also ask', and context-aware results. For factoid queries, do NOT append location."""
    # --- Factoid queries should not have location appended ---
    append_location = not is_factoid_query(query)
    search_query = query
    if append_location and location:
        search_query = f"{query} near {location}"
    # --- [NEW] Try Knowledge Graph API for factoid queries first ---
    if is_factoid_query(query) and GOOGLE_API_KEY:
        kg_answer = search_knowledge_graph(query, GOOGLE_API_KEY)
        if kg_answer:
            return [{
                "title": "Direct Answer",
                "snippet": kg_answer,
                "link": "",
                "displayLink": "Google Knowledge Graph"
            }]
    if not GOOGLE_SEARCH_API_AVAILABLE or not GOOGLE_API_KEY:
        print("⚠️ Web search not available - missing API or key")
        return []
    
    try:
        # Use Google Custom Search API with proper error handling
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        
        # Create the search request
        request = service.cse().list(
            q=search_query,
            cx="169933385e850494f",  # Custom Search Engine ID
            num=num_results
        )
        
        # Execute the request
        result = request.execute()
        
        search_results = []
        if "items" in result:
            for item in result["items"]:
                search_results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", ""),
                    "displayLink": item.get("displayLink", "")
                })
        
        print(f"🔍 Web search found {len(search_results)} results for: {search_query}")
        # After extracting search_results, prioritize official sources for sports factoids
        def is_official_source(link_or_display):
            official_domains = [
                'iplt20.com', 'bcci.tv', 'espncricinfo.com', 'icc-cricket.com', 'olympics.com', 'fifa.com', 'nba.com', 'uefa.com', 'cricbuzz.com'
            ]
            return any(domain in (link_or_display or '') for domain in official_domains)
        # Sort so that official sources come first for sports queries
        if any(word in query.lower() for word in ['ipl', 'cricket', 'score', 'match', 'tournament', 'final', 'winner', 'champion']):
            search_results = sorted(search_results, key=lambda x: not is_official_source(x.get('link', '') + x.get('displayLink', '')))
        return search_results[:num_results]
        
    except Exception as e:
        print(f"❌ Web search error: {e}")
        print(f"   Query: {search_query}")
        print(f"   API Key available: {bool(GOOGLE_API_KEY)}")
        
        # Try alternative search approach if Custom Search fails
        try:
            print("🔄 Trying improved BeautifulSoup search fallback...")
            import requests
            from bs4 import BeautifulSoup
            
            search_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(search_url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                search_results = []
                # --- 1. Try multiple selectors for direct answers (answer boxes, knowledge panels, live scores, etc.) ---
                answer_selectors = [
                    ('div', {'class': 'Z0LcW'}),  # Main answer box
                    ('span', {'class': 'hgKElc'}),  # Knowledge panel
                    ('div', {'data-attrid': 'wa:/description'}),  # Entity description
                    ('div', {'class': 'BNeawe iBp4i AP7Wnd'}),  # Common direct answer
                    ('div', {'class': 'BNeawe deIvCb AP7Wnd'}),  # For time/date
                    ('div', {'class': 'BNeawe tAd8D AP7Wnd'}),  # For time/date
                    ('div', {'class': 'BNeawe s3v9rd AP7Wnd'}),  # For weather, scores, etc.
                ]
                found_direct = False
                for tag, attrs in answer_selectors:
                    box = soup.find(tag, attrs=attrs)
                    if box and box.get_text(strip=True):
                        context = ""
                        parent = box.find_parent()
                        if parent:
                            label = parent.find('div', class_='BNeawe s3v9rd AP7Wnd')
                            if label and label.get_text(strip=True) != box.get_text(strip=True):
                                context = label.get_text(strip=True)
                        snippet = box.get_text(strip=True)
                        if context:
                            snippet = f"{context}: {snippet}"
                        search_results.append({
                            "title": "Direct Answer",
                            "snippet": snippet,
                            "link": "",
                            "displayLink": "google.com"
                        })
                        found_direct = True
                        break  # Prefer the first found direct answer
                # --- 2. Try to extract table answers (e.g., for scores, weather, etc.) ---
                if not found_direct:
                    table = soup.find('table')
                    if table:
                        # Try to extract the first row as a summary
                        rows = table.find_all('tr')
                        if rows:
                            cells = rows[0].find_all(['td', 'th'])
                            row_text = ' | '.join(cell.get_text(strip=True) for cell in cells)
                            if row_text:
                                search_results.append({
                                    "title": "Table Answer",
                                    "snippet": row_text,
                                    "link": "",
                                    "displayLink": "google.com"
                                })
                                found_direct = True
                # --- 3. Try to extract 'People also ask' answers ---
                if not found_direct:
                    paa_blocks = soup.find_all('div', class_='related-question-pair')
                    for paa in paa_blocks:
                        question = paa.find('div', class_='s75CSd')
                        answer = paa.find('div', class_='V3FYCf')
                        if question and answer:
                            q_text = question.get_text(strip=True)
                            a_text = answer.get_text(strip=True)
                            if q_text and a_text:
                                search_results.append({
                                    "title": f"People also ask: {q_text}",
                                    "snippet": a_text,
                                    "link": "",
                                    "displayLink": "google.com"
                                })
                                if len(search_results) >= num_results:
                                    break
                # --- 4. If still nothing, extract regular search results (up to num_results) ---
                if not search_results:
                    for i, result in enumerate(soup.find_all('div', class_='g')):
                        if len(search_results) >= num_results:
                            break
                        title_elem = result.find('h3')
                        snippet_elem = result.find('div', class_='VwiC3b')
                        if title_elem and snippet_elem:
                            search_results.append({
                                "title": title_elem.get_text(),
                                "snippet": snippet_elem.get_text(),
                                "link": "",
                                "displayLink": "google.com"
                            })
                # --- 5. As a last resort, scan all visible text for a likely answer using regex ---
                if not search_results and is_factoid_query(query):
                    visible_text = soup.get_text(separator='\n')
                    # Example: For 'President of India', look for 'President of India is ...' or similar
                    factoid_patterns = [
                        rf"{re.escape(query.strip('?'))} is ([A-Za-z .'-]+)",
                        rf"{re.escape(query.strip('?'))}: ([A-Za-z .'-]+)",
                        rf"([A-Za-z .'-]+) is the {re.escape(query.strip('?'))}",
                    ]
                    for pat in factoid_patterns:
                        match = re.search(pat, visible_text, re.IGNORECASE)
                        if match:
                            answer = match.group(1).strip()
                            print(f"[Factoid Regex Match] {pat} => {answer}")
                            search_results.append({
                                "title": "Regex Extracted Answer",
                                "snippet": answer,
                                "link": "",
                                "displayLink": "google.com"
                            })
                            break
                    # Log the visible text and regex attempts for debugging
                    print("[Visible Text Snippet for Debugging]:\n", '\n'.join(visible_text.split('\n')[:30]))
                # Filter for relevance: prioritize snippets that mention the main subject of the query
                keywords = [w.strip().lower() for w in query.split() if len(w) > 2]
                def relevance_score(snippet):
                    return sum(1 for k in keywords if k in snippet["snippet"].lower())
                search_results = sorted(search_results, key=relevance_score, reverse=True)
                # Return up to num_results
                return search_results[:num_results]
        except Exception as alt_e:
            print(f"❌ Alternative search also failed: {alt_e}")
            return []

def summarize_web_results(query: str, search_results: List[Dict], intent_label: str = None) -> str:
    """Use Gemini to summarize web search results, but strictly context-aware for factoid and event queries. Uses RAG for semantic retrieval."""
    if not search_results:
        return "Sorry, I couldn't find any information about that."
    if not GOOGLE_API_KEY:
        return "Sorry, I can't search the web right now due to API configuration."
    # Check for future event
    if is_future_event_query(query):
        return "Bhai, lagta hai yeh event abhi hua hi nahi hai! Jaise hi result aayega, main bata dunga."
    # For factoid/event/news queries, filter for official/news/sports sources
    official_domains = [
        'iplt20.com', 'bcci.tv', 'espncricinfo.com', 'icc-cricket.com', 'olympics.com', 'fifa.com', 'nba.com', 'uefa.com', 'cricbuzz.com',
        'wikipedia.org', 'britannica.com', 'gov.in', 'gov.uk', 'gov.au', 'data.gov', 'who.int', 'un.org', 'nasa.gov', 'noaa.gov',
        'timesofindia.indiatimes.com', 'bbc.com', 'cnn.com', 'ndtv.com', 'indiatoday.in', 'thehindu.com', 'hindustantimes.com', 'reuters.com', 'aljazeera.com', 'news18.com', 'espn.com', 'sportskeeda.com', 'cricketnext.com', 'cricket.com', 'cricketcountry.com', 'cricketworld.com', 'cricketaddictor.com', 'cricket365.com', 'cricketnmore.com', 'cricket.yahoo.net', 'yahoo.com/news', 'google.com/news', 'news.google.com'
    ]
    speculative_domains = [
        'reddit.com', 'quora.com', 'blogspot.com', 'medium.com', 'wordpress.com', 'tumblr.com', 'stackexchange.com', 'stack overflow', 'github.com', 'personal blog'
    ]
    def is_official_source(link_or_display):
        return any(domain in (link_or_display or '') for domain in official_domains)
    def is_speculative_source(link_or_display):
        return any(domain in (link_or_display or '') for domain in speculative_domains)
    # For news_query or similar, prioritize news/sports domains
    if intent_label == "news_query" or (is_factoid_query(query) and any(word in query.lower() for word in ["score", "result", "news", "update", "crash", "accident", "breaking", "latest"])):
        filtered_results = [r for r in search_results if is_official_source(r.get('link', '') + r.get('displayLink', '')) and not is_speculative_source(r.get('link', '') + r.get('displayLink', ''))]
        if filtered_results:
            search_results = filtered_results
    elif is_factoid_query(query):
        filtered_results = [r for r in search_results if is_official_source(r.get('link', '') + r.get('displayLink', '')) and not is_speculative_source(r.get('link', '') + r.get('displayLink', ''))]
        if not filtered_results:
            return "Bhai, koi official result nahi mila. Jaise hi aayega, main bata dunga."
        search_results = filtered_results
    passages = [r['snippet'] for r in search_results if r.get('snippet')]
    if not passages:
        return "Sorry, I couldn't find any information about that."
    top_passages = retrieve_relevant_passages(query, passages, top_k=3)
    context = "\n".join([f"- {p[0]}" for p in top_passages])
    today = datetime.date.today().strftime('%B %d, %Y')
    prompt = f"""
Today's date is {today}. You are Yuvi, a helpful Delhi driver assistant. A user asked: \"{query}\"\n\nHere are the most relevant web search results:\n{context}\n\nIMPORTANT:\n- Only use the information in the provided context.\n- If the answer is not present, say \"Sorry, I couldn't find the answer in the latest information.\"\n- If the event has not happened as of today, say so.\n- Be concise, Hinglish, and in a friendly Delhi driver tone.\n\nAnswer:\n    """
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    try:
        response = model.generate_content(prompt)
    except Exception as e:
        if "quota" in str(e).lower() or "ResourceExhausted" in str(e):
            print("[Gemini] Quota exhausted, returning fallback message.")
            return "Sorry, I've hit my daily limit for advanced answers (Gemini API quota exhausted). Please try again after midnight Pacific Time (when my quota resets)."
        else:
            print(f"[Gemini] Exception: {e}")
            return f"Sorry, I couldn't process your request due to an internal error: {e}"
    if response.parts:
        return response.text.strip()
    else:
        return "Sorry, I couldn't process the search results properly."

def should_use_web_search(query: str) -> bool:
    return True

# --- [NEW] Conversational Model Integration ---
CONVERSATIONAL_MODEL_PATH = "delhi_driver_conversational_model"

# --- [NEW] Smart Text Preprocessor Integration ---
try:
    from smart_text_preprocessor import SmartTextPreprocessor
    smart_preprocessor = SmartTextPreprocessor()
    SMART_PREPROCESSING_ENABLED = False  # Temporarily disabled due to duplication issues
    print("✅ Smart text preprocessor loaded successfully (disabled for now)")
except ImportError as e:
    print(f"⚠️ Smart text preprocessor not available: {e}")
    print("   Install with: pip install fuzzywuzzy python-Levenshtein nltk pyspellchecker")
    SMART_PREPROCESSING_ENABLED = False
    smart_preprocessor = None

# ------------------- Utility Functions (needed for constants below) -------------------
def normalize_phrase(phrase):
    """Strips, lowercases, and removes punctuation from a string."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", phrase.strip().lower())

def normalize_text(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())

# ------------------- Bad Phrase & Distress Keyword Setup -------------------
try:
    with open(GAZETTEER_PATH, "r", encoding="utf-8") as f:
        KNOWN_LOCATIONS = json.load(f)
except FileNotFoundError:
    KNOWN_LOCATIONS = {}
    print(f"Warning: Gazetteer file not found at {GAZETTEER_PATH}")

try:
    with open(DISTRESS_FILE_PATH, "r", encoding="utf-8") as f:
        distress_data = json.load(f)
        keyword_list = []
        if isinstance(distress_data, dict):
            keyword_list.extend(distress_data.get("panic_signal", []))
        elif isinstance(distress_data, list):
            keyword_list.extend(distress_data)

        DISTRESS_KEYWORDS = set(kw.lower() for kw in keyword_list)
        if not DISTRESS_KEYWORDS:
            print(f"Warning: Distress signals file at {DISTRESS_FILE_PATH} was loaded but is empty or invalid.")

except FileNotFoundError:
    DISTRESS_KEYWORDS = set()
    print(f"Warning: Distress signals file not found at {DISTRESS_FILE_PATH}")
except json.JSONDecodeError:
    DISTRESS_KEYWORDS = set()
    print(f"Error: Could not decode JSON from {DISTRESS_FILE_PATH}")

BAD_LOCATION_PHRASES = {
    "kya scene hai", "scene hai", "traffic ka", "kya haal", "ka kya scene",
    "ka kya haal", "kya hai", "kya chal raha", "batao", "ka update"
}
NORMALIZED_BAD_PHRASES = {normalize_phrase(p) for p in BAD_LOCATION_PHRASES}


# ------------------- State Manager Initialization -------------------
state_store = SQLiteConversationStateStore()

# --- Helper: Prune history to last 30 minutes ---
HISTORY_RETENTION_SECONDS = 30 * 60  # 30 minutes

def prune_history(history):
    now = time.time()
    pruned = []
    for msg in history:
        # If message has a timestamp, use it; else, keep it (for backward compatibility)
        if 'timestamp' in msg:
            if now - msg['timestamp'] <= HISTORY_RETENTION_SECONDS:
                pruned.append(msg)
        else:
            pruned.append(msg)
    return pruned


# ------------------- Transformers Pipelines -------------------
@lru_cache()
def get_intent_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_PATH)
    model.eval()
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

@lru_cache()
def get_emotion_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_PATH)
    model.eval()
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

@lru_cache()
def get_ner_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)
    model.eval()
    # Offload NER to CPU to save VRAM
    return pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True, device=-1)

@lru_cache()
def get_conversational_pipeline():
    """Load the conversational model for direct response generation"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(CONVERSATIONAL_MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(CONVERSATIONAL_MODEL_PATH)
        model.eval()
        return {"tokenizer": tokenizer, "model": model}
    except Exception as e:
        print(f"⚠️ Conversational model not found: {e}")
        return None

# ------------------- Request Schema -------------------
class TextInput(BaseModel):
    user_id: str
    text: str
    lat: Optional[float] = None
    lon: Optional[float] = None


# ------------------- Utility Functions -------------------
# [BUG FIX] The language detection is now more robust for short English words.
def detect_language(text):
    """Detects language, with specific overrides for common misclassifications."""
    lower_text = text.lower().strip()
    
    # Force classification for common English words
    if lower_text in ["yes", "ok", "okay", "no", "thanks", "hello", "hi", "bye", "start"]:
        return "hi-en"  # Force classification as our primary mixed language

    # Comprehensive Hindi/Hinglish word detection (Roman script)
    hinglish_words = [
        "preshan", "khush", "chahiye", "hai", "ke", "ka", "mein", "kar", "do", "le", "chalo", "jao", "jana", 
        "pahuchu", "rasta", "route", "map", "dikhao", "chalte", "udhar", "waha", "jaldi", "bc", "scene", 
        "aaj", "kal", "par", "se", "ko", "ki", "kya", "kahan", "kaise", "kab", "kya", "hai", "hoon", "ho",
        "main", "mera", "meri", "mere", "tum", "tumhara", "aap", "aapka", "hum", "hamara", "yeh", "woh",
        "accha", "theek", "sahi", "bilkul", "haan", "nahi", "na", "yaar", "bhai", "dost", "friend",
        "ghar", "office", "market", "school", "college", "restaurant", "hotel", "station", "airport",
        "traffic", "jam", "road", "car", "bus", "train", "metro", "time", "samay", "der", "jaldi",
        "khana", "peena", "sona", "uthna", "baithna", "chalna", "aana", "jana", "dekhna", "sunna",
        "bolna", "karna", "dena", "lena", "marna", "jeena", "marna", "roti", "dal", "chawal", "sabzi",
        # Additional Roman script Hindi words
        "kya", "hai", "hoon", "kar", "le", "jao", "chalo", "scene", "haal", "chal", "raha", "rahi", "rahe",
        "tha", "thi", "the", "hoga", "hogi", "honge", "karta", "karti", "karte", "karega", "karegi", "karenge"
    ]
    
    # Check for Hindi/Hinglish words
    if any(w in lower_text.split() for w in hinglish_words):
        return "hi-en"
    
    # Additional check for common Hinglish patterns (Roman script)
    hinglish_patterns = [
        "le chalo", "le jao", "jaana hai", "chalna hai", "kaise pahuchu", "rasta dikhao", 
        "route chahiye", "map dikhao", "mujhe le chalo", "chaliye chalte hain", "chalna hai udhar", 
        "le chaliye", "jaldi le chal", "jana hai", "waha le chlo", "route dikaho", "kya scene hai",
        "aaj ka kya scene hai", "kya haal hai", "kaise ho", "kaisa chal raha hai", "sab badhiya",
        "kya haal chaal", "kya chal raha", "scene kya hai", "haal kya hai", "kya haal chaal hai",
        "kaise ho aap", "kya haal hai bhai", "kya scene hai yaar", "sab theek hai", "kuch nahi"
    ]
    
    if any(phrase in lower_text for phrase in hinglish_patterns):
        return "hi-en"
    
    # Check for Hindi script (Devanagari) - fallback for any remaining Devanagari
    hindi_script_chars = set("अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक्षत्रज्ञड़ढ़")
    if any(char in hindi_script_chars for char in text):
        return "hi-en"
    
    try:
        lang = detect(text)
        print(f"🔍 Language detected: {lang} for text: '{text}'")
        
        # Override common misclassifications
        if lang in ["hi", "en", "ur"]:  # Add Urdu to the override list
            return "hi-en"
        
        # Additional override for short texts that might be misclassified
        if len(text.strip()) < 20 and lang not in ["hi", "en", "ur", "hi-en"]:
            # For short texts, if it contains any Hindi-like patterns, force hi-en
            if any(w in lower_text for w in ["hai", "hoon", "kar", "le", "jao", "chalo", "kya", "kaise"]):
                return "hi-en"
        
        return lang
    except Exception as e:
        print(f"⚠️ Language detection error: {e}")
        return "hi-en"  # Default to hi-en for safety

def is_affirmation(text: str) -> bool:
    """Checks if the user text is a simple affirmation."""
    text_lower = text.strip().lower()
    
    # Simple affirmations
    simple_affirmations = {
        "yes", "yep", "yeah", "ok", "okay", "sure", "fine", "alright", "definitely",
        "go ahead", "do it", "please do", "haan", "hanji", "theek hai", "chalo",
        "kar do", "start", "set karo", "laga de", "route laga do", "start navigation",
        "le chal", "le chalo", "le chal fir", "haan le chal", "theek hai chalo",
        "chaliye", "bilkul", "kar de", "laga de bhai", "haan bhai", "haan ji",
        "theek", "sahi hai", "bilkul sahi", "haan bilkul", "theek hai bilkul",
        "kar de bhai", "laga de yaar", "haan yaar", "theek hai yaar", "chalo yaar",
        "haan chalo", "theek hai chalo", "bilkul chalo", "haan bilkul chalo"
    }
    
    # Check exact matches first
    if text_lower in simple_affirmations:
        return True
    
    # Check for patterns that are clearly affirmations
    affirmation_patterns = [
        r"^haan\s+le\s+chal",  # "haan le chal"
        r"^theek\s+hai\s+le\s+chal",  # "theek hai le chal"
        r"^bilkul\s+le\s+chal",  # "bilkul le chal"
        r"^haan\s+bilkul",  # "haan bilkul"
        r"^theek\s+hai\s+bilkul",  # "theek hai bilkul"
        r"^kar\s+de\s+bhai",  # "kar de bhai"
        r"^laga\s+de\s+bhai",  # "laga de bhai"
        r"^haan\s+yaar",  # "haan yaar"
        r"^theek\s+hai\s+yaar",  # "theek hai yaar"
        r"^chalo\s+yaar",  # "chalo yaar"
    ]
    
    for pattern in affirmation_patterns:
        if re.match(pattern, text_lower):
            return True
    
    return False

# --- Name Recall Heuristic ---
NAME_QUERY_PHRASES = [
    "mera naam kya hai", "what's my name", "what is my name", "maine kya naam bataya tha",
    "do you remember my name", "naam yaad hai", "my name?", "naam kya hai", "apna naam bata", "tumhe mera naam pata hai?",
    "can you tell my name", "do you know my name", "tell me my name", "naam kya bataya tha", "naam kya bola tha", "naam kya tha",
    "who am i", "main kaun hoon", "main kaun hu", "who do you think i am", "mere baare mein kya pata hai", "meri pehchaan kya hai"
]

# In heuristic_intent_match, add name introduction patterns
NAME_INTRO_PATTERNS = [
    r"mera naam ([a-zA-Z]+) hai",
    r"my name is ([a-zA-Z]+)",
    r"main ([a-zA-Z]+) hoon"
]

def heuristic_intent_match(text):
    text_lower = text.lower().strip()
    # --- Emergency/Distress Heuristic (very strict) ---
    distress_phrases = [
        "help", "emergency", "madad", "bachaao", "save me", "accident ho gaya", "i had an accident", "i met with an accident", "i am in danger", "i need help", "please help", "urgent help", "call police", "call ambulance"
    ]
    for phrase in distress_phrases:
        if phrase in text_lower:
            return {"label": "distress", "confidence": 1.0, "source": "heuristic_distress"}
    # --- News Keyword Heuristic (robust, Hinglish/Hindi/English) ---
    news_keywords = [
        "news", "खबर", "khabar", "headlines", "breaking news", "samachar", "akhbar", "latest news", "current affairs"
    ]
    for word in news_keywords:
        if word in text_lower:
            return {"label": "news_query", "confidence": 0.99, "source": "heuristic_news_keyword"}
    # --- Greeting Heuristic (very strict, exact match only) ---
    greeting_phrases = ["hello", "hi", "namaste", "yo", "salaam"]
    for phrase in greeting_phrases:
        if phrase == text_lower:
            return {"label": "greeting", "confidence": 1.0, "source": "heuristic"}
    return None

# --- Expanded Distress Phrases ---
DISTRESS_PHRASES = [
    "gadi kharab", "gaadi kharab", "car breakdown", "car broke down", "tyre puncture", "puncture ho gaya",
    "engine problem", "engine fail", "battery dead", "help chahiye", "madad chahiye", "emergency", "accident ho gaya",
    "roadside assistance", "mechanic bulao", "towing chahiye", "gadi band ho gayi", "car not starting", "stuck on road"
]

def detect_distress(text):
    text_lower = text.lower()
    if any(kw in text_lower for kw in DISTRESS_KEYWORDS):
        return {"label": "panic_signal", "confidence": 0.99, "source": "distress_override"}
    if any(phrase in text_lower for phrase in DISTRESS_PHRASES):
        return {"label": "panic_signal", "confidence": 0.99, "source": "distress_phrase"}
    # Flexible match for accident/crash
    if "accident" in text_lower or "crash" in text_lower:
        return {"label": "panic_signal", "confidence": 0.99, "source": "distress_flexible"}
    return None

def get_tomtom_traffic(lat, lon):
    url = f"{TOMTOM_TRAFFIC_URL}?point={lat},{lon}&unit=KMPH&key={TOMTOM_API_KEY}"
    try:
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        if res.ok:
            data = res.json().get("flowSegmentData", {})
            return {
                "free_flow_speed": data.get("freeFlowSpeed"),
                "current_speed": data.get("currentSpeed"),
                "confidence": data.get("confidence"),
                "road_closure": data.get("roadClosure", False)
            }
    except requests.RequestException as e:
        print(f"Error fetching TomTom traffic: {e}")
        return None
    return None

def is_valid_location(location: str) -> bool:
    if not location or not isinstance(location, str) or location.isnumeric():
        return False
    coords = geocode_place(location)
    return coords is not None

def interpret_traffic(traffic_data: Optional[Dict]) -> Optional[Dict]:
    if not traffic_data or traffic_data.get("free_flow_speed") is None or traffic_data.get("current_speed") is None:
        return None

    if traffic_data.get("road_closure"):
        return {"level": "closed", "emoji": "🚫", "color": "black"}

    free_flow = traffic_data["free_flow_speed"]
    current = traffic_data["current_speed"]

    if free_flow == 0:
        return {"level": "heavy", "emoji": "🔴", "color": "red"}

    ratio = current / free_flow

    if ratio > 0.8:
        return {"level": "smooth", "emoji": "🟢", "color": "green"}
    elif ratio > 0.5:
        return {"level": "moderate", "emoji": "🟡", "color": "yellow"}
    else:
        return {"level": "heavy", "emoji": "🔴", "color": "red"}

# ------------------- Dialogue State Tracker -------------------
def get_state(user_id):
    return state_store.get_state(user_id)

def save_state(state):
    state_store.save_state(state)


def get_gemini_fallback_response(user_text: str, emotion: str, debug: dict, food_type: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Calls Gemini API to generate a response AND a potential follow-up action.
    Returns a dictionary with 'text' and 'action' keys.
    """
    import random as _random
    default_response = {
        "text": "Yaar, system mein kuch lafda lag raha hai. Navigation ya traffic ke baare mein kuch pooch le?",
        "action": None
    }

    if not GOOGLE_API_KEY:
        debug["fallback_source"] = "Gemini_SKIPPED_NO_KEY"
        default_response["text"] = "Yaar, system abhi thoda disconnected hai. Sorry."
        return default_response

    try:
        # --- Detect help/capability queries ---
        help_query = False
        help_patterns = [
            r"(kaise|kese) help (kar|kr) (sakta|skta) hai",
            r"kya (kar|kr) (sakta|skta|sakte|skte) ho",
            r"how can you help",
            r"what can you do",
            r"can you help",
            r"help (kar|kr) sakta hai",
            r"kya (kar|kr)te ho",
            r"kya (kar|kr)oge",
            r"kya feature hai",
            r"kya kaam aata hai",
            r"kya (kar|kr) (sakta|skta) hai yuvi",
            r"kya (kar|kr) (sakte|skte) ho yuvi",
        ]
        for pat in help_patterns:
            if re.search(pat, user_text.lower()):
                help_query = True
                break

        # --- Add randomization/context for more diverse responses ---
        randomizer = str(_random.randint(1000, 9999))
        user_context = f"UserID: {user_id or 'unknown'} | Randomizer: {randomizer}"

        # --- Always pass Yuvi's core features as context ---
        yuvi_features_hint = (
            "Yuvi's Core Features: Yuvi can help with navigation (finding and routing to places), traffic updates, ETA calculation, food and place recommendations, and general Delhi banter. Yuvi speaks in Hinglish and has a friendly, witty South Delhi bro persona."
        )

        # ##################################################################
        # #################### FIX ADDED: SMARTER PROMPTING ################
        # ##################################################################
        if food_type:
            core_task = f"""
            The user wants to eat {food_type}. Your job is to give a natural, in-character response and suggest navigating to a single, FAMOUS, REAL place in Delhi known for the best {food_type}.
            User's message: "{user_text}"
            """
        else:
            core_task = f"""
            The user said something the main system didn't understand. Your job is to give a natural, in-character response AND decide if a follow-up action can be suggested.
            User's message: "{user_text}"
            """
        # --- Add help intent hint if detected, with example and move to top ---
        if help_query:
            help_hint = (
                "IMPORTANT: The user is asking what you can do. "
                "You MUST list your main features and services (like navigation, traffic updates, recommendations, etc.) in Hinglish, as Yuvi. "
                "Be specific and friendly.\n"
                "Example: 'Main navigation, traffic update, aur recommendations de sakta hoon. Dilli ki sadkon pe kuch bhi poochh le!'\n\n"
            )
            core_task = help_hint + core_task
        # --- Always add Yuvi's core features as context ---
        core_task = yuvi_features_hint + "\n" + core_task
        # --- Add user context and anti-repetition instructions ---
        core_task += f"\n{user_context}\n"

        prompt = f"""
        You are 'Yuvi', an in-car AI assistant with a sharp, witty, and helpful South Delhi "bro" persona, operating in Delhi.

        IMPORTANT: If you are not sure about the answer, or the user's request is unclear, say "Sorry, I'm not sure about that. Can you rephrase or ask something else?" Do NOT make up facts or hallucinate. If you don't know, say so.

        Your Core Task:
        {core_task}
        Detected emotion: '{emotion}'.

        Your Persona Rules:
        1.  **Use Natural Hinglish:** Mix Hindi and English like a real person from Delhi.
        2.  **Be Empathetic & Cool:** Acknowledge their feeling, but keep it chill.
        3.  **Handle Profanity Smartly:** React with mild surprise ("Oye, aaram se bhai!") before helping.
        4.  **Keep it Short & Punchy:** Two short sentences max.
        5.  **Be Specific with Suggestions:** If you suggest a place, use a FAMOUS, REAL place name in Delhi. For food, suggest a well-known restaurant.
        6.  **Avoid repeating the same response for different users or on repeated queries.** Vary your phrasing, use different Delhi slang, and if the user has told you their name, use it in your response. If you have responded to a similar query before, try to say it differently this time.
        7.  **If the user introduces themselves (e.g., 'mera naam X hai'), remember their name and use it in your reply.**

        --- CRITICAL: Your Output MUST be a VALID JSON object ---
        You must respond with a JSON object containing two keys: "responseText" and "proposedAction".

        1.  `responseText`: (String) Your natural, in-character Hinglish response.
        2.  `proposedAction`: (JSON Object or null) The follow-up action you are suggesting.
            - If you suggest navigation, the object must be:
              `{{"intent": "nav_start_navigation", "entities": {{"location": "Famous Place Name, Area"}}}}`
            - If you are just chatting or can't suggest a concrete action, it MUST be `null`.

        Examples:
        - User wants 'chole bhature':
          Your JSON Output:
          {{
            "responseText": "Bhai, mann toh mera bhi kar gaya! Chalo, Dilli ke best chole bhature khane chalte hain. Route lagaun kya Sita Ram Diwan Chand, Paharganj ka?",
            "proposedAction": {{
              "intent": "nav_start_navigation",
              "entities": {{"location": "Sita Ram Diwan Chand, Paharganj"}}
            }}
          }}
        - User: "I'm feeling very down today" (Emotion: 'sadness')
            Your JSON Output:
            {{
            "responseText": "Arre, tension nahi le. Main hoon na yahaan tere saath. Sab a-one hai. Kahe toh koi Jagjit Singh ki ghazal lagaun, aacha lagega?",
            "proposedAction": null
            }}

        Now, generate the perfect, emotionally-aware JSON output for the user's request.
        """
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt, safety_settings={'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE'})

        if not response.parts:
            raise ValueError("Response was empty, possibly due to safety filters.")

        # Clean the response text and parse the JSON
        response_text = response.text.strip()
        
        # More robust JSON extraction - look for the outermost JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        if not json_match:
            # Fallback: try to find any JSON-like structure
            json_match = re.search(r'\{.*?"responseText".*?"proposedAction".*?\}', response_text, re.DOTALL)
        
        if not json_match:
            print(f"⚠️ Could not extract JSON from Gemini response: {response_text[:200]}...")
            raise ValueError("No valid JSON object found in the response.")

        try:
            parsed_json = json.loads(json_match.group(0))
        except json.JSONDecodeError as json_err:
            print(f"⚠️ JSON decode error: {json_err}")
            print(f"⚠️ Attempted to parse: {json_match.group(0)}")
            raise ValueError(f"Invalid JSON format: {json_err}")

        # --- GUARDRAIL: If Gemini is unsure or gives a blank/irrelevant answer, return a safe fallback ---
        safe_fallback = "Sorry, I'm not sure about that. Can you rephrase or ask something else?"
        response_text = parsed_json.get("responseText", default_response["text"])
        if not response_text or "i'm not sure" in response_text.lower() or "i do not know" in response_text.lower() or "not sure" in response_text.lower():
            response_text = safe_fallback
            parsed_json["proposedAction"] = None
        # --- LOGGING: Print/log all LLM fallback responses ---
        print(f"[LLM Fallback] User: {user_text} | Response: {response_text}")
        debug["fallback_source"] = "Gemini_Success_Structured"
        return {
            "text": response_text,
            "action": parsed_json.get("proposedAction")
        }

    except Exception as e:
        print(f"--- GEMINI API ERROR ---: {e}")
        error_type_name = type(e).__name__
        debug["fallback_source"] = f"Gemini_Error: {error_type_name}"
        if "API key not valid" in str(e):
            default_response["text"] = "Bhai, lagta hai API key mein kuch gadbad hai. Check kar le ek baar."
        elif "NotFound" in error_type_name:
            default_response["text"] = "Model load nahi ho raha. Ek baar Google Cloud project check kar lo ki 'Vertex AI API' on hai ya nahi."
        return default_response


# --- Subtle Delhi Bro Phrases (for occasional use) ---
SUBTLE_DELHI_BRO_PHRASES = [
    "Bas, Dilli ki hawa mein kuch baat hai.",
    "Yahan ka traffic aur khana, dono famous hain.",
    "Dilli ka scene alag hi hai.",
    "Yahan sab jugaad ho jata hai.",
    "Kuch bhi ho, Dilli ki vibe best hai.",
    "Yahan ke log bhi bindaas hain.",
    "Kuch plan hai toh batao, bro.",
    "Kuch naya try karein?",
    "Aaj mausam bhi sahi hai, kuch karte hain."
]

# --- Profanity/Rudeness Detection and Responses ---
RUDE_WORDS = [
    "haramzada", "haramzyada", "bhosdi", "madarchod", "behenchod", "chutiya", "gandu", "saala", "kutte", "kamina", "bitch", "bastard", "fuck", "shit", "asshole", "idiot", "stupid", "bakchod", "ullu", "ullu ka pattha", "pagal", "suar", "dog", "moron"
]
RUDE_RESPONSES = [
    "Arre bhai, aise kaun bolta hai? Chill maar, batao kya help chahiye.",
    "Bhai, dosti mein aisi baatein nahi! Batao, kahan ka scene set karna hai?",
    "Oye, language! Dilli mein bhi thoda tameez chahiye.",
    "Aaram se bhai, sab theek hai. Batao, kya chahiye?",
    "Bhai, thoda shaant ho ja. Batao, kaise madad kar sakta hoon?",
    "Dilli ka swag alag hai, par tameez bhi zaroori hai!"
]

def contains_rude(text):
    text = text.lower()
    return any(word in text for word in RUDE_WORDS)

# --- Famous Delhi Food Spots Mapping ---
FAMOUS_FOOD_SPOTS = {
    "chole bhature": "Sita Ram Diwan Chand, Paharganj",
    "momos": "Dolma Aunty Momos, Lajpat Nagar",
    "biryani": "Alkakori Alkauser, Chanakyapuri",
    "butter chicken": "Moti Mahal, Daryaganj",
    "parathe": "Paranthe Wali Gali, Chandni Chowk",
    "kebab": "Karim's, Jama Masjid",
    "golgappe": "Vaishno Chaat Bhandar, Kamla Nagar",
    "pizza": "Big Chill Cafe, Khan Market",
    "rajma chawal": "Shankar Market, Connaught Place",
    "chaat": "UPSRTC Chaat, Bengali Market",
    "paneer tikka": "Rajinder Da Dhaba, Safdarjung",
    "dal makhani": "Pind Balluchi, CP",
    "naan": "Kake Da Hotel, CP",
    "burger": "Burger Singh, Hudson Lane",
    "pasta": "Diggin, Chanakyapuri",
    "samosa": "Munirka Samosa Wala, Munirka",
    "jalebi": "Old Famous Jalebi Wala, Chandni Chowk",
    "lassi": "Amritsari Lassi Wala, Chandni Chowk",
    "kulfi": "Roshan Di Kulfi, Karol Bagh",
    "chaap": "Wah Ji Wah, NSP",
    "rolls": "Nizam's Kathi Kabab, CP",
    "shawarma": "Al Bake, NFC",
    "tandoori": "Gulati, Pandara Road",
    "malai tikka": "Rajinder Da Dhaba, Safdarjung",
    "dosa": "Sagar Ratna, Defence Colony",
    "idli": "Saravana Bhavan, CP",
    "vada pav": "Goli Vada Pav, Lajpat Nagar",
    "pav bhaji": "Kailash Parbat, CP",
    "maggi": "Tom Uncle Maggi, North Campus",
    "chai": "Chaayos, CP",
    "coffee": "Indian Coffee House, CP"
}

# --- Famous Place Coordinates Fallback ---
FAMOUS_PLACE_COORDS = {
    "India Gate": (28.6129, 77.2295),
    "Connaught Place": (28.6315, 77.2167),
    "Rajiv Chowk": (28.6328, 77.2197),
    "AIIMS": (28.5672, 77.2100),
    "Chandni Chowk": (28.6562, 77.2301),
    "Paharganj": (28.6448, 77.2167),
    "Hauz Khas": (28.5535, 77.1926),
    "Saket": (28.5222, 77.2076),
    "Karol Bagh": (28.6517, 77.1907),
    "Lajpat Nagar": (28.5677, 77.2433),
    "Sarojini Nagar": (28.5755, 77.1990),
    "Janpath": (28.6262, 77.2186),
    "Greater Kailash": (28.5485, 77.2406),
    "Qutub Minar": (28.5244, 77.1855),
    "kutub minar": (28.5244, 77.1855),  # Added spelling variation
    "Red Fort": (28.6562, 77.2410),
    "Lotus Temple": (28.5535, 77.2588),
    "Akshardham": (28.6127, 77.2773),
}

# --- Place Nicknames (moved to global scope) ---
PLACE_NICKNAMES = {
    "cp": "Connaught Place",
    "gk": "Greater Kailash",
    "nfc": "New Friends Colony",
    "north campus": "North Campus",
    "south campus": "South Campus",
    "pitampura": "Pitampura",
    "rohini": "Rohini",
    "dwarka": "Dwarka",
    "noida": "Noida",
    "gurgaon": "Gurgaon",
    "faridabad": "Faridabad",
    "airport": "Indira Gandhi International Airport",
    "railway station": "New Delhi Railway Station",
    "metro": "Nearest Metro Station"
}

def geocode_place_with_fallback(place):
    coords = geocode_place(place)
    if not coords:
        coords = FAMOUS_PLACE_COORDS.get(place)
    return coords

def generate_response(intent_label: str, emotion_label: str, dst: dict, debug: dict, user_text: str, entities: List[Dict], summary: Optional[Dict] = None, traffic_data: Optional[Dict] = None, turn_by_turn_steps: Optional[List] = None, profile: dict = None) -> Dict[str, Any]:
    """
    Generates a contextual response and a potential action.
    Returns a dictionary: {"text": ..., "action": ...}
    """
    
    # --- [NEW] Try conversational model first ---
    conversational_response = generate_conversational_response(user_text)
    if conversational_response:
        debug["response_source"] = "conversational_model"
        print(f"🤖 Conversational model response: {conversational_response}")
        
        # Extract action from conversational response if possible
        proposed_action = None
        
        # Check if response contains navigation intent
        if any(word in conversational_response.lower() for word in ["route", "navigation", "chalein", "chalo", "set kar raha"]):
            # Try to extract location from entities or context
            location = next((e["value"] for e in entities if e["entity"].lower() == "location"), None)
            if location:
                proposed_action = {
                    "intent": "nav_start_navigation",
                    "entities": {"location": location}
                }
        
        return {"text": conversational_response, "action": proposed_action}
    
    # --- Fallback to existing logic ---
    debug["response_source"] = "rule_based_fallback"
    print(f"⚠️ Using rule-based fallback for response generation")
    
    core_response = ""
    proposed_action = None

    # --- Profanity/rudeness check ---
    if contains_rude(user_text):
        core_response = random.choice(RUDE_RESPONSES)
        return {"text": core_response, "action": None}

    # --- Personalization ---
    personality_prefix = ""
    playful_suffix = ""
    name_prompted = False
    is_greeting = any(greet in user_text.lower() for greet in ["hello", "hi", "namaste", "yo", "salaam", "aur bhai", "aur yaar", "bhai", "yaar"])
    if profile:
        if profile.get("name"):
            personality_prefix += f"{profile['name']}, "
        else:
            if not is_greeting and random.random() < 0.3:
                personality_prefix += "Naam bata do, dosti mein kya sharmana. "
                name_prompted = True
        # Only add 'Phir se {place}?' if not already present
        if profile.get("favorite_place") and profile.get("favorite_place") in user_text.lower():
            phir_se_phrase = f"Phir se {profile['favorite_place'].title()}? "
            if phir_se_phrase not in personality_prefix:
                personality_prefix += phir_se_phrase
        if profile.get("favorite_food") and profile.get("favorite_food") in user_text.lower():
            personality_prefix += f"{profile['favorite_food']} ka mood hai! "
        if profile.get("last_mood") == "happy":
            personality_prefix += "Mood sahi hai aaj. "
        elif profile.get("last_mood") == "sadness":
            personality_prefix += "Udaas mat ho. "
        elif profile.get("last_mood") == "anger":
            personality_prefix += "Gussa kam karo. "
        elif profile.get("last_mood") == "fear":
            personality_prefix += "Tension na le. "
    if not is_greeting and random.random() < 0.33:
        playful_suffix = " " + random.choice(SUBTLE_DELHI_BRO_PHRASES)

    if intent_label == "nav_start_navigation":
        # Use the location entity if available, else fallback to dst.get('destination')
        destination_name = None
        for e in entities:
            if e["entity"].lower() == "location":
                destination_name = e["value"]
                break
        if not destination_name:
            destination_name = dst.get('destination', 'teri manzil')
        core_response = f"{personality_prefix}Theek hai, {destination_name} ke liye route set kar raha hoon. Let's go!"
        if turn_by_turn_steps and len(turn_by_turn_steps) > 0:
            first_instruction = turn_by_turn_steps[0].get("instruction")
            if first_instruction:
                if "unavailable" in first_instruction.lower():
                    core_response = f"{personality_prefix}Yaar, {destination_name} ke liye route set kar raha hoon, lekin abhi routing service down hai. Google Maps ya Waze use kar le, ya thodi der baad try kar."
                else:
                    core_response += f" Sabse pehle, {first_instruction}."
    elif intent_label == "traffic_update":
        traffic_info = interpret_traffic(traffic_data)
        if traffic_info:
            level = traffic_info['level']
            emoji = traffic_info['emoji']
            if level == "closed":
                core_response = f"{personality_prefix}Oye, aage raasta band hai {emoji}. Bol, doosra route check karun?"
            else:
                level_hinglish = {"smooth": "ekdam clear", "moderate": "theek-thaak", "heavy": "bhayankar jaam"}
                core_response = f"{personality_prefix}Aage traffic {level_hinglish.get(level, 'ajeeb sa')} hai {emoji}. Sambhal ke."
        else:
            core_response = f"{personality_prefix}Ek second, traffic ka scene check kar raha hoon..."
    elif intent_label == "ETA_request":
        if summary and summary.get("duration_min") is not None and summary.get("distance_km") is not None:
            duration = summary["duration_min"]
            distance = summary["distance_km"]
            try:
                duration_str = f"{float(duration):.0f}"
            except (ValueError, TypeError):
                duration_str = str(duration)
            try:
                distance_str = f"{float(distance):.1f}"
            except (ValueError, TypeError):
                distance_str = str(distance)
            core_response = f"{personality_prefix}Bhai, abhi ke hisaab se toh lagbhag {duration_str} minute aur lagenge. Poora {distance_str} km ka raasta hai."
        else:
            core_response = f"{personality_prefix}Ruk ja, calculate kar raha hoon kab tak pahunchenge."
    
    # This block handles the food heuristic specifically
    elif intent_label == "find_food_spot":
        # Get the food type from the entities we extracted in the heuristic
        food_type = next((e["value"] for e in entities if e["entity"] == "food_type"), None)
        spot = FAMOUS_FOOD_SPOTS.get(food_type, None)
        # Call Gemini specifically to handle this creative task
        gemini_response = get_gemini_fallback_response(user_text, emotion_label, debug, food_type=food_type)
        core_response = gemini_response["text"]
        proposed_action = gemini_response["action"]
        # If Gemini or heuristic didn't provide a spot, use our mapping
        if not proposed_action and spot:
            core_response = f"{food_type.title()} ka mood hai! {spot} chalein?"
            proposed_action = {
                "intent": "nav_start_navigation",
                "entities": {"location": spot}
            }
        # --- Always mention the spot in the response if available ---
        spot_from_action = None
        if proposed_action and "entities" in proposed_action and "location" in proposed_action["entities"]:
            spot_from_action = proposed_action["entities"]["location"]
        if spot_from_action and spot_from_action.lower() not in core_response.lower():
            core_response = f"{core_response.strip()} {spot_from_action} chalein?"

    elif intent_label == "panic_signal":
        return {"text": "Distress signal mil gaya hai. Emergency services ko khabar kar raha hoon aur aapke contacts ko alert bhej raha hoon. Himmat rakhein.", "action": None}
    elif intent_label == "greeting":
        return {"text": random.choice(["Haanji bhai, batao! Scene on karein?", "Aur bhai! Kaise help kar sakta hoon?", "Yo! Kya sewa karun aapki?"]), "action": None}
    elif intent_label == "thanks":
        return {"text": random.choice(["Koi scene nahi hai, bro!", "Anytime, yaar.", "Chill maar."]), "action": None}
    elif intent_label == "help_request":
        core_response = f"{personality_prefix}Main navigation, traffic updates, aur ETA bata sakta hoon. Basically, Dilli ki sadko pe tera saathi. Bata, kya karna hai?"
    elif intent_label == "user_name_query":
        user_name = profile.get("name") if profile else None
        if user_name:
            core_response = f"Main Yuvi hoon, {user_name}! Batao, kya help karun?"
        else:
            core_response = "Main Yuvi hoon! Waise, tera naam kya hai?"

    elif intent_label == "user_name_introduction":
        name = None
        for e in entities:
            if e["entity"] == "name":
                name = e["value"]
        if name:
            core_response = f"Arre {name}, ab toh dosti pakki! Batao, kya help karun?"
        else:
            core_response = "Naam mil gaya, bro! Batao, kya scene hai?"

    elif intent_label == "recommendation_request":
        core_response = random.choice([
            "Bhai, Dilli mein toh options hi options hain! Chole bhature try kar, mood fresh ho jayega.",
            "Aaj kuch naya try kar, Dolma Aunty ke momos ya Sita Ram ke chole bhature mast hain!",
            "Bro, Big Chill ki pizza ya Karim's ke kebab, dono hi legendary hain.",
            "Mood bana hai toh Paranthe Wali Gali ka chakkar laga le!",
            "Rajinder Da Dhaba ka paneer tikka, must try!",
            "Dilli ki vibe chahiye toh Chandni Chowk ki jalebi le aa!"
        ])

    # --- Web Search for unsupported queries ---
    elif intent_label == "unsupported_query":
        # Check if this query should trigger web search
        if should_use_web_search(user_text):
            print(f"🔍 Triggering web search for: {user_text}")
            # Location-aware web search
            location_str = None
            if dst.get('lat') and dst.get('lon'):
                location_str = f"{dst['lat']},{dst['lon']}"
            search_results = search_web(user_text, location=location_str)
            if search_results:
                core_response = summarize_web_results(user_text, search_results)
                debug["web_search_used"] = True
                debug["search_results_count"] = len(search_results)
                print(f"✅ Web search completed with {len(search_results)} results")
            else:
                # Fallback to regular Gemini if web search fails
                gemini_response = get_gemini_fallback_response(user_text, emotion_label, debug)
                core_response = gemini_response["text"]
                proposed_action = gemini_response["action"]
                debug["web_search_failed"] = True
        else:
            # Use regular Gemini fallback for non-web-search queries
            gemini_response = get_gemini_fallback_response(user_text, emotion_label, debug)
            core_response = gemini_response["text"]
            proposed_action = gemini_response["action"]

    # --- Universal fallback for any unhandled intent ---
    else:
        # Check if this query should trigger web search
        if should_use_web_search(user_text):
            print(f"🔍 Triggering web search for unhandled intent: {user_text}")
            search_results = search_web(user_text)
            if search_results:
                core_response = summarize_web_results(user_text, search_results)
                debug["web_search_used"] = True
                debug["search_results_count"] = len(search_results)
                print(f"✅ Web search completed with {len(search_results)} results")
            else:
                # Fallback to regular Gemini if web search fails
                gemini_response = get_gemini_fallback_response(user_text, emotion_label, debug)
                core_response = gemini_response["text"]
                proposed_action = gemini_response["action"]
                debug["web_search_failed"] = True
        else:
            # Use regular Gemini fallback for non-web-search queries
            gemini_response = get_gemini_fallback_response(user_text, emotion_label, debug)
            core_response = gemini_response["text"]
            proposed_action = gemini_response["action"]

    return {"text": core_response, "action": proposed_action}

# --- Profile Extraction Helper ---
def extract_user_profile(history, last_emotion=None):
    """Extracts a rich user profile from the last 10 messages."""
    foods = []
    places = []
    intents = []
    greetings = []
    names = []
    lingo_words = ["bhai", "yaar", "scene", "bro", "dilli", "chalo", "theek hai", "mast", "sahi hai", "fir", "bindaas", "chill", "setting", "jugaad"]
    lingo_count = 0
    # Ignore common question words
    ignore_words = {"kya", "kaun", "bata", "batau", "batao", "hai", "hoon", "naam", "name", "mera", "my", "main", "is", "the", "who", "what", "am", "are", "you", "your"}
    # Patterns for name introduction
    patterns = [
        r"mera naam ([a-zA-Z]{2,20}) hai",
        r"my name is ([a-zA-Z]{2,20})",
        r"main ([a-zA-Z]{2,20}) hoon",
        r"naam ([a-zA-Z]{2,20}) hai",
        r"i am ([a-zA-Z]{2,20})",
        r"i'm ([a-zA-Z]{2,20})",
        r"this is ([a-zA-Z]{2,20})",
        r"me ([a-zA-Z]{2,20})"
    ]
    for msg in history:
        if msg["role"] == "user":
            text = msg["text"].lower()
            # Name extraction (robust)
            for pat in patterns:
                match = re.search(pat, text, re.IGNORECASE)
                if match:
                    name = match.group(1).capitalize()
                    if name.lower() not in ignore_words and name.isalpha() and 2 <= len(name) <= 20:
                        names.append(name)
            # Greeting extraction
            if any(greet in text for greet in ["hello", "hi", "namaste", "yo", "salaam", "aur bhai", "aur yaar", "bhai", "yaar"]):
                greetings.append(msg["text"])
            # Food detection (expanded)
            for food in ["chole bhature", "butter chicken", "momos", "parathe", "biryani", "kebab", "golgappe", "rajma chawal", "chaat", "paneer tikka", "dal makhani", "naan", "pizza", "burger", "pasta", "samosa", "jalebi", "lassi", "kulfi", "chaap", "rolls", "shawarma", "tandoori", "malai tikka", "dosa", "idli", "vada pav", "pav bhaji", "maggi", "chai", "coffee"]:
                if food in text:
                    foods.append(food)
            # Place detection (expanded)
            for place in ["cp", "connaught place", "rajiv chowk", "aiims", "india gate", "chandni chowk", "paharganj", "hauz khas", "saket", "karol bagh", "lajpat nagar", "sarojini nagar", "janpath", "greater kailash", "gk", "south ex", "north campus", "south campus", "delhi haat", "qutub minar", "red fort", "lotus temple", "akshardham", "dilli haat", "pitampura", "rohini", "dwarka", "noida", "gurgaon", "faridabad"]:
                if place in text:
                    places.append(place)
            # Lingo detection
            lingo_count += sum(1 for word in lingo_words if word in text)
        if msg["role"] == "yuvi":
            # Try to extract intent from Yuvi's response (if present)
            if "intent:" in msg["text"].lower():
                intents.append(msg["text"].lower().split("intent:")[-1].strip())
    profile = {}
    if foods:
        profile["favorite_food"] = collections.Counter(foods).most_common(1)[0][0]
    if places:
        profile["favorite_place"] = collections.Counter(places).most_common(1)[0][0]
    if intents:
        profile["common_intent"] = collections.Counter(intents).most_common(1)[0][0]
    if greetings:
        profile["favorite_greeting"] = greetings[0]
    if names:
        profile["name"] = names[-1]
    if lingo_count > 2:
        profile["delhi_lingo"] = True
    else:
        profile["delhi_lingo"] = False
    if last_emotion:
        profile["last_mood"] = last_emotion.get("label")
    return profile

# --- TomTom Routing Helper ---
def get_tomtom_route(start_coords, end_coords):
    try:
        url = f"https://api.tomtom.com/routing/1/calculateRoute/{start_coords[0]},{start_coords[1]}:{end_coords[0]},{end_coords[1]}/json?key={TOMTOM_API_KEY}&instructionsType=text&computeBestOrder=false&routeType=fastest&traffic=true"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        if "routes" in data and data["routes"]:
            route = data["routes"][0]
            summary = route["summary"]
            instructions = []
            for leg in route.get("legs", []):
                for step in leg.get("points", []):
                    instr = step.get("instruction", "")
                    if instr and instr.strip():
                        instructions.append({
                            "instruction": instr,
                            "distance_meters": step.get("lengthInMeters", 0),
                            "duration_seconds": step.get("travelTimeInSeconds", 0),
                            "name": step.get("roadNumbers", [""])[0] if step.get("roadNumbers") else "",
                            "type": step.get("maneuver", ""),
                            "way_points": []
                        })
            # Fallback if no valid instructions
            if not instructions:
                instructions = [{"instruction": "Route found, but no turn-by-turn steps available.", "distance_meters": summary.get("lengthInMeters", 0), "duration_seconds": summary.get("travelTimeInSeconds", 0), "name": "", "type": "", "way_points": []}]
            return {
                "distance_km": round(summary["lengthInMeters"] / 1000, 2),
                "duration_min": round(summary["travelTimeInSeconds"] / 60, 2),
                "turn_by_turn_steps": instructions,
                "geometry": None
            }
    except Exception as e:
        print(f"TomTom routing error: {e}")
        return None
    return None

# --- Name Extraction Helper ---
def extract_name_from_text(text):
    text = text.lower()
    # Ignore common question words
    ignore_words = {"kya", "kaun", "bata", "batau", "batao", "hai", "hoon", "naam", "name", "mera", "my", "main", "is", "the", "who", "what", "am", "are", "you", "your"}
    # Patterns for name introduction
    patterns = [
        r"mera naam ([a-zA-Z]{2,20}) hai",
        r"my name is ([a-zA-Z]{2,20})",
        r"main ([a-zA-Z]{2,20}) hoon",
        r"naam ([a-zA-Z]{2,20}) hai",
        r"i am ([a-zA-Z]{2,20})",
        r"i'm ([a-zA-Z]{2,20})",
        r"this is ([a-zA-Z]{2,20})",
        r"me ([a-zA-Z]{2,20})"
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            name = match.group(1).capitalize()
            # Filter out ignore words and non-names
            if name.lower() not in ignore_words and name.isalpha() and 2 <= len(name) <= 20:
                return name
    return None

# --- Contextual Suggestion Helper ---
def get_contextual_suggestion(history):
    # Look for last food or place in history
    for msg in reversed(history):
        if msg["role"] == "user":
            text = msg["text"].lower()
            for food in [
                "chole bhature", "butter chicken", "momos", "parathe", "biryani", "kebab", "golgappe", "rajma chawal", "chaat", "paneer tikka", "dal makhani", "naan", "pizza", "burger", "pasta", "samosa", "jalebi", "lassi", "kulfi", "chaap", "rolls", "shawarma", "tandoori", "malai tikka", "dosa", "idli", "vada pav", "pav bhaji", "maggi", "chai", "coffee"
            ]:
                if food in text:
                    spot = FAMOUS_FOOD_SPOTS.get(food)
                    if not spot and food == "pizza":
                        spot = "Big Chill Cafe, Khan Market"  # Fallback for pizza
                    elif not spot:
                        spot = "Haldiram's, Connaught Place"  # Generic fallback
                    return {"type": "food", "value": food, "spot": spot}
            for place in [
                "cp", "connaught place", "rajiv chowk", "aiims", "india gate", "chandni chowk", "paharganj", "hauz khas", "saket", "karol bagh", "lajpat nagar", "sarojini nagar", "janpath", "greater kailash", "gk", "south ex", "north campus", "south campus", "delhi haat", "qutub minar", "red fort", "lotus temple", "akshardham", "dilli haat", "pitampura", "rohini", "dwarka", "noida", "gurgaon", "faridabad"
            ]:
                if place in text:
                    return {"type": "place", "value": place}
    return None

# --- Helper: Get last Yuvi food/place suggestion ---
def get_last_yuvi_suggestion(history):
    for msg in reversed(history):
        if msg["role"] == "yuvi" and "text" in msg:
            text = msg["text"].lower()
            # Look for a famous food spot or place in the response
            for food, spot in FAMOUS_FOOD_SPOTS.items():
                if food in text or spot.lower() in text:
                    return {"type": "food", "value": food, "spot": spot}
            for place in [
                "cp", "connaught place", "rajiv chowk", "aiims", "india gate", "chandni chowk", "paharganj", "hauz khas", "saket", "karol bagh", "lajpat nagar", "sarojini nagar", "janpath", "greater kailash", "gk", "south ex", "north campus", "south campus", "delhi haat", "qutub minar", "red fort", "lotus temple", "akshardham", "dilli haat", "pitampura", "rohini", "dwarka", "noida", "gurgaon", "faridabad"
            ]:
                if place in text:
                    return {"type": "place", "value": place}
    return None

class Entity(BaseModel):
    entity: str
    value: str

class NLUResponse(BaseModel):
    text: str
    intent: Dict[str, Any]
    emotion: Dict[str, Any]
    entities: List[Entity]
    debug: Dict[str, Any]
    route_summary: Optional[Dict[str, Any]] = None
    turn_by_turn_steps: Optional[List[Dict[str, Any]]] = None
    route_geometry: Optional[Dict[str, Any]] = None
    map_html: Optional[str] = None
    response_text: str
    tts_audio_base64: Optional[str] = None

# --- Context Extraction Helper ---
def get_last_relevant_intent(history, exclude_intents=None):
    """Find the last non-affirmation, non-unsupported intent and its entities from Yuvi's history."""
    if exclude_intents is None:
        exclude_intents = {"unsupported_query", "affirmation_without_context", "greeting", "thanks"}
    for msg in reversed(history):
        if msg["role"] == "yuvi" and "intent" in msg:
            intent = msg["intent"]
            if intent and intent.get("label") not in exclude_intents:
                entities = msg.get("entities", [])
                return intent, entities
    return None, []

# ------------------- Main NLU Endpoint -------------------
@app.post("/nlu", response_model=NLUResponse)
def nlu_pipeline(input_data: TextInput):
    import time
    start_time = time.time()
    
    user_id = input_data.user_id
    text = input_data.text

    # Always get state at the start so it is available for all logic
    state = get_state(user_id)

    # --- Gemma-2 Hinglish Text Normalization (SDK version) ---
    original_text = text
    try:
        normalized_text = normalize_transcript_with_gemma(text)
    except Exception as e:
        print(f"[Gemma Normalizer Error] {e}. Using original text.")
        normalized_text = text
    # Use normalized_text for all downstream processing
    text = normalized_text

    # --- Remove wake word from the beginning of the input, if present ---
    WAKE_WORDS = [
        "hey yuvii", "okay yuvii", "hey yuvi", "hello yuvii", "hello yuvi",
        "yuvii", "yuvi"
    ]
    text_lower = text.lower().strip()
    for wake in WAKE_WORDS:
        if text_lower.startswith(wake):
            text = text[len(wake):].lstrip(' ,.!?')
            break

    # --- [NEW] Smart Text Preprocessing (disabled for now) ---
    preprocessing_info = None
    
    if SMART_PREPROCESSING_ENABLED and smart_preprocessor:
        preprocessing_result = smart_preprocessor.preprocess_text(text)
        if preprocessing_result['was_corrected']:
            text = preprocessing_result['corrected_text']
            preprocessing_info = {
                "original_text": preprocessing_result['original_text'],
                "corrected_text": preprocessing_result['corrected_text'],
                "suggestions": preprocessing_result['suggestions'],
                "correction_applied": True
            }
            print(f"🔄 Text preprocessing: '{original_text}' → '{text}'")
            if preprocessing_result['suggestions']:
                print(f"   💡 Suggestions: {preprocessing_result['suggestions']}")
    
    # --- FAST PATH: Check for simple affirmations first ---
    if is_affirmation(text):
        # Check for a recent proposed_action (navigation or food) in the state
        proposed_action = state.get('proposed_action')
        if proposed_action and proposed_action.get('intent') in ["nav_start_navigation", "find_food_spot"]:
            response_text = "Theek hai, kar raha hoon!"
            # Confirm and trigger the proposed action
            intent = {
                "label": proposed_action["intent"],
                "confidence": 1.0,
                "source": "affirmation_followup"
            }
            final_entities = [{"entity": k, "value": v} for k, v in proposed_action.get("entities", {}).items()]
            # Do NOT remove the proposed_action here; keep it for further affirmations
            # Only remove it after a new action is proposed elsewhere in the pipeline
            # Continue to downstream processing with this intent/entities
        else:
            # Quick response for affirmations without heavy processing
            response_text = "Theek hai, kar raha hoon!"
            return {
                "text": input_data.text,
                "intent": {"label": "affirmation", "confidence": 1.0, "source": "fast_path"},
                "emotion": {"label": "neutral", "confidence": 1.0},
                "entities": [],
                "debug": {"fast_path": True, "affirmation": True},
                "route_summary": None,
                "turn_by_turn_steps": None,
                "route_geometry": None,
                "map_html": None,
                "response_text": response_text,
                "tts_audio_base64": None
            }
    
    # --- FAST PATH: Check for distress signals ---
    distress = detect_distress(text)
    if distress:
        # Quick response for distress without heavy processing
        response_text = "Distress signal mil gaya hai. Emergency services ko khabar kar raha hoon aur aapke contacts ko alert bhej raha hoon. Himmat rakhein."
        return {
            "text": input_data.text,
            "intent": distress,
            "emotion": {"label": "fear", "confidence": 1.0},
            "entities": [],
            "debug": {"fast_path": True, "distress": True},
            "route_summary": None,
            "turn_by_turn_steps": None,
            "route_geometry": None,
            "map_html": None,
            "response_text": response_text,
            "tts_audio_base64": None
        }
    
    # 1. Retrieve Dialogue State and Detect Language
    state = get_state(user_id)
    # Prune history to last 30 minutes
    if 'history' in state and isinstance(state['history'], list):
        state['history'] = prune_history(state['history'])
    
    # --- OPTIMIZED: Skip language detection for common patterns ---
    language = "hi-en"  # Default for most Delhi driver inputs
    if any(word in text_lower for word in ["hello", "hi", "thanks", "thank you", "ok", "okay", "yes", "no"]):
        language = "en"
    
    # Debug: Print what we're processing
    print(f"🎯 Processing text: '{text}'")
    print(f"🌐 Detected language: {language}")

    debug = {
        "original_text": original_text,
        "normalized_text": normalized_text,
        "language": language,
        "language_source": "optimized",
        "intent_model_raw": None,
        "heuristic_applied": False,
        "urgent_flag": False,
        "navigation_used": False,
        "fallback_source": "None",
        "fallback_forced_by_threshold": False,
        "follow_up_triggered": False,
        "context_used": False,
        "escalation": None,
        "preprocessing_info": preprocessing_info
    }

    # --- ROBUST LOGIC REFACTOR START ---
    # This section is refactored for clarity and correctness, preserving all features.

    # Step 2.1: Initialize variables that will be determined by the logic flow
    intent = None
    final_entities = []
    emotion = {}

    # --- Vague Navigation Handling (ALWAYS CHECK FIRST) ---
    vague_nav_phrases = [
        "le chal mujhe", "haan le chalo", "le chalo", "le chal", "chalein", "chalna hai", "le jao", "take me", "let's go", "chalo fir", "chalo yaar"
    ]
    contextual_suggestion_used = False

    # --- IMPROVED: Extract full destination after navigation trigger phrases ---
    navigation_triggers = [
        "take me to ", "drive to ", "navigate to ", "go to ", "route to ", "le chalo ", "le jao ", "chalein ", "chalna hai ", "drop me at ", "drop at ", "jaana hai ", "chalna hai "
    ]
    extracted_destination = None
    lowered_text = text.lower()
    for trigger in navigation_triggers:
        idx = lowered_text.find(trigger)
        if idx != -1:
            # Extract everything after the trigger as the destination
            after = text[idx + len(trigger):].strip()
            # Remove trailing punctuation
            after = after.rstrip('.!?')
            # If there's a sentence break, only take up to the first one
            for sep in ['.', '?', '!', ',']:
                if sep in after:
                    after = after.split(sep)[0].strip()
            if after:
                extracted_destination = after
                break
    if extracted_destination:
        intent = {"label": "nav_start_navigation", "confidence": 1.0, "source": "pattern_destination_extraction"}
        final_entities = [{"entity": "location", "value": extracted_destination}]
        contextual_suggestion_used = True
    elif any(phrase in text.lower() for phrase in vague_nav_phrases):
        # Use contextual suggestion from history/profile
        suggestion = get_contextual_suggestion(state.get("history", []))
        profile = extract_user_profile(state.get("history", []))
        # --- FIX: If previous turn had a proposed_action with a location, use that ---
        proposed_action_from_last_turn = state.get('proposed_action')
        if proposed_action_from_last_turn and "entities" in proposed_action_from_last_turn and "location" in proposed_action_from_last_turn["entities"]:
            intent = {"label": "nav_start_navigation", "confidence": 1.0, "source": "proposed_action_followup"}
            final_entities = [{"entity": "location", "value": proposed_action_from_last_turn["entities"]["location"]}]
            contextual_suggestion_used = True
        elif suggestion:
            if suggestion["type"] == "food":
                intent = {"label": "nav_start_navigation", "confidence": 1.0, "source": "contextual_suggestion"}
                final_entities = [{"entity": "location", "value": suggestion["spot"]}]
                contextual_suggestion_used = True
            elif suggestion["type"] == "place":
                intent = {"label": "nav_start_navigation", "confidence": 1.0, "source": "contextual_suggestion"}
                final_entities = [{"entity": "location", "value": suggestion["value"].title()}]
                contextual_suggestion_used = True
        else:
            response_text = f"{profile.get('name', 'Bhai')}, kahan le chalun? Batao!"
            return {
                "text": response_text,
                "intent": {"label": "unsupported_query", "confidence": 1.0, "source": "vague_navigation_no_context"},
                "emotion": emotion,
                "entities": [],
                "debug": debug,
                "route_summary": None,
                "turn_by_turn_steps": None,
                "route_geometry": None,
                "map_html": None,
                "response_text": response_text,
                "tts_audio_base64": None
            }
    # If contextual suggestion was used, skip intent/entity detection below
    if not contextual_suggestion_used:
        # Step 2.2: Determine Intent and Entities based on a clear priority order
        # (rest of the logic continues as before)
        # PRIORITY 1: Handle follow-up actions from the previous turn
        proposed_action_from_last_turn = state.get('proposed_action')
        if proposed_action_from_last_turn and is_affirmation(text):
            debug["follow_up_triggered"] = True
            intent = {
                "label": proposed_action_from_last_turn["intent"],
                "confidence": 1.0,
                "source": "follow_up_confirmation"
            }
            final_entities = [{"entity": k, "value": v} for k, v in proposed_action_from_last_turn.get("entities", {}).items()]
            state.pop('proposed_action', None)  # Clear the action now that it's confirmed
            # Skip contextual suggestion and proceed to navigation logic
        else:
            # PRIORITY 2: Check for high-confidence heuristics (like your food list)
            # BUT first check if it's a simple affirmation that shouldn't trigger heuristics
            if is_affirmation(text):
                # If it's an affirmation but no proposed action, treat as unsupported
                intent = {
                    "label": "unsupported_query",
                    "confidence": 0.95,
                    "source": "affirmation_without_context"
                }
                debug["heuristic_applied"] = True
                # Clear any old proposals when affirmation has no context
                if proposed_action_from_last_turn:
                    state.pop('proposed_action', None)
            else:
                # Clear any old proposals for non-affirmation inputs
                if proposed_action_from_last_turn:
                    state.pop('proposed_action', None)
                
                heuristic = heuristic_intent_match(text)
                if heuristic:
                    intent = heuristic
                    debug["heuristic_applied"] = True
                    print(f"🎯 Heuristic matched: {intent['label']} (confidence: {intent['confidence']}, source: {intent['source']})")
                    # Extract entities directly from the heuristic if they exist
                    if "entities" in heuristic:
                        final_entities = [{"entity": k, "value": v} for k, v in heuristic.get("entities", {}).items()]
                    else:
                        # For navigation heuristics, still try to extract location entities
                        if intent["label"] in ["nav_start_navigation", "nav_reroute", "ETA_request", "traffic_update"]:
                            # Try to extract location from text using gazetteer
                            normalized_text = text.lower()
                            gazetteer_entities = []
                            raw_entities = []  # Initialize for heuristic path
                            
                            # --- [NEW] Smart preprocessing for location extraction ---
                            if SMART_PREPROCESSING_ENABLED and smart_preprocessor:
                                # Try to extract location using smart preprocessor
                                preprocess_result = smart_preprocessor.preprocess_text(text)
                                if preprocess_result['was_corrected']:
                                    corrected_text = preprocess_result['corrected_text']
                                    # Check if the corrected text contains a known location
                                    for place_name in FAMOUS_PLACE_COORDS.keys():
                                        if place_name.lower() in corrected_text:
                                            gazetteer_entities.append({"entity": "location", "value": place_name})
                                            print(f"🔍 Smart preprocessor found location: '{text}' → '{place_name}'")
                                            break
                            
                            # Fallback to manual gazetteer lookup
                            for abbr, full in {
                                "cp": "Connaught Place",
                                "rajiv chowk": "Connaught Place",
                                "aiims": "All India Institute of Medical Sciences"
                            }.items():
                                if abbr in normalized_text:
                                    gazetteer_entities.append({"entity": "location", "value": full})
                            for key, val in KNOWN_LOCATIONS.items():
                                if re.search(rf"\b{re.escape(key)}\b", text, re.IGNORECASE):
                                    gazetteer_entities.append({"entity": "location", "value": val})
                            
                            # Also check FAMOUS_PLACE_COORDS for direct matches
                            for place_name in FAMOUS_PLACE_COORDS.keys():
                                if place_name.lower() in normalized_text:
                                    gazetteer_entities.append({"entity": "location", "value": place_name})
                                    break
                            
                            # --- [NEW] Direct spelling variation handling ---
                            # Handle specific cases like "kutub minar" → "Qutub Minar"
                            if "kutub minar" in normalized_text:
                                gazetteer_entities.append({"entity": "location", "value": "Qutub Minar"})
                                print(f"🔍 Direct mapping: 'kutub minar' → 'Qutub Minar'")
                            
                            # Combine and filter entities
                            all_entities = raw_entities + gazetteer_entities
                            seen_entities = set()
                            # Only keep entities that are in a known set (location, food_type, place, etc.)
                            valid_entity_types = {"location", "food_type", "place", "restaurant"}
                            for entity in all_entities:
                                entity_type = entity["entity"].lower()
                                raw_value = entity["value"]
                                norm_value = normalize_phrase(raw_value)
                                entity_key = (entity_type, norm_value)
                                if entity_key in seen_entities:
                                    continue
                                is_valid = True
                                if entity_type == 'location':
                                    if norm_value in NORMALIZED_BAD_PHRASES:
                                        is_valid = False
                                if entity_type not in valid_entity_types:
                                    is_valid = False
                                if is_valid:
                                    final_entities.append({"entity": entity["entity"], "value": raw_value})
                                    seen_entities.add(entity_key)
                else:
                    # PRIORITY 3: Fallback to the main Transformer model (ONLY if heuristic fails)
                    with torch.no_grad():
                        intent_result = get_intent_pipeline()(text)[0]
                    debug["intent_model_raw"] = intent_result.copy()
                    print(f"🎯 Model result: {intent_result['label']} (confidence: {intent_result['score']:.3f})")
                    
                    # DEBUG: Show top 3 predictions
                    with torch.no_grad():
                        all_predictions = get_intent_pipeline()(text)
                    print(f"🔍 Top 3 intent predictions:")
                    for i, pred in enumerate(all_predictions[:3]):
                        print(f"   {i+1}. {pred['label']}: {pred['score']:.3f}")

                    # Apply Delhi driver intent mapping
                    intent_result = map_delhi_driver_intent(text, intent_result)

                    # If confidence is too low, treat it as an unsupported query for Gemini to handle
                    # Handle both 'score' and 'confidence' keys for compatibility
                    confidence = intent_result.get('score', intent_result.get('confidence', 0.0))
                    if confidence < CONFIDENCE_THRESHOLD:
                        intent_result['label'] = 'unsupported_query'
                        debug["fallback_forced_by_threshold"] = True
                        print(f"⚠️ Low confidence ({confidence:.3f}), forcing unsupported_query")

                    intent = {
                        "label": intent_result["label"],
                        "confidence": round(intent_result.get("score", intent_result.get("confidence", 0.0)), 3),
                        "source": intent_result.get("source", "model")
                    }

                    # Run Named Entity Recognition (NER) only for the ML model path
                    with torch.no_grad():
                        ner_results = get_ner_pipeline()(text)
                    raw_entities = [{"entity": e["entity_group"], "value": e["word"]} for e in ner_results if e["entity_group"] != "O"]
                    # Clear CUDA cache after NER inference
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    gazetteer_entities = []
                    normalized_text = text.lower()
                    
                    # --- [NEW] Smart preprocessing for location extraction ---
                    if SMART_PREPROCESSING_ENABLED and smart_preprocessor:
                        # Try to extract location using smart preprocessor
                        preprocess_result = smart_preprocessor.preprocess_text(text)
                        if preprocess_result['was_corrected']:
                            corrected_text = preprocess_result['corrected_text']
                            # Check if the corrected text contains a known location
                            for place_name in FAMOUS_PLACE_COORDS.keys():
                                if place_name.lower() in corrected_text:
                                    gazetteer_entities.append({"entity": "location", "value": place_name})
                                    print(f"🔍 Smart preprocessor found location: '{text}' → '{place_name}'")
                                    break
                    
                    # Fallback to manual gazetteer lookup
                    for abbr, full in {
                        "cp": "Connaught Place",
                        "rajiv chowk": "Connaught Place",
                        "aiims": "All India Institute of Medical Sciences"
                    }.items():
                        if abbr in normalized_text:
                            gazetteer_entities.append({"entity": "location", "value": full})
                    for key, val in KNOWN_LOCATIONS.items():
                        if re.search(rf"\b{re.escape(key)}\b", text, re.IGNORECASE):
                            gazetteer_entities.append({"entity": "location", "value": val})
                    
                    # Also check FAMOUS_PLACE_COORDS for direct matches
                    for place_name in FAMOUS_PLACE_COORDS.keys():
                        if place_name.lower() in normalized_text:
                            gazetteer_entities.append({"entity": "location", "value": place_name})
                            break
                    
                    # --- [NEW] Direct spelling variation handling ---
                    # Handle specific cases like "kutub minar" → "Qutub Minar"
                    if "kutub minar" in normalized_text:
                        gazetteer_entities.append({"entity": "location", "value": "Qutub Minar"})
                        print(f"🔍 Direct mapping: 'kutub minar' → 'Qutub Minar'")
                    
                    # Combine and filter entities
                    all_entities = raw_entities + gazetteer_entities
                    seen_entities = set()
                    # Only keep entities that are in a known set (location, food_type, place, etc.)
                    valid_entity_types = {"location", "food_type", "place", "restaurant"}
                    for entity in all_entities:
                        entity_type = entity["entity"].lower()
                        raw_value = entity["value"]
                        norm_value = normalize_phrase(raw_value)
                        entity_key = (entity_type, norm_value)
                        if entity_key in seen_entities:
                            continue
                        is_valid = True
                        if entity_type == 'location':
                            if norm_value in NORMALIZED_BAD_PHRASES:
                                is_valid = False
                        if entity_type not in valid_entity_types:
                            is_valid = False
                        if is_valid:
                            final_entities.append({"entity": entity["entity"], "value": raw_value})
                            seen_entities.add(entity_key)

    # Step 2.3: Detect emotion for any non-distress case (OPTIMIZED)
    if not emotion: # Emotion is only pre-filled for distress signals
        # --- OPTIMIZED: Skip emotion detection for simple cases ---
        if intent["label"] in ["affirmation", "greeting", "thanks", "help_request"]:
            emotion = {"label": "neutral", "confidence": 1.0}
        else:
            with torch.no_grad():
                emotion_result = get_emotion_pipeline()(text)[0]
            emotion = {
                "label": emotion_result["label"],
                "confidence": round(emotion_result["score"], 3)
            }

    # --- ROBUST LOGIC REFACTOR END ---

    # 4. Downstream Processing
    route_summary = None
    turn_by_turn_steps = None
    route_geometry = None
    traffic_data = None
    map_html = None

    # --- Name Extraction and Persistence ---
    extracted_name = extract_name_from_text(text)
    if extracted_name:
        state.setdefault("profile", {})["name"] = extracted_name
    # --- Conversation History Update (append user message) ---
    if "history" not in state or not isinstance(state["history"], list):
        state["history"] = []
    state["history"].append({
        "role": "user",
        "text": text,
        "timestamp": time.time()
    })

    # --- User Profile Extraction ---
    profile = extract_user_profile(state["history"], last_emotion=emotion)
    # --- Ensure name is persistent ---
    if state.get("profile", {}).get("name") and not profile.get("name"):
        profile["name"] = state["profile"]["name"]
    state["profile"] = profile

    if intent["label"] in ["nav_start_navigation", "nav_reroute", "ETA_request", "traffic_update"]:
        destination = next((e["value"] for e in final_entities if e["entity"].lower() == "location"), None)
        # --- NEW: Use last known destination from last 30 min if no location entity ---
        if not destination:
            # Search history for last navigation or food spot
            for msg in reversed(state.get("history", [])):
                if msg["role"] == "yuvi" and "proposed_action" in msg:
                    pa = msg["proposed_action"]
                    if pa and "entities" in pa and "location" in pa["entities"]:
                        destination = pa["entities"]["location"]
                        break
                if msg["role"] == "yuvi" and "text" in msg and ("route set kar raha hoon" in msg["text"] or "chalein" in msg["text"]):
                    # Try to extract location from text (very basic fallback)
                    m = re.search(r"\b([A-Za-z ]+, [A-Za-z ]+)\b", msg["text"])
                    if m:
                        destination = m.group(1)
                        break
            # If still not found, use default
            if not destination:
                destination = "Connaught Place, Delhi"
        if not destination and state.get("destination"):
            destination = state["destination"]

        if destination:
            dest_coords = geocode_place_with_fallback(destination)
            if dest_coords:
                # Always require real user location; if not provided, return error
                if input_data.lat is not None and input_data.lon is not None:
                    start_coords = (input_data.lat, input_data.lon)
                else:
                    return {
                        "text": "Location not provided. Please enable location access or provide your current coordinates.",
                        "intent": intent,
                        "emotion": emotion,
                        "entities": final_entities,
                        "debug": debug,
                        "route_summary": None,
                        "turn_by_turn_steps": None,
                        "route_geometry": None,
                        "map_html": None,
                        "response_text": "Location not provided. Please enable location access or provide your current coordinates.",
                        "tts_audio_base64": None
                    }
                # --- ORS Retry Logic ---
                route_details = None
                for attempt in range(3):
                    route_details = get_route_details(start_coords, dest_coords)
                    if route_details:
                        break
                # --- TomTom fallback ---
                if not route_details:
                    route_details = get_tomtom_route(start_coords, dest_coords)
                    if route_details:
                        debug["navigation_service_used"] = "tomtom"
                if not route_details:
                    debug["navigation_service_unavailable"] = True
                    route_summary = {
                        "distance_km": "Unknown",
                        "duration_min": "Unknown",
                        "note": "Routing service temporarily unavailable. Try again later or use Google Maps."
                    }
                    turn_by_turn_steps = [{"instruction": "Route calculation service is currently unavailable. Please use Google Maps or your preferred navigation app."}]
                    state["destination"] = destination
                else:
                    route_summary = {
                        "distance_km": route_details["distance_km"],
                        "duration_min": route_details["duration_min"]
                    }
                    turn_by_turn_steps = route_details["turn_by_turn_steps"]
                    route_geometry = route_details.get("geometry")
                    debug["navigation_used"] = True
                    state["destination"] = destination

                    if route_geometry:
                        try:
                            if isinstance(route_geometry, str):
                                decoded_coords = polyline.decode(route_geometry)
                                route_geometry = {
                                    "type": "LineString",
                                    "coordinates": [[lon, lat] for lat, lon in decoded_coords]
                                }
                                map_html = generate_static_map_html(route_geometry, start_coords, dest_coords)
                            elif isinstance(route_geometry, dict) and route_geometry.get("coordinates"):
                                map_html = generate_static_map_html(route_geometry, start_coords, dest_coords)
                            
                            # Automatically save map HTML to file
                            if map_html:
                                try:
                                    with open("map_preview.html", "w", encoding="utf-8") as f:
                                        f.write(map_html)
                                    print(f"✅ Map HTML automatically saved to map_preview.html for route to {destination}")
                                except Exception as e:
                                    print(f"⚠️ Could not save map HTML: {e}")
                                    
                        except Exception as e:
                            print("Error while rendering map:", e)

                if intent["label"] == "traffic_update":
                    traffic_data = get_tomtom_traffic(*dest_coords)

    # 5. Generate Response
    response_obj = generate_response(
        intent_label=intent["label"],
        emotion_label=emotion["label"],
        dst=state,
        debug=debug,
        user_text=text,
        entities=final_entities,
        summary=route_summary,
        traffic_data=traffic_data,
        turn_by_turn_steps=turn_by_turn_steps,
        profile=profile
    )

    response_text = response_obj["text"]
    new_proposed_action = response_obj["action"]

    # --- [NEW] Add preprocessing suggestions to response ---
    if preprocessing_info and preprocessing_info.get("suggestions"):
        suggestion_text = " ".join(preprocessing_info["suggestions"])
        response_text = f"{suggestion_text} {response_text}"
        print(f"💡 Added suggestion to response: {suggestion_text}")

    # --- Conversation History Update (append Yuvi's response) ---
    state["history"].append({
        "role": "yuvi",
        "text": response_text,
        "timestamp": time.time()
    })

    # 6. Dialogue State Update
    state["current_intent"] = intent["label"]
    if new_proposed_action:
        state["proposed_action"] = new_proposed_action
        debug["proposed_action_to_save"] = new_proposed_action
    # Remove proposed_action if it was just confirmed by an affirmation
    elif intent["source"] == "affirmation_followup" and "proposed_action" in state:
        state.pop("proposed_action", None)
    save_state(state)

    # 7. Log the NLU event for analytics
    analytics_logger.log_nlu_event(
        user_id=user_id,
        text=text,
        intent=intent,
        emotion=emotion,
        entities=final_entities,
        debug_info=debug,
        response_text=response_text
    )

    # --- Automatic TTS generation for response_text (OPTIMIZED) ---
    tts_audio_base64 = None
    # --- OPTIMIZED: Skip TTS for simple responses to reduce latency ---
    if len(response_text) > 50 and intent["label"] not in ["affirmation", "greeting", "thanks"]:
        try:
            tts = gTTS(response_text, lang='en')
            from io import BytesIO
            audio_fp = BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            tts_audio_base64 = base64.b64encode(audio_fp.read()).decode('utf-8')
        except Exception as e:
            print(f"TTS generation failed: {e}")
            tts_audio_base64 = None

    # 8. Return Response
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"⚡ NLU Processing completed in {processing_time:.2f} seconds")
    
    return {
        "text": input_data.text,  # Return original text for display
        "intent": intent,
        "emotion": emotion,
        "entities": final_entities,
        "debug": debug,
        "route_summary": route_summary,
        "turn_by_turn_steps": turn_by_turn_steps,
        "route_geometry": route_geometry,
        "map_html": map_html,
        "response_text": response_text,
        "tts_audio_base64": tts_audio_base64
    }

    # --- Ensure Yuvi never returns a blank response ---
    if not response_text or not response_text.strip():
        response_text = "Bhai, system thoda busy hai, par main hoon na! Kuch aur poochh le ya try kar le baad mein."

@app.get("/dst/{user_id}")
def get_dst(user_id: str):
    """Retrieve the dialogue state for a given user."""
    return get_state(user_id)

@app.post("/tts")
def tts_endpoint(text: str, lang: str = 'en'):
    """
    Generate speech from text using gTTS and return as MP3 audio.
    Args:
        text (str): The text to synthesize.
        lang (str): Language code ('en' for Hinglish, 'hi' for Hindi). Default is 'en'.
    Returns:
        MP3 audio as a streaming response.
    """
    tts = gTTS(text, lang=lang)
    from io import BytesIO
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return Response(content=audio_fp.read(), media_type="audio/mpeg")

@app.get("/")
def root():
    return FileResponse("yuvi_tts_frontend.html")

def strip_wake_word(text: str) -> str:
    """Removes the wake word from the beginning of the input, if present."""
    WAKE_WORDS = [
        "hey yuvii", "okay yuvii", "hey yuvi", "hello yuvii", "hello yuvi",
        "yuvii", "yuvi"
    ]
    text_lower = text.lower().strip()
    for wake in WAKE_WORDS:
        if text_lower.startswith(wake):
            return text[len(wake):].lstrip(' ,.!?')
    return text

@app.post("/stt")
async def stt_endpoint(file: UploadFile = File(...)):
    try:
        print(f"🎤 Received audio file: {file.filename}, size: {file.size} bytes")
        
        # Save uploaded audio to a temp file (unknown format)
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
            print(f"📁 Saved audio to: {tmp_path}")

        # Convert to WAV using ffmpeg
        wav_path = tmp_path + ".wav"
        print(f"🔄 Converting to WAV: {wav_path}")
        
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path
            ], check=True, capture_output=True)
            print(f"✅ Audio conversion successful")
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg conversion failed: {e}")
            print(f"FFmpeg stderr: {e.stderr.decode()}")
            return JSONResponse({"error": "Audio conversion failed"}, status_code=500)

        # Import and use the transcribe function from stt.py
        from stt import transcribe
        import scipy.io.wavfile as wav
        
        try:
            sample_rate, audio_np = wav.read(wav_path)
            print(f"📊 Audio loaded: {len(audio_np)} samples, {sample_rate} Hz")
            
            transcript = transcribe(audio_np)
            if transcript:
                print(f"✅ Transcription successful: '{transcript}'")
                return JSONResponse({"transcript": transcript})
            else:
                print(f"⚠️ No transcript generated")
                return JSONResponse({"transcript": ""})
                
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return JSONResponse({"error": f"Transcription failed: {str(e)}"}, status_code=500)
            
    except Exception as e:
        print(f"❌ STT endpoint error: {e}")
        return JSONResponse({"error": f"STT processing failed: {str(e)}"}, status_code=500)
    finally:
        # Clean up temp files
        try:
            if 'tmp_path' in locals():
                os.remove(tmp_path)
            if 'wav_path' in locals():
                os.remove(wav_path)
        except:
            pass

# --- Hinglish to English Translation ---
HINGLISH_TO_ENGLISH = {
    # Greetings and common phrases
    "aaj ka kya scene hai": "what's the scene today",
    "kya scene hai": "what's the scene",
    "kya haal hai": "how are you",
    "kaise ho": "how are you",
    "kaisa chal raha hai": "how is it going",
    "sab badhiya": "everything is good",
    "kya haal chaal": "how are things",
    "kya chal raha": "what's going on",
    "scene kya hai": "what's the scene",
    "haal kya hai": "how are you",
    "kya haal chaal hai": "how are things going",
    "kaise ho aap": "how are you",
    "kya haal hai bhai": "how are you bro",
    "kya scene hai yaar": "what's the scene dude",
    "sab theek hai": "everything is fine",
    "kuch nahi": "nothing",
    
    # Navigation phrases
    "le chalo": "take me",
    "le jao": "take me",
    "jaana hai": "want to go",
    "chalna hai": "want to go",
    "kaise pahuchu": "how do I reach",
    "rasta dikhao": "show me the way",
    "route chahiye": "need route",
    "map dikhao": "show me the map",
    "mujhe le chalo": "take me",
    "chaliye chalte hain": "let's go",
    "chalna hai udhar": "want to go there",
    "le chaliye": "let's go",
    "jaldi le chal": "take me quickly",
    "jana hai": "want to go",
    "waha le chlo": "take me there",
    "route dikaho": "show me the route",
    
    # Food and places
    "khana khane": "eat food",
    "khane chal": "let's eat",
    "kuch khana hai": "want to eat something",
    "kuch khilao": "feed me something",
    "kuch khila de": "give me something to eat",
    "kuch khana khilao": "feed me something",
    "kuch order kar": "order something",
    "kuch khane ka": "something to eat",
    
    # Common words
    "hai": "is",
    "hoon": "am",
    "kar": "do",
    "le": "take",
    "jao": "go",
    "chalo": "let's go",
    "kya": "what",
    "kaise": "how",
    "kahan": "where",
    "kab": "when",
    "main": "I",
    "mera": "my",
    "meri": "my",
    "mere": "my",
    "tum": "you",
    "aap": "you",
    "hum": "we",
    "yeh": "this",
    "woh": "that",
    "accha": "good",
    "theek": "okay",
    "sahi": "right",
    "bilkul": "absolutely",
    "haan": "yes",
    "nahi": "no",
    "na": "no",
    "yaar": "dude",
    "bhai": "bro",
    "dost": "friend",
    "ghar": "home",
    "office": "office",
    "market": "market",
    "school": "school",
    "college": "college",
    "restaurant": "restaurant",
    "hotel": "hotel",
    "station": "station",
    "airport": "airport",
    "traffic": "traffic",
    "jam": "jam",
    "road": "road",
    "car": "car",
    "bus": "bus",
    "train": "train",
    "metro": "metro",
    "time": "time",
    "samay": "time",
    "der": "late",
    "jaldi": "quickly",
    "khana": "food",
    "peena": "drink",
    "sona": "sleep",
    "uthna": "get up",
    "baithna": "sit",
    "chalna": "walk",
    "aana": "come",
    "jana": "go",
    "dekhna": "see",
    "sunna": "hear",
    "bolna": "speak",
    "karna": "do",
    "dena": "give",
    "lena": "take",
    "marna": "hit",
    "jeena": "live",
    "roti": "bread",
    "dal": "lentils",
    "chawal": "rice",
    "sabzi": "vegetables"
}

# --- Intent Mapping for Delhi Driver Hinglish ---
DELHI_DRIVER_INTENT_MAPPING = {
    # Navigation patterns
    "le chalo": "nav_start_navigation",
    "le jao": "nav_start_navigation", 
    "cp": "nav_start_navigation",
    "connaught place": "nav_start_navigation",
    "india gate": "nav_start_navigation",
    "chandni chowk": "nav_start_navigation",
    "route dikhao": "nav_start_navigation",
    "rasta batao": "nav_start_navigation",
    "navigate": "nav_start_navigation",
    "drive to": "nav_start_navigation",
    "take me to": "nav_start_navigation",
    "jaana hai": "nav_start_navigation",
    "chalna hai": "nav_start_navigation",
    "waha le chalo": "nav_start_navigation",
    "kaise pahuchu": "nav_start_navigation",
    "destination par le chalo": "nav_start_navigation",
    "drop location set karo": "nav_start_navigation",
    "mujhe le chalo": "nav_start_navigation",
    "chaliye chalte hain": "nav_start_navigation",
    "chalna hai udhar": "nav_start_navigation",
    "le chaliye": "nav_start_navigation",
    "jaldi le chal": "nav_start_navigation",
    "jana hai": "nav_start_navigation",
    "waha le chlo": "nav_start_navigation",
    "route dikaho": "nav_start_navigation",
    "le chlo": "nav_start_navigation",
    "le chal": "nav_start_navigation",
    "le jao": "nav_start_navigation",
    
    # Food patterns
    "chole bhature": "find_food_spot",
    "momos": "find_food_spot",
    "biryani": "find_food_spot",
    "khane chal": "find_food_spot",
    "kuch khana hai": "find_food_spot",
    "kuch khilao": "find_food_spot",
    "kuch khila de": "find_food_spot",
    "kuch khana khilao": "find_food_spot",
    "kuch order kar": "find_food_spot",
    "kuch khane ka": "find_food_spot",
    "khana khane": "find_food_spot",
    "hungry": "find_food_spot",
    "bhook": "find_food_spot",
    "khana": "find_food_spot",
    
    # Greeting patterns
    "kya scene hai": "general_greet",
    "kya haal hai": "general_greet",
    "kaise ho": "general_greet",
    "aaj ka kya scene hai": "general_greet",
    "scene kya hai": "general_greet",
    "haal kya hai": "general_greet",
    "kya chal raha": "general_greet",
    "kya haal chaal": "general_greet",
    "sab badhiya": "general_greet",
    "kaisa chal raha hai": "general_greet",
    "kya haal chaal hai": "general_greet",
    "kaise ho aap": "general_greet",
    "kya haal hai bhai": "general_greet",
    "kya scene hai yaar": "general_greet",
    "sab theek hai": "general_greet",
    "kuch nahi": "general_greet",
    
    # Thanks patterns
    "thanks": "thanks",
    "thank you": "thanks",
    "dhanyawad": "thanks",
    "shukriya": "thanks",
    "thanku": "thanks",
    "thanks bhai": "thanks",
    "thank you yaar": "thanks",
    
    # Help patterns
    "help": "panic_signal",
    "help chahiye": "panic_signal",
    "madad chahiye": "panic_signal",
    "kya kar sakta hai": "panic_signal",
    "kaise help kar sakta hai": "panic_signal",
    "kya help kar sakta hai": "panic_signal",
    "kya karoge": "panic_signal",
    "kya kar sakte ho": "general_greet",
    "how can you help": "panic_signal",
    "what can you do": "panic_signal",
    "help kar sakta hai": "panic_signal",
    "kya karte ho": "panic_signal",
    "kya kar sakta hai tu": "panic_signal",
    "kya kar sakte ho yuvi": "panic_signal",
    "kaise help karoge": "panic_signal",
    "kaise help kar sakte ho": "panic_signal",
    "kya kaam aata hai": "panic_signal",
    "kya feature hai": "panic_signal",
    "kya kar sakta hai yuvi": "panic_signal",
    
    # Traffic patterns
    "traffic": "nav_query_traffic",
    "traffic kaisa hai": "nav_query_traffic",
    "jam": "nav_query_traffic",
    "bheed": "nav_query_traffic",
    "road block": "nav_query_traffic",
    "rush": "nav_query_traffic",
    "jam laga": "nav_query_traffic",
    "bheed hai": "nav_query_traffic",
    "traffic ka": "nav_query_traffic",
    "how's the traffic": "nav_query_traffic",
    "any traffic on the way": "nav_query_traffic",
    "is there traffic": "nav_query_traffic",
    "traffic update": "nav_query_traffic",
    "road block ahead": "nav_query_traffic",
    "any congestion": "nav_query_traffic",
    "road conditions": "nav_query_traffic",
    "jam ahead": "nav_query_traffic",
    "heavy traffic": "nav_query_traffic",
    "is it jammed": "nav_query_traffic",
    "rush hour now": "nav_query_traffic",
    "traffic ka kya scene": "nav_query_traffic",
    "jam hai kya": "nav_query_traffic",
    "jam laga hai": "nav_query_traffic",
    "road band hai": "nav_query_traffic",
    "signal pe rukna hai": "nav_query_traffic",
    "police check hai": "nav_query_traffic",
    "diversion hai": "nav_query_traffic",
    
    # ETA patterns
    "eta": "nav_query_eta",
    "eta kitna": "nav_query_eta",
    "estimated time": "nav_query_eta",
    "estimated arrival": "nav_query_eta",
    "arrival time": "nav_query_eta",
    "kab tak": "nav_query_eta",
    "kitna time": "nav_query_eta",
    "kitna samay": "nav_query_eta",
    "kitna der": "nav_query_eta",
    "kitna time lagega": "nav_query_eta",
    "kab tak pahuchenge": "nav_query_eta",
    "kitne minute lagenge": "nav_query_eta",
    "kitne baje tak": "nav_query_eta",
    "kab tak pahuchna": "nav_query_eta",
    "kitne mein pahuch jaayenge": "nav_query_eta",
    "kitna waqt lagega": "nav_query_eta",
    "pahuchne mein kitna time": "nav_query_eta",
    "ka eta kya hai": "nav_query_eta",
    "time lagega kya": "nav_query_eta",
    "abhi kitna time": "nav_query_eta",
    "kitne der mein": "nav_query_eta",
    "destination tak kitna time": "nav_query_eta",
    "reach karne mein kitna": "nav_query_eta",
    "kitna samay lagega": "nav_query_eta",
    "kitne dino mein": "nav_query_eta",
    "ka reach time": "nav_query_eta",
    
    # Weather patterns
    "weather": "get_weather",
    "mausam": "get_weather",
    "weather kaisa hai": "get_weather",
    "mausam kaisa hai": "get_weather",
    "baarish": "get_weather",
    "rain": "get_weather",
    "garmi": "get_weather",
    "thand": "get_weather",
    "temperature": "get_weather",
    "humidity": "get_weather",
    "forecast": "get_weather",
    
    # Music patterns
    "music": "play_music",
    "song": "play_music",
    "music baja": "play_music",
    "song play": "play_music",
    "music laga": "play_music",
    "song laga": "play_music",
    "music chala": "play_music",
    "song chala": "play_music",
    "radio": "play_music",
    "radio laga": "play_music",
    "volume": "app_adjust_volume",
    "volume badha": "app_adjust_volume",
    "volume kam": "app_adjust_volume",
    "sound": "app_play_sound",
    "ac": "app_open_settings",
    "ac chala de": "app_open_settings",
    
    # Emergency patterns
    "emergency": "panic_signal",
    "help chahiye": "panic_signal",
    "madad": "panic_signal",
    "help": "panic_signal",  # <--- ADDED
    "madad chahiye": "panic_signal",
    "accident": "panic_signal",
    "gadi kharab": "panic_signal",
    "car breakdown": "panic_signal",
    "tyre puncture": "panic_signal",
    "puncture": "panic_signal",
    "engine problem": "panic_signal",
    "battery dead": "panic_signal",
    "roadside assistance": "panic_signal",
    "mechanic": "panic_signal",
    "towing": "panic_signal",
    "stuck": "panic_signal",
    
    # Time patterns
    "time": "query_time",
    "samay": "query_time",
    "baje": "query_time",
    "kitne baje": "query_time",
    "what time": "query_time",
    "kya time": "query_time",
    "kya samay": "query_time",
    
    # Date patterns
    "date": "query_date",
    "aaj": "query_date",
    "kal": "query_date",
    "parso": "query_date",
    "what date": "query_date",
    "kya date": "query_date",
    "kya din": "query_date",
    
    # Speed patterns
    "speed": "query_speed",
    "speed kya hai": "query_speed",
    "kitni speed": "query_speed",
    "speed limit": "query_speed",
    "speed kam": "query_speed",
    "speed badha": "query_speed",
    
    # System patterns
    "system": "query_system_status",
    "status": "query_system_status",
    "system status": "query_system_status",
    "kya status": "query_system_status",
    "system kaisa hai": "query_system_status",
    "status kya hai": "query_system_status"
}

def map_delhi_driver_intent(text: str, model_prediction: dict) -> dict:
    """Map model predictions to correct intents for Delhi driver Hinglish"""
    text_lower = text.lower()
    
    # Check if any Delhi driver patterns match
    for pattern, correct_intent in DELHI_DRIVER_INTENT_MAPPING.items():
        if pattern in text_lower:
            print(f"🔄 Mapping '{model_prediction['label']}' → '{correct_intent}' for pattern '{pattern}'")
            return {
                "label": correct_intent,
                "confidence": max(model_prediction.get("score", model_prediction.get("confidence", 0.0)), 0.8),  # Boost confidence
                "source": "delhi_driver_mapping"
            }
    
    # Special case: If model predicted music_play but text contains navigation words
    if model_prediction['label'] == 'music_play' and any(word in text_lower for word in ['chalo', 'jao', 'le', 'cp', 'connaught', 'india', 'chandni']):
        print(f"🔄 Fixing music_play → nav_start_navigation for navigation context")
        return {
            "label": "nav_start_navigation",
            "confidence": 0.9,
            "source": "delhi_driver_mapping_fix"
        }
    
    # Special case: If model predicted unsupported_query but text contains common Delhi patterns
    if model_prediction['label'] == 'unsupported_query':
        # Check for common Delhi patterns that should be recognized
        if any(word in text_lower for word in ['thanks', 'thank', 'dhanyawad', 'shukriya']):
            return {
                "label": "general_thank_you",
                "confidence": 0.9,
                "source": "delhi_driver_mapping_thanks"
            }
        elif any(word in text_lower for word in ['help', 'madad', 'kar sakta']):
            return {
                "label": "general_help",
                "confidence": 0.9,
                "source": "delhi_driver_mapping_help"
            }
        elif any(word in text_lower for word in ['scene', 'haal', 'kaise']):
            return {
                "label": "general_greet",
                "confidence": 0.9,
                "source": "delhi_driver_mapping_greeting"
            }
    
    # If no pattern matches, return original prediction
    return model_prediction

def generate_conversational_response(user_text: str, context: str = "") -> str:
    """Generate response using conversational model"""
    pipeline = get_conversational_pipeline()
    if not pipeline:
        return None
    
    tokenizer = pipeline["tokenizer"]
    model = pipeline["model"]
    
    # Format input
    instruction = "You are Yuvi, a friendly Delhi driver assistant. Respond in Hinglish with a Delhi driver persona."
    prompt = f"### Instruction:\n{instruction}\n\n### User:\n{user_text}\n\n### Assistant:\n"
    
    # Generate response
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant part
    if "### Assistant:" in response:
        assistant_response = response.split("### Assistant:")[-1].strip()
        return assistant_response
    else:
        return response

# --- Add a FastAPI endpoint for batch model performance testing ---
router = APIRouter()

class TestCase(BaseModel):
    user_id: str
    text: str
    expected_intent: str = None  # Optional

class TestBatchRequest(BaseModel):
    cases: List[TestCase]

@router.post("/test_model_performance")
def test_model_performance(request: TestBatchRequest):
    results = []
    for case in request.cases:
        # Call your NLU pipeline directly
        response = nlu_pipeline(TextInput(user_id=case.user_id, text=case.text))
        result = {
            "input": case.text,
            "predicted_intent": response["intent"]["label"],
            "expected_intent": case.expected_intent,
            "match": (response["intent"]["label"] == case.expected_intent) if case.expected_intent else None,
            "response_text": response["response_text"]
        }
        results.append(result)
    return {"results": results}

# --- Include the new router in the FastAPI app ---
app.include_router(router)

# --- SAMPLE TEST CASES (for /test_model_performance endpoint) ---
# Example payload:
# {
#   "cases": [
#     {"user_id": "test1", "text": "le chalo cp", "expected_intent": "nav_start_navigation"},
#     {"user_id": "test2", "text": "kya scene hai", "expected_intent": "greeting"},
#     {"user_id": "test3", "text": "mera naam amit hai", "expected_intent": "user_name_introduction"}
#   ]
# } 

# REMOVED: Global pipeline loads - using @lru_cache functions instead

@app.post("/voice")
async def process_voice(file: UploadFile = File(...)):
    # Save incoming audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_path = tmp.name
        tmp.write(await file.read())

    tts_path = f"{audio_path}.mp3"
    try:
        # 1. Transcribe Hinglish audio
        text = transcribe_audio(audio_path)
        print(f"🗣 Transcribed: {text}")

        # 1.5. Normalize text (LLM-powered)
        normalized_text = normalize_text_llm(text)
        print(f"📝 Normalized: {normalized_text}")

        # 2. Run intent/emotion/ner in parallel
        intent_raw, emotion_raw, ner_raw = run_nlu_in_parallel(
            normalized_text, get_intent_pipeline(), get_emotion_pipeline(), get_ner_pipeline()
        )
        intent = intent_raw[0]
        emotion = emotion_raw[0]
        print(f"🎯 Intent: {intent}, Emotion: {emotion}, NER: {ner_raw}")

        # 3. Format NER into list of {entity, value}
        entities = [{"entity": e["entity"], "value": e["word"]} for e in ner_raw]

        # 4. Generate response from your system
        debug = {
            "nlu_source": "parallel_pipeline",
            "original_text": text,
            "normalized_text": normalized_text
        }
        dst = {}  # Use your DST logic here
        response_obj = generate_response(
            intent_label=intent["label"] if "label" in intent else intent["label"],
            emotion_label=emotion["label"] if "label" in emotion else emotion["label"],
            dst=dst,
            debug=debug,
            user_text=normalized_text,
            entities=entities,
            summary=None,
            traffic_data=None,
            turn_by_turn_steps=None,
            profile=None
        )
        reply_text = response_obj["text"]

        # 5. TTS reply (gTTS)
        synthesize_reply(reply_text, tts_path, lang="hi")

        with open(tts_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")

        return {
            "transcript": text,
            "normalized_transcript": normalized_text,
            "intent": intent,
            "emotion": emotion,
            "entities": entities,
            "response_text": reply_text,
            "audio_base64": audio_base64
        }

    finally:
        os.remove(audio_path)
        if os.path.exists(tts_path):
            os.remove(tts_path)

SPORTS_API_KEY = "d170376a-5d99-4cc7-85c5-a39308274ee4"
NEWS_API_KEY = "5705da0b098949a2b3b4c86c9977053a"

# --- API Cricket Integration ---
def get_live_cricket_score(api_key=SPORTS_API_KEY, team1='india', team2='england'):
    url = f"https://api.cricapi.com/v1/currentMatches?apikey={api_key}&offset=0"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        # Try to find a match with both teams
        for match in data.get('data', []):
            teams = [t.lower() for t in match.get('teams', [])]
            if team1 in teams and team2 in teams:
                team1_name = match.get('teamInfo', [{}])[0].get('name', match.get('teams', [''])[0])
                team2_name = match.get('teamInfo', [{}])[-1].get('name', match.get('teams', ['',''])[-1])
                score = match.get('score', [{}])
                score_str = ''
                if score:
                    for s in score:
                        score_str += f"{s.get('inning', '')}: {s.get('r', '')}/{s.get('w', '')} in {s.get('o', '')} overs\n"
                return f"{team1_name} vs {team2_name} (Live):\n{score_str.strip()}\nStatus: {match.get('status', '')}"
        # If not found, try to scrape Cricbuzz
        return scrape_cricbuzz_score(team1, team2)
    except Exception as e:
        return f"Error fetching cricket score: {e}"

# --- Cricbuzz Scraping Fallback ---
def scrape_cricbuzz_score(team1='india', team2='england'):
    try:
        url = "https://www.cricbuzz.com/cricket-match/live-scores"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return f"Cricbuzz returned status code {r.status_code}."
        soup = BeautifulSoup(r.text, "html.parser")
        found = False
        for match in soup.find_all("div", class_="cb-mtch-lst"):
            text = match.get_text(separator=" ", strip=True)
            # Fuzzy match team names
            teams_in_text = [team1.lower() in text.lower(), team2.lower() in text.lower()]
            if all(teams_in_text):
                found = True
                return f"[Cricbuzz] {text}"
            # Try fuzzy matching if direct match fails
            if not all(teams_in_text):
                words = text.lower().split()
                close1 = difflib.get_close_matches(team1.lower(), words, n=1, cutoff=0.7)
                close2 = difflib.get_close_matches(team2.lower(), words, n=1, cutoff=0.7)
                if close1 and close2:
                    found = True
                    return f"[Cricbuzz] {text}"
        if not found:
            # Log a snippet of the HTML for debugging
            snippet = r.text[:1000]
            print(f"[Cricbuzz HTML snippet]:\n{snippet}")
            return f"Sorry, I couldn't find a recent or live {team1.title()} vs {team2.title()} match. Please check the official site for the latest score."
    except Exception as e:
        return f"Error scraping Cricbuzz: {e}"

# --- NewsAPI Integration ---
def get_latest_news(query, api_key=NEWS_API_KEY, top_n=3):
    print(f"🔑 NewsAPI key used for news query: {query}")
    url = f'https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={api_key}'
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data['status'] == 'ok' and data['articles']:
            articles = data['articles'][:top_n]
            news_list = [f"{i+1}. {article['title']} - {article['source']['name']} ({article['url']})" for i, article in enumerate(articles)]
            return "\n".join(news_list)
        else:
            return f"No recent news found for '{query}'."
    except Exception as e:
        return f"Error fetching news for '{query}': {e}"

def correct_common_asr_errors(text):
    corrections = {
        "ola driver": "ok driver",
        "ola driver.": "ok driver.",
        "ola drivers": "ok driver",
        "ola drive": "ok driver",
        # Add more as needed
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

def nlu_pipeline(input_data: TextInput):
    import time
    start_time = time.time()
    user_id = input_data.user_id
    text = correct_common_asr_errors(input_data.text)
    print(f"[NLU] Received input: user_id={user_id}, text='{text}'")
    # --- Emergency/Distress Heuristic (keep for safety) ---
    distress = detect_distress(text)
    if distress:
        print(f"[NLU] Distress detected: {distress}")
        response_text = "Distress signal mil gaya hai. Emergency services ko khabar kar raha hoon aur aapke contacts ko alert bhej raha hoon. Himmat rakhein."
        print(f"[NLU] Returning distress response: {response_text}")
        return {
            "text": input_data.text,
            "intent": distress,
            "emotion": {"label": "fear", "confidence": 1.0},
            "entities": [],
            "debug": {"fast_path": True, "distress": True},
            "route_summary": None,
            "turn_by_turn_steps": None,
            "route_geometry": None,
            "map_html": None,
            "response_text": response_text,
            "tts_audio_base64": None
        }
    # --- Use heuristics only for strict distress/greeting/news ---
    heuristic_intent = heuristic_intent_match(text)
    if heuristic_intent:
        print(f"[NLU] Heuristic intent detected: {heuristic_intent}")
        intent = heuristic_intent
        # For distress, return immediately for safety
        if intent["label"] == "distress":
            response_text = "Distress signal mil gaya hai. Emergency services ko khabar kar raha hoon aur aapke contacts ko alert bhej raha hoon. Himmat rakhein."
            print(f"[NLU] Returning distress response: {response_text}")
            return {
                "text": input_data.text,
                "intent": intent,
                "emotion": {"label": "fear", "confidence": 1.0},
                "entities": [],
                "debug": {"fast_path": True, "distress": True},
                "route_summary": None,
                "turn_by_turn_steps": None,
                "route_geometry": None,
                "map_html": None,
                "response_text": response_text,
                "tts_audio_base64": None
            }
        # For greeting, return a friendly greeting
        if intent["label"] == "greeting":
            response_text = "Namaste! Kaise madad kar sakta hoon?"
            print(f"[NLU] Returning greeting response: {response_text}")
            return {
                "text": input_data.text,
                "intent": intent,
                "emotion": {"label": "neutral", "confidence": 1.0},
                "entities": [],
                "debug": {"fast_path": True, "greeting": True},
                "route_summary": None,
                "turn_by_turn_steps": None,
                "route_geometry": None,
                "map_html": None,
                "response_text": response_text,
                "tts_audio_base64": None
            }
        # For news, continue to news logic below
    # --- Otherwise, use zero-shot classifier for all other queries ---
    print("[NLU] Running zero-shot classifier...")
    zsc_result = zero_shot_classifier(text, CANDIDATE_INTENTS)
    print(f"[NLU] Zero-shot classifier result: {zsc_result}")
    intent_label = zsc_result["labels"][0]
    intent = {"label": intent_label, "confidence": float(zsc_result["scores"][0]), "source": "zero_shot_xnli"}
    # --- Cricket Score and News API Integration ---
    if intent["label"] == "cricket_score":
        print("[NLU] Cricket score intent detected. Calling get_live_cricket_score...")
        try:
            response_text = get_live_cricket_score()
            print(f"[NLU] Cricket score response: {response_text}")
        except Exception as e:
            response_text = f"Error fetching cricket score: {e}"
            print(f"[NLU] Exception in cricket score: {e}")
        return {
            "text": input_data.text,
            "intent": intent,
            "emotion": {"label": "neutral", "confidence": 1.0},
            "entities": [],
            "debug": {"zero_shot": True},
            "route_summary": None,
            "turn_by_turn_steps": None,
            "route_geometry": None,
            "map_html": None,
            "response_text": response_text,
            "tts_audio_base64": None
        }
    if intent["label"] == "news_query":
        print("[NLU] News intent detected. Calling get_latest_news...")
        try:
            if any(loc in text.lower() for loc in ["delhi", "ncr"]):
                if "ncr" in text.lower():
                    response_text = get_latest_news("Delhi NCR")
                else:
                    response_text = get_latest_news("Delhi")
            else:
                response_text = get_latest_news(text)
            print(f"[NLU] NewsAPI response: {response_text}")
        except Exception as e:
            response_text = f"Error fetching news: {e}"
            print(f"[NLU] Exception in news: {e}")
        return {
            "text": input_data.text,
            "intent": intent,
            "emotion": {"label": "neutral", "confidence": 1.0},
            "entities": [],
            "debug": {"zero_shot": True},
            "route_summary": None,
            "turn_by_turn_steps": None,
            "route_geometry": None,
            "map_html": None,
            "response_text": response_text,
            "tts_audio_base64": None
        }
    print(f"[NLU] Returning fallback/other intent: {intent}")
    # ... rest of your pipeline logic ...

CANDIDATE_INTENTS = [
    "news",
    "latest news",
    "current affairs",
    "headlines",
    "local news",
    "breaking news",
    "cricket score",
    "navigation",
    "distress",
    "thanks",
    "greeting",
    "unsupported"
]

NEWS_LABELS = {"news", "latest news", "current affairs", "headlines", "local news", "breaking news"}