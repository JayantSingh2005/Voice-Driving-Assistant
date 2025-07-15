#!/usr/bin/env python3
"""
Smart Text Preprocessor for Delhi Driver Assistant
Handles spelling corrections, pronunciation variations, and fuzzy matching
"""

import re
import nltk
from fuzzywuzzy import fuzz, process
from spellchecker import SpellChecker
import json
import language_tool_python

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Example known phrases for Hinglish/phonetic correction
KNOWN_PHRASES = [
    "traffic kaisa hai",
    "le chalo cp",
    "chole bhature khane chal",
    "india gate",
    "qutub minar",
    "chandni chowk",
    "hauz khas",
    "aiims",
    "connaught place",
    # Add more as needed
]

class TextNormalizer:
    def __init__(self, custom_words=None):
        self.spell = SpellChecker(language=None)
        self.spell.word_frequency.load_words(KNOWN_PHRASES)
        if custom_words:
            self.spell.word_frequency.load_words(custom_words)
        # Example abbreviation/synonym mapping
        self.synonym_map = {
            "tmrw": "tomorrow",
            "traiphik": "traffic",
            "trafik": "traffic",
            "traphic": "traffic",
            "le jao": "le chalo",
            "let's go": "le chalo",
            "take me": "le chalo",
            # Add more as needed
        }

    def normalize(self, text):
        # Lowercase
        text = text.lower()
        # Remove extra spaces and punctuation (except for important ones)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Spell correction (word by word)
        words = text.split()
        corrected = [self.spell.correction(w) or w for w in words]
        text = ' '.join(corrected)
        # Expand abbreviations, standardize synonyms
        for k, v in self.synonym_map.items():
            if k in text:
                text = text.replace(k, v)
        return text

# Example usage:
# normalizer = TextNormalizer()
# normalized_text = normalizer.normalize("traiphik kaisa hai?")
# print(normalized_text)  # Output: 'traffic kaisa hai'

# --- Grammarly-like Normalizer using LanguageTool ---
class GrammarlyLikeNormalizer:
    def __init__(self, lang='en-US'):
        self.tool = language_tool_python.LanguageTool(lang)
        # For Hinglish, you can use 'en-IN' or custom logic

    def normalize(self, text, hinglish=False):
        # First, try English grammar/spell correction
        matches = self.tool.check(text)
        corrected = language_tool_python.utils.correct(text, matches)
        corrected = corrected.lower().strip()
        if not hinglish:
            return corrected
        # For Hinglish: try to transliterate Hindi words to Devanagari, correct, and transliterate back
        # (This is a placeholder; for real use, plug in a Hindi spellchecker or transformer model here)
        # Example: Use indic-transliteration for script conversion
        # from indic_transliteration.sanscript import transliterate, ITRANS, DEVANAGARI
        # corrected = transliterate(corrected, ITRANS, DEVANAGARI)
        # ... run Hindi spellchecker/grammar correction ...
        # corrected = transliterate(corrected, DEVANAGARI, ITRANS)
        # For now, just return the English-corrected text
        return corrected

# Example usage:
# normalizer = GrammarlyLikeNormalizer()
# print(normalizer.normalize("traiphik kaisa hai?", hinglish=True))  # Output: 'traffic kaisa hai?'

class SmartTextPreprocessor:
    def __init__(self):
        self.spell_checker = SpellChecker()
        
        # Delhi-specific location variations
        self.location_variations = {
            # Qutub Minar variations
            "qutub minar": ["kutub minar", "kutub minaar", "qutub minaar", "kutub", "qutub"],
            "kutub minar": ["qutub minar", "qutub minaar", "kutub minaar", "kutub", "qutub"],
            "qutub minaar": ["kutub minar", "kutub minaar", "qutub minar", "kutub", "qutub"],
            "kutub minaar": ["qutub minar", "qutub minaar", "kutub minar", "kutub", "qutub"],
            
            # Connaught Place variations
            "connaught place": ["cp", "connaught", "connaught place", "connaught place cp"],
            "cp": ["connaught place", "connaught", "connaught place cp"],
            
            # India Gate variations
            "india gate": ["india gate", "india gate delhi", "gate"],
            "gate": ["india gate", "india gate delhi"],
            
            # Chandni Chowk variations
            "chandni chowk": ["chandni chowk", "chandni", "chandni chowk delhi"],
            "chandni": ["chandni chowk", "chandni chowk delhi"],
            
            # Paharganj variations
            "paharganj": ["paharganj", "pahar ganj", "paharganj delhi"],
            "pahar ganj": ["paharganj", "paharganj delhi"],
            
            # Hauz Khas variations
            "hauz khas": ["hauz khas", "hauz", "hauz khas delhi"],
            "hauz": ["hauz khas", "hauz khas delhi"],
            
            # Saket variations
            "saket": ["saket", "saket delhi"],
            
            # Karol Bagh variations
            "karol bagh": ["karol bagh", "karol", "karol bagh delhi"],
            "karol": ["karol bagh", "karol bagh delhi"],
            
            # Lajpat Nagar variations
            "lajpat nagar": ["lajpat nagar", "lajpat", "lajpat nagar delhi"],
            "lajpat": ["lajpat nagar", "lajpat nagar delhi"],
            
            # Sarojini Nagar variations
            "sarojini nagar": ["sarojini nagar", "sarojini", "sarojini nagar delhi"],
            "sarojini": ["sarojini nagar", "sarojini nagar delhi"],
            
            # Janpath variations
            "janpath": ["janpath", "janpath delhi"],
            
            # Greater Kailash variations
            "greater kailash": ["greater kailash", "gk", "greater kailash delhi"],
            "gk": ["greater kailash", "greater kailash delhi"],
            
            # Red Fort variations
            "red fort": ["red fort", "lal qila", "red fort delhi"],
            "lal qila": ["red fort", "red fort delhi"],
            
            # Lotus Temple variations
            "lotus temple": ["lotus temple", "lotus", "lotus temple delhi"],
            "lotus": ["lotus temple", "lotus temple delhi"],
            
            # Akshardham variations
            "akshardham": ["akshardham", "akshardham temple", "akshardham delhi"],
            "akshardham temple": ["akshardham", "akshardham delhi"],
            
            # Airport variations
            "airport": ["indira gandhi international airport", "delhi airport", "igia", "igia airport"],
            "delhi airport": ["indira gandhi international airport", "igia", "igia airport"],
            "igia": ["indira gandhi international airport", "delhi airport", "igia airport"],
            
            # Railway Station variations
            "railway station": ["new delhi railway station", "delhi railway station", "ndls"],
            "ndls": ["new delhi railway station", "delhi railway station"],
            
            # Metro variations
            "metro": ["delhi metro", "metro station", "nearest metro"],
            "delhi metro": ["metro station", "nearest metro"],
            
            # Food variations
            "chole bhature": ["chole bhature", "chole bhatura", "chole", "bhature"],
            "chole bhatura": ["chole bhature", "chole", "bhature"],
            "momos": ["momos", "momo", "dumplings"],
            "momo": ["momos", "dumplings"],
            "biryani": ["biryani", "biriyani", "biryani rice"],
            "biriyani": ["biryani", "biryani rice"],
            "butter chicken": ["butter chicken", "murgh makhani", "butter chicken curry"],
            "murgh makhani": ["butter chicken", "butter chicken curry"],
            "parathe": ["parathe", "paratha", "paranthe"],
            "paratha": ["parathe", "paranthe"],
            "paranthe": ["parathe", "paratha"],
            "kebab": ["kebab", "kabab", "seekh kebab"],
            "kabab": ["kebab", "seekh kebab"],
            "golgappe": ["golgappe", "golgappa", "pani puri"],
            "golgappa": ["golgappe", "pani puri"],
            "pani puri": ["golgappe", "golgappa"],
            "rajma chawal": ["rajma chawal", "rajma rice", "rajma"],
            "chaat": ["chaat", "delhi chaat"],
            "paneer tikka": ["paneer tikka", "paneer", "tikka"],
            "dal makhani": ["dal makhani", "dal", "makhani"],
            "naan": ["naan", "naan bread"],
            "pizza": ["pizza", "pizza hut", "dominos"],
            "burger": ["burger", "burger king", "mcdonalds"],
            "samosa": ["samosa", "samose"],
            "samose": ["samosa"],
            "jalebi": ["jalebi", "jalebi sweet"],
            "lassi": ["lassi", "lassi drink"],
            "kulfi": ["kulfi", "kulfi ice cream"],
            "chaap": ["chaap", "soya chaap"],
            "rolls": ["rolls", "kathi rolls"],
            "shawarma": ["shawarma", "shwarma"],
            "shwarma": ["shawarma"],
            "tandoori": ["tandoori", "tandoori chicken"],
            "dosa": ["dosa", "masala dosa"],
            "idli": ["idli", "idli sambar"],
            "vada pav": ["vada pav", "vada pao"],
            "vada pao": ["vada pav"],
            "pav bhaji": ["pav bhaji", "pao bhaji"],
            "pao bhaji": ["pav bhaji"],
            "maggi": ["maggi", "maggi noodles"],
            "chai": ["chai", "tea", "masala chai"],
            "tea": ["chai", "masala chai"],
            "coffee": ["coffee", "filter coffee"]
        }
        
        # Create reverse mapping for quick lookup
        self.reverse_variations = {}
        for correct, variations in self.location_variations.items():
            for variation in variations:
                self.reverse_variations[variation.lower()] = correct
        
        # Load famous Delhi places for fuzzy matching
        self.famous_places = [
            "Qutub Minar", "Connaught Place", "India Gate", "Chandni Chowk", 
            "Paharganj", "Hauz Khas", "Saket", "Karol Bagh", "Lajpat Nagar",
            "Sarojini Nagar", "Janpath", "Greater Kailash", "Red Fort",
            "Lotus Temple", "Akshardham", "Indira Gandhi International Airport",
            "New Delhi Railway Station", "Delhi Metro"
        ]
        
        # Load famous Delhi foods for fuzzy matching
        self.famous_foods = [
            "chole bhature", "momos", "biryani", "butter chicken", "parathe",
            "kebab", "golgappe", "rajma chawal", "chaat", "paneer tikka",
            "dal makhani", "naan", "pizza", "burger", "samosa", "jalebi",
            "lassi", "kulfi", "chaap", "rolls", "shawarma", "tandoori",
            "dosa", "idli", "vada pav", "pav bhaji", "maggi", "chai", "coffee"
        ]

    def preprocess_text(self, text):
        """
        Main preprocessing function that handles spelling corrections and variations
        """
        original_text = text
        text = text.lower().strip()
        
        # Step 1: Check for exact location/food variations
        corrected_text = self._correct_location_variations(text)
        
        # Step 2: Fuzzy match for similar locations/foods
        if corrected_text == text:  # No exact match found
            corrected_text = self._fuzzy_match_locations_foods(text)
        
        # Step 3: General spell checking for other words
        if corrected_text == text:  # No location/food match found
            corrected_text = self._spell_check_text(text)
        
        # Step 4: Generate suggestions if corrections were made
        suggestions = []
        if corrected_text != text:
            suggestions.append(f"Did you mean: '{corrected_text}'?")
        
        return {
            "original_text": original_text,
            "corrected_text": corrected_text,
            "suggestions": suggestions,
            "was_corrected": corrected_text != text
        }

    def _correct_location_variations(self, text):
        """Correct known location and food variations"""
        # First, check for exact multi-word matches (higher priority)
        corrected_text = text
        
        # Sort variations by length (longest first) to avoid partial matches
        sorted_variations = sorted(self.reverse_variations.items(), key=lambda x: len(x[0]), reverse=True)
        
        for variation, correct in sorted_variations:
            if variation in corrected_text:
                corrected_text = corrected_text.replace(variation, correct)
                # Only replace once to avoid duplication
                break
        
        # If no multi-word match was found, try single word corrections
        if corrected_text == text:
            words = corrected_text.split()
            corrected_words = []
            
            for word in words:
                # Check for single word variations
                if word in self.reverse_variations:
                    corrected_words.append(self.reverse_variations[word])
                else:
                    corrected_words.append(word)
            
            corrected_text = " ".join(corrected_words)
        
        return corrected_text

    def _fuzzy_match_locations_foods(self, text):
        """Use fuzzy matching for similar locations and foods"""
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Try to match with famous places
            best_match_place = process.extractOne(word, self.famous_places, scorer=fuzz.ratio)
            if best_match_place and best_match_place[1] >= 80:  # 80% similarity threshold
                corrected_words.append(best_match_place[0].lower())
                continue
            
            # Try to match with famous foods
            best_match_food = process.extractOne(word, self.famous_foods, scorer=fuzz.ratio)
            if best_match_food and best_match_food[1] >= 80:  # 80% similarity threshold
                corrected_words.append(best_match_food[0])
                continue
            
            # No good match found, keep original word
            corrected_words.append(word)
        
        return " ".join(corrected_words)

    def _spell_check_text(self, text):
        """General spell checking for other words"""
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Skip short words and numbers
            if len(word) <= 2 or word.isdigit():
                corrected_words.append(word)
                continue
            
            # Check if word is misspelled
            if word not in self.spell_checker:
                correction = self.spell_checker.correction(word)
                if correction and correction != word:
                    corrected_words.append(correction)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        return " ".join(corrected_words)

    def get_suggestions(self, text):
        """Get suggestions for a given text"""
        result = self.preprocess_text(text)
        return result["suggestions"]

# Test the preprocessor
if __name__ == "__main__":
    preprocessor = SmartTextPreprocessor()
    
    test_cases = [
        "drive me to kutub minar",
        "le chalo cp",
        "chole bhatura khane chal",
        "kutub minar jaana hai",
        "qutub minaar le chalo",
        "connaught place jaana hai",
        "india gate le jao",
        "chandni chowk ka rasta dikhao",
        "pahar ganj mein khana khane chal",
        "hauz khas mein shopping karte hain",
        "saket mein movie dekhne chal",
        "karol bagh mein shopping",
        "lajpat nagar mein market",
        "sarojini nagar mein shopping",
        "janpath mein walk karte hain",
        "gk mein food court",
        "red fort dekhne chal",
        "lotus temple jaana hai",
        "akshardham temple darshan",
        "airport jaana hai",
        "railway station jaana hai",
        "metro station jaana hai",
        "momos khane chal",
        "biryani khana hai",
        "butter chicken order karo",
        "paratha khane chal",
        "kebab khana hai",
        "golgappa khane chal",
        "rajma chawal khana hai",
        "chaat khane chal",
        "paneer tikka order karo",
        "dal makhani khana hai",
        "naan khane chal",
        "pizza order karo",
        "burger khana hai",
        "samosa khane chal",
        "jalebi khana hai",
        "lassi peene chal",
        "kulfi khane chal",
        "chaap khana hai",
        "rolls khane chal",
        "shawarma order karo",
        "tandoori chicken khana hai",
        "dosa khane chal",
        "idli khana hai",
        "vada pav khane chal",
        "pav bhaji khane chal",
        "maggi khana hai",
        "chai peene chal",
        "coffee peene chal"
    ]
    
    print("🧪 Testing Smart Text Preprocessor...")
    print("=" * 60)
    
    for test_case in test_cases:
        result = preprocessor.preprocess_text(test_case)
        print(f"\n📝 Original: '{test_case}'")
        print(f"   ✅ Corrected: '{result['corrected_text']}'")
        if result['suggestions']:
            print(f"   💡 Suggestions: {result['suggestions']}")
        if result['was_corrected']:
            print(f"   🔄 CORRECTION APPLIED!")
    
    print("\n" + "=" * 60)
    print("✅ Smart text preprocessor testing complete!") 