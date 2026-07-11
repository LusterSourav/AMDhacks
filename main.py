"""
Track 1: Hybrid Token-Efficient Routing Agent
ponytail: 2 deps (sympy, openai), 2-tier pipeline (deterministic + cloud)
"""

import os
import re
import ast
import json
import sys
import subprocess
import tempfile
from collections import Counter
from itertools import product as iterproduct

import sympy
from sympy import Symbol, solve

from cloud import cloud


# ─── QUERY NORMALIZATION ─────────────────────────────────────────────────────

# ponytail: strip common prefixes/suffixes for better FACTS matching
_NORM_PATTERNS = [
    (r'^(?:can you |could you |please |kindly )', ''),
    (r'^(?:tell me about |describe |explain |what do you know about )', ''),
    (r"^(?:what is the |what's the |what is a |what's a )", ''),
    (r"^(?:what is |what's )", ''),
    (r'^(?:who is the |who was the |who is |who was )', ''),
    (r'^(?:where is the |where is |where was )', ''),
    (r'^(?:when is the |when was the |when is |when was |when did )', ''),
    (r'^(?:how many |how much |how far |how long |how old |how deep |how high )', ''),
    (r'^(?:define |definition of |meaning of )', ''),
    (r'^(?:name the |name the |list the )', ''),
    (r'[?.!]+$', ''),
    (r'\s+', ' '),
]


def _normalize_query(q: str) -> str:
    q = q.lower().strip()
    for pattern, replacement in _NORM_PATTERNS:
        q = re.sub(pattern, replacement, q)
    return q.strip()


# ─── CLASSIFIER ──────────────────────────────────────────────────────────────


def classify(query: str) -> str:
    q = query.lower()

    # ponytail: hard overrides for unambiguous keywords
    if re.search(r'\b(summarize|summarise|tldr|summary|summarizing|shorten|brief)\b', q):
        return "summarization"
    if re.search(r'\b(debug|fix|error|bug|traceback|exception|wrong|broken|fails?|crash|syntax)\b', q):
        return "code_debug"
    if re.search(r'\b(extract|identify|find|list|name)\b.*\b(entities|names|people|places|dates|emails|organizations)\b', q):
        return "ner"
    if re.search(r'@[\w\.-]+|[\w\.-]+@[\w\.-]+\.\w+|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', q):
        return "ner"
    if re.search(r'\b(write|create|generate|implement|build|code|check|determine)\b.*\b(function|script|program|class|method|def|palindrome|prime|anagram|implementation|traversal|sort|search|tree|graph|stack|queue)\b', q):
        return "code_gen"
    if re.search(r'\b(quicksort|mergesort|bubblesort|bfs|dfs|binary.search)\b', q):
        return "code_gen"
        return "code_gen"
    if re.search(r'\b(calculate|compute|solve|evaluate)\b|\d+[\+\-\*\/\^\=]+\d+|\bsqrt\b|\bfactorial\b|\b(x\s*[\+\-\*\/\=])\b|\b\d+%\b|\bhow much is\b|\bwhat is \d+', q):
        return "math"
    if re.search(r'\b(sentiment|feeling|emotion|positive|negative|neutral|opinion|review|amazing|wonderful|terrible|horrible|love|hate)\b', q):
        return "sentiment"
    if re.search(r'\b(who|what|when|where)\b.*\b(is|was|are|were|wrote|invented|discovered|created)\b', q):
        return "factual"
    if re.search(r'logic|puzzle|riddle|if.*then|if.*,|deduce|conclude|constraint|assume|given that', q):
        return "logic"
    # ponytail: extra patterns for common queries
    if re.search(r'\b(define|definition|meaning of|explain|what does .* mean)\b', q):
        return "factual"
    if re.search(r'\b(largest|smallest|fastest|slowest|highest|lowest|most|least|longest|shortest)\b', q):
        return "factual"
    if re.search(r'\b(population|area|gdp|currency|language|capital|country|city|state)\b', q):
        return "factual"
    if re.search(r'\b(how many|how much|how far|how long|how old|how deep|how high)\b', q):
        return "factual"

    return "default"


# ─── MATH SOLVER ─────────────────────────────────────────────────────────────

# ponytail: word-to-number mapping for word problems
_WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
}


def _word_to_num(text: str) -> str:
    """Convert word numbers in text to digits. 'fifteen times seven' -> '15 * 7'."""
    result = text
    # replace word operators
    result = re.sub(r'\btimes\b', '*', result)
    result = re.sub(r'\bmultiplied by\b', '*', result)
    result = re.sub(r'\bplus\b', '+', result)
    result = re.sub(r'\badded to\b', '+', result)
    result = re.sub(r'\bminus\b', '-', result)
    result = re.sub(r'\bsubtracted from\b', '-', result)
    result = re.sub(r'\bdivided by\b', '/', result)
    result = re.sub(r'\bover\b', '/', result)
    result = re.sub(r'\bsquared\b', '**2', result)
    result = re.sub(r'\bcubed\b', '**3', result)
    result = re.sub(r'\bto the power of\b', '**', result)
    # handle composite numbers: "one hundred" -> "100", "one hundred twenty" -> "120"
    for hundreds_word, multiplier in [('hundred', 100), ('thousand', 1000)]:
        pattern = r'\b(\w+)\s+' + hundreds_word + r'\b'
        while re.search(pattern, result):
            match = re.search(pattern, result)
            prefix = match.group(1)
            if prefix in _WORD_NUMS:
                val = _WORD_NUMS[prefix] * multiplier
                result = result[:match.start()] + str(val) + result[match.end():]
    # replace remaining single word numbers
    for word, num in sorted(_WORD_NUMS.items(), key=lambda x: -x[1]):
        result = re.sub(r'\b' + word + r'\b', str(num), result)
    return result


def solve_math(query: str) -> str:
    q = query.lower()

    # ponytail: convert word numbers first
    q = _word_to_num(q)

    # ponytail: handle percentages first
    pct = re.search(r'(\d+)%\s*(?:of)?\s*(\d+)', q)
    if pct:
        result = int(pct.group(1)) * int(pct.group(2)) / 100
        return json.dumps({"answer": str(result)})

    # ponytail: unit conversions
    unit_conv = re.search(r'convert (\d+(?:\.\d+)?)\s*(\w+)\s*to\s*(\w+)', q)
    if unit_conv:
        val = float(unit_conv.group(1))
        src = unit_conv.group(2).lower()
        dst = unit_conv.group(3).lower()
        # temperature
        if src in ('c', 'celsius', 'centigrade') and dst in ('f', 'fahrenheit'):
            result = val * 9/5 + 32
            return json.dumps({"answer": f"{result:.1f} {dst}"})
        if src in ('f', 'fahrenheit') and dst in ('c', 'celsius', 'centigrade'):
            result = (val - 32) * 5/9
            return json.dumps({"answer": f"{result:.1f} {dst}"})
        # length
        if src in ('km', 'kilometers', 'kilometres') and dst in ('miles', 'mile'):
            result = val * 0.621371
            return json.dumps({"answer": f"{result:.2f} miles"})
        if src in ('miles', 'mile') and dst in ('km', 'kilometers', 'kilometres'):
            result = val * 1.60934
            return json.dumps({"answer": f"{result:.2f} km"})
        # weight
        if src in ('kg', 'kilograms', 'kilogrammes') and dst in ('lbs', 'pounds', 'pound'):
            result = val * 2.20462
            return json.dumps({"answer": f"{result:.2f} lbs"})
        if src in ('lbs', 'pounds', 'pound') and dst in ('kg', 'kilograms', 'kilogrammes'):
            result = val * 0.453592
            return json.dumps({"answer": f"{result:.2f} kg"})

    # ponytail: simple eval for arithmetic
    try:
        expr = q
        expr = re.sub(r'sqrt\(([^)]+)\)', r'(\1)**0.5', expr)
        expr = re.sub(r'\^', '**', expr)
        expr = re.sub(r'[^0-9\+\-\*\/\.\(\)\s\*]', '', expr)
        if expr.strip():
            result = eval(expr)
            return json.dumps({"answer": str(result)})
    except:
        pass

    # ponytail: sympy for equations
    try:
        eq_match = re.search(r'([\w\+\-\*\/\^\(\)\s]+)\s*=\s*([\w\+\-\*\/\^\(\)\s]+)', query)
        if eq_match:
            lhs, rhs = eq_match.group(1).strip(), eq_match.group(2).strip()
            x = Symbol('x')
            equation = sympy.sympify(lhs) - sympy.sympify(rhs)
            solutions = solve(equation, x)
            return json.dumps({"answer": str(solutions)})
    except:
        pass

    return None  # ponytail: let cloud handle complex math


# ─── SENTIMENT SOLVER ────────────────────────────────────────────────────────

POSITIVE = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'like', 'happy', 'best', 'awesome', 'perfect', 'brilliant',
            'outstanding', 'superb', 'nice', 'pleasant', 'enjoy', 'beautiful',
            'yes', 'agree', 'positive', 'recommend', 'impressive', 'remarkable',
            'delicious', 'fun', 'exciting', 'interesting', 'lovely', 'sweet',
            'kind', 'helpful', 'useful', 'effective', 'efficient', 'reliable',
            'safe', 'easy', 'fast', 'successful', 'beneficial', 'peaceful',
            'calm', 'relaxing', 'comfortable', 'cozy', 'warm', 'bright',
            'cheerful', 'grateful', 'thankful', 'blessed', 'proud', 'confident',
            'optimistic', 'hopeful', 'inspiring', 'motivating', 'encouraging'}

NEGATIVE = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike',
            'poor', 'ugly', 'boring', 'stupid', 'waste', 'disappointing',
            'annoying', 'frustrating', 'angry', 'sad', 'no', 'disagree',
            'negative', 'never', 'fail', 'broken', 'wrong', 'useless', 'sucks',
            'expensive', 'overpriced', 'costly', 'pricey', 'slow', 'worse',
            'painful', 'difficult', 'hard', 'upset', 'depressed', 'anxious',
            'worry', 'scared', 'fear', 'dangerous', 'risk', 'problem', 'issue',
            'mistake', 'error', 'loss', 'damage', 'harm', 'hateful', 'cruel',
            'nasty', 'rude', 'mean', 'selfish', 'greedy', 'lazy', 'dull',
            'ugliest', 'messy', 'dirty', 'toxic', 'poisonous', 'deadly'}

NEGATION_WORDS = {'not', "n't", 'never', 'no', 'neither', 'nor', 'nothing', 'nowhere', 'nobody'}

CONTRAST_WORDS = {'but', 'however', 'although', 'though', 'nevertheless', 'nonetheless', 'yet', 'whereas'}

INTENSIFIERS = {'very', 'extremely', 'incredibly', 'absolutely', 'totally', 'really', 'highly',
                'deeply', 'completely', 'entirely', 'quite', 'remarkably', 'exceptionally'}


def solve_sentiment(query: str) -> str:
    q_lower = query.lower()
    words = re.findall(r'\b\w+\b', q_lower)
    word_set = set(words)

    pos_count = len(word_set & POSITIVE)
    neg_count = len(word_set & NEGATIVE)
    intensifier_count = len(word_set & INTENSIFIERS)

    # ponytail: negation handling — if a negation word appears within 3 words
    # before a positive word, flip it to negative contribution
    negated_pos = 0
    for i, w in enumerate(words):
        if w in POSITIVE:
            # check up to 3 words before
            for j in range(max(0, i - 3), i):
                if words[j] in NEGATION_WORDS:
                    negated_pos += 1
                    break

    negated_neg = 0
    for i, w in enumerate(words):
        if w in NEGATIVE:
            for j in range(max(0, i - 3), i):
                if words[j] in NEGATION_WORDS:
                    negated_neg += 1
                    break

    # adjust: negated positive = negative contribution, negated negative = positive
    pos_count -= negated_pos
    neg_count -= negated_neg
    effective_pos = pos_count + negated_neg  # negated negatives count as positive
    effective_neg = neg_count + negated_pos  # negated positives count as negative

    # ponytail: contrast handling — "good but expensive" = mixed
    has_contrast = any(cw in q_lower for cw in CONTRAST_WORDS)

    if effective_pos > effective_neg:
        if has_contrast and effective_neg > 0:
            label = "neutral"
            score = 0.0
        else:
            label = "positive"
            score = min(1.0, 0.5 + effective_pos * 0.15 + intensifier_count * 0.05 - effective_neg * 0.1)
    elif effective_neg > effective_pos:
        if has_contrast and effective_pos > 0:
            label = "neutral"
            score = 0.0
        else:
            label = "negative"
            score = max(-1.0, -0.5 - effective_neg * 0.15 - intensifier_count * 0.05 + effective_pos * 0.1)
    else:
        label = "neutral"
        score = 0.0

    return json.dumps({"sentiment": label, "score": round(score, 2)})


# ─── NER SOLVER ──────────────────────────────────────────────────────────────

NER_PATTERNS = {
    "EMAIL": r'\b[\w\.-]+@[\w\.-]+\.\w+\b',
    "PHONE": r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
    "DATE": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}\b',
    "URL": r'https?://\S+|www\.\S+',
    "NUMBER": r'\b\d+(?:\.\d+)?\b',
    "ORG": r'\b(?:Inc|Corp|LLC|Ltd|Company|Organization)\b',
}


def solve_ner(query: str) -> str:
    entities = []
    for label, pattern in NER_PATTERNS.items():
        for match in re.finditer(pattern, query):
            entities.append({"text": match.group(), "type": label})
    return json.dumps({"entities": entities})


# ─── FACTUAL Q&A SOLVER ─────────────────────────────────────────────────────

# ponytail: hardcoded facts, no API dependency, covers ~800+ simple factual queries
FACTS = {
    # Country capitals (major countries)
    "capital of afghanistan": "Kabul",
    "capital of albania": "Tirana",
    "capital of algeria": "Algiers",
    "capital of argentina": "Buenos Aires",
    "capital of australia": "Canberra",
    "capital of austria": "Vienna",
    "capital of bangladesh": "Dhaka",
    "capital of belgium": "Brussels",
    "capital of bolivia": "Sucre",
    "capital of brazil": "Brasilia",
    "capital of bulgaria": "Sofia",
    "capital of cambodia": "Phnom Penh",
    "capital of cameroon": "Yaounde",
    "capital of canada": "Ottawa",
    "capital of chile": "Santiago",
    "capital of china": "Beijing",
    "capital of colombia": "Bogota",
    "capital of congo": "Kinshasa",
    "costa rica": "San Jose",
    "capital of croatia": "Zagreb",
    "capital of cuba": "Havana",
    "capital of czech republic": "Prague",
    "capital of czechia": "Prague",
    "capital of denmark": "Copenhagen",
    "capital of dominican republic": "Santo Domingo",
    "capital of ecuador": "Quito",
    "capital of egypt": "Cairo",
    "capital of el salvador": "San Salvador",
    "capital of ethiopia": "Addis Ababa",
    "capital of finland": "Helsinki",
    "capital of france": "Paris",
    "capital of germany": "Berlin",
    "capital of ghana": "Accra",
    "capital of greece": "Athens",
    "capital of guatemala": "Guatemala City",
    "capital of honduras": "Tegucigalpa",
    "capital of hungary": "Budapest",
    "capital of iceland": "Reykjavik",
    "capital of india": "New Delhi",
    "capital of indonesia": "Jakarta",
    "capital of iran": "Tehran",
    "capital of iraq": "Baghdad",
    "capital of ireland": "Dublin",
    "capital of israel": "Jerusalem",
    "capital of italy": "Rome",
    "capital of jamaica": "Kingston",
    "capital of japan": "Tokyo",
    "capital of jordan": "Amman",
    "capital of kenya": "Nairobi",
    "capital of kuwait": "Kuwait City",
    "capital of laos": "Vientiane",
    "capital of lebanon": "Beirut",
    "capital of libya": "Tripoli",
    "capital of madagascar": "Antananarivo",
    "capital of malaysia": "Kuala Lumpur",
    "capital of mexico": "Mexico City",
    "capital of mongolia": "Ulaanbaatar",
    "capital of morocco": "Rabat",
    "capital of mozambique": "Maputo",
    "capital of myanmar": "Naypyidaw",
    "capital of nepal": "Kathmandu",
    "capital of netherlands": "Amsterdam",
    "capital of new zealand": "Wellington",
    "capital of nigeria": "Abuja",
    "capital of north korea": "Pyongyang",
    "capital of norway": "Oslo",
    "capital of pakistan": "Islamabad",
    "capital of palestine": "Ramallah",
    "capital of panama": "Panama City",
    "capital of papua new guinea": "Port Moresby",
    "capital of paraguay": "Asuncion",
    "capital of peru": "Lima",
    "capital of philippines": "Manila",
    "capital of poland": "Warsaw",
    "capital of portugal": "Lisbon",
    "capital of qatar": "Doha",
    "capital of romania": "Bucharest",
    "capital of russia": "Moscow",
    "capital of saudi arabia": "Riyadh",
    "capital of senegal": "Dakar",
    "capital of serbia": "Belgrade",
    "capital of singapore": "Singapore",
    "capital of slovakia": "Bratislava",
    "capital of slovenia": "Ljubljana",
    "capital of south africa": "Pretoria",
    "capital of south korea": "Seoul",
    "capital of spain": "Madrid",
    "capital of sri lanka": "Sri Jayawardenepura Kotte",
    "capital of sudan": "Khartoum",
    "capital of sweden": "Stockholm",
    "capital of switzerland": "Bern",
    "capital of syria": "Damascus",
    "capital of taiwan": "Taipei",
    "capital of tanzania": "Dodoma",
    "capital of thailand": "Bangkok",
    "capital of tunisia": "Tunis",
    "capital of turkey": "Ankara",
    "capital of turkiye": "Ankara",
    "capital of uae": "Abu Dhabi",
    "capital of united arab emirates": "Abu Dhabi",
    "capital of uganda": "Kampala",
    "capital of ukraine": "Kyiv",
    "capital of united kingdom": "London",
    "capital of uk": "London",
    "capital of united states": "Washington, D.C.",
    "capital of usa": "Washington, D.C.",
    "capital of uruguay": "Montevideo",
    "capital of uzbekistan": "Tashkent",
    "capital of venezuela": "Caracas",
    "capital of vietnam": "Hanoi",
    "capital of yemen": "Sanaa",
    "capital of zimbabwe": "Harare",
    "capital of oman": "Muscat",
    "capital of bahrain": "Manama",
    "capital of bhutan": "Thimphu",
    "capital of maldives": "Male",
    "capital of montenegro": "Podgorica",
    "capital of north macedonia": "Skopje",
    "capital of kosovo": "Pristina",
    "capital of bosnia and herzegovina": "Sarajevo",
    "capital of albania": "Tirana",
    "capital of estonia": "Tallinn",
    "capital of latvia": "Riga",
    "capital of lithuania": "Vilnius",
    "capital of moldova": "Chisinau",
    "capital of belarus": "Minsk",
    "capital of georgia": "Tbilisi",
    "capital of armenia": "Yerevan",
    "capital of azerbaijan": "Baku",
    "capital of kazakhstan": "Astana",
    "capital of kyrgyzstan": "Bishkek",
    "capital of tajikistan": "Dushanbe",
    "capital of turkmenistan": "Ashgabat",
    "capital of monaco": "Monaco",
    "capital of liechtenstein": "Vaduz",
    "capital of san marino": "San Marino",
    "capital of vatican city": "Vatican City",
    "capital of andorra": "Andorra la Vella",
    "capital of malta": "Valletta",
    "capital of cyprus": "Nicosia",
    # US Presidents
    "who is the president of the united states": "As of 2026, the President is Donald Trump.",
    "who is the president of the usa": "As of 2026, the President is Donald Trump.",
    "who was the first president of the united states": "George Washington was the first President (1789-1797).",
    "who was the second president": "John Adams was the second President (1797-1801).",
    "who was abraham lincoln": "Abraham Lincoln was the 16th President (1861-1865), led during the Civil War and abolished slavery.",
    "who was the president during world war 2": "Franklin D. Roosevelt was President during most of WWII; Harry Truman finished it.",
    "who was thomas jefferson": "Thomas Jefferson was the 3rd President (1801-1809) and primary author of the Declaration of Independence.",
    "who was theodore roosevelt": "Theodore Roosevelt was the 26th President (1901-1909), known for conservation and the Panama Canal.",
    "who was john f kennedy": "John F. Kennedy was the 35th President (1961-1963), assassinated in Dallas.",
    "who was the youngest president": "Theodore Roosevelt became President at 42, the youngest person to hold the office.",
    "who was the oldest president": "Joe Biden was the oldest president inaugurated at 78.",
    # Science facts
    "what is the sun": "The Sun is a star at the center of our solar system.",
    "what is water": "Water is a chemical compound with formula H2O.",
    "what is python": "Python is a high-level programming language.",
    "what is ai": "Artificial Intelligence (AI) is the simulation of human intelligence by machines.",
    "what is artificial intelligence": "Artificial Intelligence (AI) is the simulation of human intelligence by machines.",
    "what is machine learning": "Machine Learning is a subset of AI where systems learn from data.",
    "what is deep learning": "Deep Learning is a subset of ML using neural networks with many layers.",
    "what is the earth": "Earth is the third planet from the Sun, the only known planet with life.",
    "how old is the earth": "Approximately 4.54 billion years old.",
    "how far is the moon": "About 384,400 km (238,855 miles) from Earth.",
    "what is speed of light": "299,792,458 meters per second in vacuum.",
    "speed of light": "299,792,458 meters per second in vacuum.",
    "what is gravity": "Gravity is a force of attraction between objects with mass.",
    "what is photosynthesis": "Photosynthesis is the process by which plants convert light energy to chemical energy.",
    "what is dna": "DNA (Deoxyribonucleic Acid) carries genetic instructions for life.",
    "what is the periodic table": "The periodic table organizes chemical elements by atomic number.",
    "what is the largest planet": "Jupiter is the largest planet in our solar system.",
    "what is the smallest planet": "Mercury is the smallest planet in our solar system.",
    "what is the closest star to earth": "The Sun is the closest star to Earth, about 93 million miles away.",
    "what is absolute zero": "Absolute zero is -273.15 degrees Celsius, the lowest possible temperature.",
    "what is avogadro's number": "6.022 x 10^23, the number of particles in one mole of a substance.",
    "what is the speed of sound": "About 343 meters per second in air at 20 degrees Celsius.",
    "what is the boiling point of water": "100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
    "what is the freezing point of water": "0 degrees Celsius (32 degrees Fahrenheit).",
    "what is electromagnetism": "One of the four fundamental forces, governing electric and magnetic interactions.",
    "what is the theory of relativity": "Einstein's theory describing space, time, and gravity. Special relativity (1905) and general relativity (1915).",
    "what is evolution": "The process by which species change over generations through natural selection.",
    "what is the human genome": "The complete set of DNA in humans, about 3 billion base pairs.",
    "what is a black hole": "A region of spacetime where gravity is so strong that nothing can escape.",
    "what is dark matter": "Hypothetical matter that does not emit light but exerts gravitational force.",
    "what is the atom": "The basic unit of a chemical element, consisting of a nucleus and electrons.",
    "what is newton's second law": "Force equals mass times acceleration (F = ma).",
    "what is the law of thermodynamics": "Energy cannot be created or destroyed, only transformed (First Law).",
    "what is entropy": "A measure of disorder in a system; tends to increase over time.",
    "what is the mitochondria": "The powerhouse of the cell, generating most of its ATP.",
    "what is the speed of an electron": "Electrons in atoms move at about 2.2 million m/s, about 1% the speed of light.",
    "what is a prime number": "A natural number greater than 1 that has no positive divisors other than 1 and itself.",
    "what is pi": "Pi is approximately 3.14159, the ratio of a circle's circumference to its diameter.",
    # Historical figures
    "who invented the telephone": "Alexander Graham Bell is credited with inventing the telephone.",
    "who wrote hamlet": "William Shakespeare wrote Hamlet.",
    "who was albert einstein": "Albert Einstein was a theoretical physicist who developed the theory of relativity.",
    "who invented the lightbulb": "Thomas Edison is credited with inventing the practical incandescent lightbulb.",
    "who invented the internet": "The internet evolved from ARPANET; key contributors include Vint Cerf and Bob Kahn.",
    "who was isaac newton": "Isaac Newton formulated the laws of motion and universal gravitation.",
    "who was leonardo da vinci": "Leonardo da Vinci was a Renaissance polymath: painter, scientist, engineer.",
    "who was charles darwin": "Charles Darwin developed the theory of evolution by natural selection.",
    "who was nikola tesla": "Nikola Tesla invented the alternating current (AC) electrical system.",
    "who was marie curie": "Marie Curie discovered radioactivity and won two Nobel Prizes.",
    "who wrote 1984": "George Orwell wrote 1984.",
    "who wrote romeo and juliet": "William Shakespeare wrote Romeo and Juliet.",
    "who painted the mona lisa": "Leonardo da Vinci painted the Mona Lisa.",
    "who was galileo": "Galileo Galilei was an astronomer who championed heliocentrism.",
    "who was mahatma gandhi": "Mahatma Gandhi led India's independence movement through nonviolent resistance.",
    "who was martin luther king jr": "Martin Luther King Jr. was a leader of the American civil rights movement.",
    "who was nelson mandela": "Nelson Mandela was the first Black president of South Africa and anti-apartheid leader.",
    "who was cleopatra": "Cleopatra VII was the last active ruler of the Ptolemaic Kingdom of Egypt.",
    "who was julius caesar": "Julius Caesar was a Roman general and dictator, assassinated in 44 BC.",
    # Common definitions
    "what is the internet": "A global network of interconnected computers using the TCP/IP protocol.",
    "what is a computer": "An electronic device that processes data according to programmed instructions.",
    "what is electricity": "The flow of electric charge through a conductor.",
    "what is a hormone": "A chemical messenger produced by glands and transported by blood.",
    "what is vitamin c": "Ascorbic acid, an essential vitamin for immune function and collagen synthesis.",
    "what is an algorithm": "A finite set of well-defined instructions for solving a problem.",
    "what is a database": "An organized collection of structured data stored electronically.",
    "what is encryption": "The process of converting data into a coded format to prevent unauthorized access.",
    "what is blockchain": "A distributed, immutable ledger that records transactions across many computers.",
    "what is climate change": "Long-term shifts in global temperatures and weather patterns, largely caused by human activities.",
    "what is democracy": "A system of government where citizens exercise power by voting.",
    "what is capitalism": "An economic system based on private ownership of the means of production.",
    "what is socialism": "An economic system based on social ownership and democratic control of the means of production.",
    "what is evolution": "The process of change in species over successive generations through natural selection.",
    "what is gravity": "The force of attraction between objects with mass.",
    "what is the atom": "The smallest unit of a chemical element, consisting of protons, neutrons, and electrons.",
    "what is a molecule": "A group of two or more atoms held together by chemical bonds.",
    "what is energy": "The capacity to do work; exists in kinetic, potential, thermal, and other forms.",
    "what is matter": "Anything that has mass and occupies space.",
    "what is force": "An interaction that changes the motion of an object (F = ma).",
    "what is velocity": "The speed of an object in a given direction.",
    "what is acceleration": "The rate of change of velocity over time.",
    "what is momentum": "The product of an object's mass and velocity (p = mv).",
    "what is wavelength": "The distance between successive crests of a wave.",
    "what is frequency": "The number of waves that pass a point per unit time.",
    "what is temperature": "A measure of the average kinetic energy of particles in a substance.",
    "what is pressure": "Force applied per unit area.",
    "what is density": "Mass per unit volume of a substance.",
    "what is friction": "The force that opposes motion between two surfaces in contact.",
    "what is inertia": "The tendency of an object to resist changes in its state of motion.",
    # US states and capitals
    "capital of alabama": "Montgomery",
    "capital of alaska": "Juneau",
    "capital of arizona": "Phoenix",
    "capital of arkansas": "Little Rock",
    "capital of california": "Sacramento",
    "capital of colorado": "Denver",
    "capital of connecticut": "Hartford",
    "capital of delaware": "Dover",
    "capital of florida": "Tallahassee",
    "capital of georgia state": "Atlanta",
    "capital of hawaii": "Honolulu",
    "capital of idaho": "Boise",
    "capital of illinois": "Springfield",
    "capital of indiana": "Indianapolis",
    "capital of iowa": "Des Moines",
    "capital of kansas": "Topeka",
    "capital of kentucky": "Frankfort",
    "capital of louisiana": "Baton Rouge",
    "capital of maine": "Augusta",
    "capital of maryland": "Annapolis",
    "capital of massachusetts": "Boston",
    "capital of michigan": "Lansing",
    "capital of minnesota": "Saint Paul",
    "capital of mississippi": "Jackson",
    "capital of missouri": "Jefferson City",
    "capital of montana": "Helena",
    "capital of nebraska": "Lincoln",
    "capital of nevada": "Carson City",
    "capital of new hampshire": "Concord",
    "capital of new jersey": "Trenton",
    "capital of new mexico": "Santa Fe",
    "capital of new york": "Albany",
    "capital of north carolina": "Raleigh",
    "capital of north dakota": "Bismarck",
    "capital of ohio": "Columbus",
    "capital of oklahoma": "Oklahoma City",
    "capital of oregon": "Salem",
    "capital of pennsylvania": "Harrisburg",
    "capital of rhode island": "Providence",
    "capital of south carolina": "Columbia",
    "capital of south dakota": "Pierre",
    "capital of tennessee": "Nashville",
    "capital of texas": "Austin",
    "capital of utah": "Salt Lake City",
    "capital of vermont": "Montpelier",
    "capital of virginia": "Richmond",
    "capital of washington": "Olympia",
    "capital of west virginia": "Charleston",
    "capital of wisconsin": "Madison",
    "capital of wyoming": "Cheyenne",
    # Currency and finance
    "currency of usa": "US Dollar (USD)",
    "currency of united states": "US Dollar (USD)",
    "currency of uk": "British Pound Sterling (GBP)",
    "currency of united kingdom": "British Pound Sterling (GBP)",
    "currency of europe": "Euro (EUR)",
    "currency of european union": "Euro (EUR)",
    "currency of japan": "Japanese Yen (JPY)",
    "currency of china": "Chinese Yuan (CNY)",
    "currency of india": "Indian Rupee (INR)",
    "currency of canada": "Canadian Dollar (CAD)",
    "currency of australia": "Australian Dollar (AUD)",
    "currency of switzerland": "Swiss Franc (CHF)",
    "currency of brazil": "Brazilian Real (BRL)",
    "currency of russia": "Russian Ruble (RUB)",
    "currency of mexico": "Mexican Peso (MXN)",
    "currency of south korea": "South Korean Won (KRW)",
    # More historical events
    "when was the declaration of independence signed": "August 2, 1776.",
    "when did world war 1 start": "July 28, 1914.",
    "when did world war 1 end": "November 11, 1918.",
    "when did world war 2 start": "September 1, 1939.",
    "when did world war 2 end": "September 2, 1945.",
    "when was the berlin wall built": "August 13, 1961.",
    "when did the berlin wall fall": "November 9, 1989.",
    "when was the Magna Carta signed": "1215.",
    "when was the first moon landing": "July 20, 1969. Neil Armstrong was the first person to walk on the Moon.",
    "who was the first person on the moon": "Neil Armstrong, on July 20, 1969.",
    "when was the printing press invented": "Around 1440 by Johannes Gutenberg.",
    "who discovered america": "Christopher Columbus reached the Americas in 1492.",
    "when was the french revolution": "1789 to 1799.",
    "when did the civil war start": "April 12, 1861.",
    "when did the civil war end": "May 26, 1865.",
    # More science
    "what is an electron": "A subatomic particle with negative charge, orbiting the nucleus of an atom.",
    "what is a proton": "A subatomic particle with positive charge, found in the nucleus of an atom.",
    "what is a neutron": "A subatomic particle with no charge, found in the nucleus of an atom.",
    "what is the nucleus": "The dense central region of an atom, containing protons and neutrons.",
    "what is a chemical bond": "A force that holds atoms together in a molecule.",
    "what is oxidation": "A chemical reaction involving loss of electrons.",
    "what is ph": "A measure of acidity or alkalinity, ranging from 0 to 14.",
    "what is the speed of light in mph": "About 670,616,629 mph.",
    "how far is the sun from earth": "About 93 million miles (150 million km).",
    "how many bones in the human body": "206 bones in an adult human.",
    "what is the largest organ": "The skin is the largest organ of the human body.",
    "what is the smallest bone": "The stapes in the middle ear is the smallest bone.",
    "how many chromosomes do humans have": "46 chromosomes (23 pairs).",
    "what is the powerhouse of the cell": "The mitochondria.",
    "what is the function of red blood cells": "Carry oxygen from the lungs to body tissues.",
    "what is the function of white blood cells": "Fight infection and disease.",
    # Geography
    "largest country by area": "Russia.",
    "smallest country by area": "Vatican City.",
    "largest ocean": "Pacific Ocean.",
    "smallest ocean": "Arctic Ocean.",
    "longest river": "The Nile River at about 6,650 km.",
    "tallest mountain": "Mount Everest at 8,849 meters.",
    "deepest ocean trench": "Mariana Trench at about 11,034 meters.",
    "largest desert": "Sahara Desert (largest hot desert); Antarctic Desert (largest overall).",
    "largest island": "Greenland.",
    "most populous country": "India (as of 2024).",
    "most populous city": "Tokyo metropolitan area.",
    "largest continent": "Asia.",
    "smallest continent": "Australia.",
    "largest freshwater lake": "Lake Superior.",
    "longest wall": "The Great Wall of China.",
    # Technology
    "who founded apple": "Steve Jobs, Steve Wozniak, and Ronald Wayne.",
    "who founded microsoft": "Bill Gates and Paul Allen.",
    "who founded google": "Larry Page and Sergey Brin.",
    "who founded amazon": "Jeff Bezos.",
    "who founded facebook": "Mark Zuckerberg.",
    "who founded tesla": "Martin Eberhard and Marc Tarpenning; Elon Musk joined early.",
    "who founded spacex": "Elon Musk in 2002.",
    "what is html": "HyperText Markup Language, the standard language for web pages.",
    "what is css": "Cascading Style Sheets, used for styling web pages.",
    "what is javascript": "A programming language for web browsers.",
    "what is sql": "Structured Query Language, used for database management.",
    "what is api": "Application Programming Interface, a set of rules for software communication.",
    "what is http": "HyperText Transfer Protocol, the foundation of data communication on the web.",
    "what is https": "HTTP Secure, the encrypted version of HTTP.",
    "what is tcp": "Transmission Control Protocol, a core internet protocol.",
    "what is ip": "Internet Protocol, the method for sending data across the internet.",
    "what is dns": "Domain Name System, translates domain names to IP addresses.",
    "what is ssh": "Secure Shell, a protocol for secure remote login.",
    "what is cloud computing": "Delivery of computing services over the internet.",
    "what is machine learning": "A subset of AI where systems learn from data.",
    "what is natural language processing": "AI ability to understand and generate human language.",
    "what is computer vision": "AI that interprets visual information from images or video.",
    # Biology
    "what is photosynthesis": "Process by which plants convert light energy to chemical energy.",
    "what is mitosis": "Cell division producing two identical daughter cells.",
    "what is meiosis": "Cell division producing four genetically unique gametes.",
    "what is dna": "Deoxyribonucleic acid, carries genetic instructions.",
    "what is rna": "Ribonucleic acid, involved in protein synthesis.",
    "what is a gene": "A segment of DNA that codes for a specific protein.",
    "what is a cell": "The basic structural unit of all living organisms.",
    "what is osmosis": "Movement of water through a semipermeable membrane.",
    "what is a ecosystem": "A community of living organisms interacting with their environment.",
    "what is biodiversity": "The variety of life in a particular ecosystem.",
    # Space
    "how many planets in the solar system": "8 planets.",
    "what are the planets": "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.",
    "what is a light year": "The distance light travels in one year, about 9.46 trillion km.",
    "what is a galaxy": "A massive system of stars, gas, and dust held together by gravity.",
    "what is the milky way": "The galaxy that contains our Solar System.",
    "what is a constellation": "A group of stars forming a pattern in the sky.",
    "what is a comet": "A celestial body of ice and dust that orbits the Sun.",
    "what is an asteroid": "A small rocky body orbiting the Sun.",
    "what is a meteor": "A streak of light caused by a meteoroid entering Earth's atmosphere.",
    "what is the ozone layer": "A layer of ozone gas in the stratosphere that protects from UV radiation.",
    # More chemistry
    "what is a chemical reaction": "A process that transforms one or more substances into different substances.",
    "what is an element": "A pure substance consisting of only one type of atom.",
    "what is a compound": "A substance made of two or more elements chemically bonded.",
    "what is a mixture": "A combination of two or more substances not chemically bonded.",
    "what is an acid": "A substance that donates hydrogen ions in water (pH < 7).",
    "what is a base": "A substance that accepts hydrogen ions in water (pH > 7).",
    "what is a salt": "A compound formed when an acid and a base react.",
    "what is a catalyst": "A substance that speeds up a chemical reaction without being consumed.",
    "what is a covalent bond": "A chemical bond where atoms share electron pairs.",
    "what is an ionic bond": "A chemical bond where electrons are transferred between atoms.",
    "what is a molecule": "A group of atoms bonded together.",
    "what is the periodic table": "The periodic table organizes chemical elements by atomic number and properties.",
    # More physics
    "what is quantum mechanics": "The branch of physics describing atomic and subatomic particles.",
    "what is wave-particle duality": "The principle that quantum entities behave as both particles and waves.",
    "what is the heisenberg uncertainty principle": "You cannot simultaneously know both position and momentum of a particle exactly.",
    "what is schrodingers cat": "A thought experiment showing quantum superposition, involving a cat that is both alive and dead.",
    "what is a photon": "A quantum of light, the smallest unit of electromagnetic radiation.",
    "what is kinetic energy": "Energy of motion, given by KE = 1/2 mv^2.",
    "what is potential energy": "Stored energy due to position or configuration.",
    "what is thermal energy": "Energy from the random motion of particles in a substance.",
    "what is nuclear fission": "Splitting of an atomic nucleus, releasing energy.",
    "what is nuclear fusion": "Combining of two light atomic nuclei, releasing energy (powers the Sun).",
    "what is static electricity": "An imbalance of electric charge on a surface.",
    "what is the difference between ac and dc": "AC alternates direction; DC flows in one direction.",
    "what is a magnet": "An object that produces a magnetic field, attracting ferromagnetic materials.",
    "what is resonance": "When a system vibrates at its natural frequency with maximum amplitude.",
    "what is a frequency": "The number of wave cycles per second, measured in Hertz.",
    "what is an amplitude": "The maximum displacement of a wave from its equilibrium position.",
    # Psychology
    "what is cognitive dissonance": "Mental discomfort from holding contradictory beliefs.",
    "what is confirmation bias": "Tendency to seek information confirming existing beliefs.",
    "what is the placebo effect": "Improvement from believing a treatment works, not from the treatment itself.",
    "what is classical conditioning": "Learning through association, studied by Pavlov.",
    "what is operant conditioning": "Learning through rewards and punishments, studied by Skinner.",
    "what is maslows hierarchy of needs": "A pyramid of human needs: physiological, safety, love, esteem, self-actualization.",
    "what is the dunning-kruger effect": "Unskilled individuals overestimate their ability, experts underestimate theirs.",
    "what is a phobia": "An extreme or irrational fear of something.",
    "what is anxiety": "A feeling of worry, nervousness, or unease about something.",
    "what is depression": "A mental health condition causing persistent low mood and loss of interest.",
    "what is ptsd": "Post-Traumatic Stress Disorder, caused by experiencing trauma.",
    # Economics
    "what is gdp": "Gross Domestic Product, the total value of goods and services produced in a country.",
    "what is inflation": "The rate at which the general level of prices for goods and services is rising.",
    "what is supply and demand": "An economic model where prices are determined by the relationship between supply and demand.",
    "what is interest rate": "The percentage charged or paid for borrowing or lending money.",
    "what is a stock": "A share of ownership in a company.",
    "what is a bond": "A fixed-income instrument representing a loan from an investor to a borrower.",
    "what is a recession": "A period of temporary economic decline, typically two consecutive quarters of negative GDP growth.",
    "what is monopoly": "When a single company dominates a market with no competition.",
    "what is oligopoly": "A market dominated by a small number of large firms.",
    "what is capitalism": "An economic system based on private ownership and profit motive.",
    "what is socialism": "An economic system based on social ownership of production means.",
    "what is a market economy": "An economy where prices are determined by supply and demand with minimal government intervention.",
    # More programming concepts
    "what is oop": "Object-Oriented Programming, organizes code into objects containing data and methods.",
    "what is recursion": "When a function calls itself to solve smaller instances of the same problem.",
    "what is a data structure": "A way to organize and store data for efficient access and modification.",
    "what is a linked list": "A linear data structure where elements are nodes connected by pointers.",
    "what is a tree": "A hierarchical data structure with a root node and child nodes.",
    "what is a graph": "A data structure consisting of vertices connected by edges.",
    "what is a hash table": "A data structure that maps keys to values using a hash function.",
    "what is big o notation": "Describes the time or space complexity of an algorithm as input size grows.",
    "what is a compiler": "A program that translates source code into machine code.",
    "what is an interpreter": "A program that executes source code line by line without compilation.",
    "what is a variable": "A named storage location in memory that holds a value.",
    "what is a function": "A reusable block of code that performs a specific task.",
    "what is an array": "A collection of elements stored at contiguous memory locations.",
    "what is a pointer": "A variable that stores the memory address of another variable.",
    "what is a class": "A blueprint for creating objects in object-oriented programming.",
    "what is inheritance": "A mechanism where a class derives properties from another class.",
    "what is polymorphism": "The ability of objects to take on multiple forms through a common interface.",
    "what is encapsulation": "The bundling of data and methods operating on that data within a single unit.",
    "what is a thread": "The smallest unit of execution within a process.",
    "what is a process": "An instance of a program running on a computer.",
    "what is a socket": "An endpoint for communication between two programs over a network.",
    "what is rest api": "A web API using HTTP methods (GET, POST, PUT, DELETE) and stateless communication.",
    "what is json": "JavaScript Object Notation, a lightweight data interchange format.",
    "what is xml": "Extensible Markup Language, used for structured data representation.",
    "what is yaml": "A human-readable data serialization language, often used for config files.",
    # More math
    "what is algebra": "A branch of mathematics dealing with symbols and rules for manipulating them.",
    "what is geometry": "A branch of mathematics dealing with shapes, sizes, and spatial properties.",
    "what is calculus": "A branch of mathematics dealing with rates of change (differentiation) and accumulation (integration).",
    "what is a logarithm": "The inverse operation of exponentiation log_b(a) = c means b^c = a.",
    "what is a derivative": "The rate of change of a function with respect to a variable.",
    "what is an integral": "The area under a curve, the inverse of a derivative.",
    "what is a matrix": "A rectangular array of numbers arranged in rows and columns.",
    "what is a vector": "A quantity having both magnitude and direction.",
    "what is a theorem": "A statement that has been proven to be true through logical reasoning.",
    "what is a proof": "A logical argument demonstrating the truth of a mathematical statement.",
    "what is the pythagorean theorem": "In a right triangle, a^2 + b^2 = c^2 where c is the hypotenuse.",
    "what is eulers number": "e is approximately 2.71828, the base of natural logarithms.",
    "what is fibonacci sequence": "A sequence where each number is the sum of the two preceding ones: 0, 1, 1, 2, 3, 5, 8...",
    "what is the golden ratio": "Phi, approximately 1.618, found in nature and art.",
    # More geography
    "what is the capital of canada": "Ottawa.",
    "what is the capital of australia": "Canberra.",
    "what is the capital of brazil": "Brasilia.",
    "what is the capital of india": "New Delhi.",
    "what is the capital of china": "Beijing.",
    "what is the capital of japan": "Tokyo.",
    "what is the capital of south africa": "Pretoria (administrative), Cape Town (legislative), Bloemfontein (judicial).",
    "what is the longest river in africa": "The Nile, about 6,650 km.",
    "what is the longest river in europe": "The Volga, about 3,530 km.",
    "what is the longest river in south america": "The Amazon, about 6,400 km.",
    "what is the largest lake in africa": "Lake Victoria.",
    "what is the largest lake in europe": "Lake Ladoga.",
    "what is the highest mountain in africa": "Mount Kilimanjaro.",
    "what is the highest mountain in europe": "Mount Elbrus.",
    "what is the highest mountain in north america": "Denali (Mount McKinley).",
    "what is the highest mountain in south america": "Aconcagua.",
    "how many continents are there": "Seven: Asia, Africa, North America, South America, Antarctica, Europe, Australia.",
    "how many oceans are there": "Five: Pacific, Atlantic, Indian, Southern, Arctic.",
    "what is the andes": "The longest continental mountain range, along South America.",
    "what is the himalayas": "A mountain range in Asia containing Mount Everest.",
    "what is the amazon rainforest": "The largest tropical rainforest, spanning 9 South American countries.",
    "largest desert in africa": "The Sahara Desert.",
    # More historical events
    "when was the roman empire founded": "27 BC when Augustus became the first emperor.",
    "when did the roman empire fall": "476 AD (Western Roman Empire).",
    "when was the industrial revolution": "1760 to 1840.",
    "when was the american revolution": "1765 to 1783.",
    "when was the russian revolution": "1917.",
    "when was the great depression": "1929 to 1939.",
    "when was the cold war": "1947 to 1991.",
    "when was the vietnam war": "1955 to 1975.",
    "when was the korean war": "1950 to 1953.",
    "who was the first emperor of china": "Qin Shi Huang, unified China in 221 BC.",
    "who built the great wall of china": "Built over centuries by various Chinese dynasties, notably the Ming.",
    "who was genghis khan": "Founder of the Mongol Empire, the largest contiguous land empire in history.",
    "who was alexander the great": "King of Macedonia who created one of the largest empires of the ancient world.",
    # Famous people
    "who is the richest person in the world": "As of 2026, the richest person varies; list includes Bernard Arnault, Elon Musk, and Jeff Bezos.",
    "who wrote the iliad": "Homer is credited with writing The Iliad.",
    "who wrote the odyssey": "Homer is credited with writing The Odyssey.",
    "who wrote the divine comedy": "Dante Alighieri.",
    "who wrote moby dick": "Herman Melville.",
    "who wrote the great gatsby": "F. Scott Fitzgerald.",
    "who wrote to kill a mockingbird": "Harper Lee.",
    "who wrote pride and prejudice": "Jane Austen.",
    "who wrote war and peace": "Leo Tolstoy.",
    "who wrote the catcher in the rye": "J.D. Salinger.",
    "who was plato": "A Greek philosopher, student of Socrates, teacher of Aristotle.",
    "who was socrates": "A Greek philosopher known for the Socratic method of questioning.",
    "who was aristotle": "A Greek philosopher, student of Plato, tutor of Alexander the Great.",
    "who was confucius": "A Chinese philosopher whose teachings emphasize morality and social harmony.",
    "who was hypatia": "A Greek mathematician and philosopher in Alexandria.",
    "who was florence nightingale": "Founder of modern nursing.",
    "who was mother teresa": "A Catholic nun who served the poor in Calcutta.",
    "who was wright brothers": "Orville and Wilbur Wright, inventors of the first successful airplane.",
    "who was thomas edison": "Inventor of the practical lightbulb, phonograph, and motion picture camera.",
    "who was alexander graham bell": "Inventor of the telephone.",
    "who was steve jobs": "Co-founder of Apple, pioneer of personal computers and smartphones.",
    "who was bill gates": "Co-founder of Microsoft.",
    "who was mark zuckerberg": "Co-founder of Facebook.",
    "who was stephen hawking": "A theoretical physicist known for black hole radiation.",
    "who was sigmund freud": "Founder of psychoanalysis.",
    # More biology
    "what is a virus": "A microscopic infectious agent that replicates only inside living cells.",
    "what is a bacteria": "Single-celled microorganisms that can cause disease or be beneficial.",
    "what is a fungus": "A kingdom of organisms including yeasts, molds, and mushrooms.",
    "what is a protein": "A molecule made of amino acids, essential for structure and function in living organisms.",
    "what is an enzyme": "A protein that catalyzes chemical reactions in living organisms.",
    "what is photosynthesis": "The process where plants use sunlight to convert CO2 and water into glucose and oxygen.",
    "what is respiration": "The process of converting glucose and oxygen into energy, CO2, and water.",
    "what is atp": "Adenosine Triphosphate, the energy currency of cells.",
    "what is a chromosome": "A thread-like structure of DNA carrying genetic information.",
    "what is a mutation": "A change in the DNA sequence of an organism.",
    "what is natural selection": "The process where organisms better adapted to their environment survive and reproduce.",
    "what is a stem cell": "An undifferentiated cell that can develop into different cell types.",
    "what is the circulatory system": "The heart and blood vessels that transport blood throughout the body.",
    "what is the nervous system": "Brain, spinal cord, and nerves that transmit signals throughout the body.",
    "what is the immune system": "The body's defense system against infections and diseases.",
    "what is a vaccine": "A biological preparation that provides active acquired immunity to a disease.",
    "what is an antibiotic": "A drug that kills or inhibits bacteria.",
    "what is a hormone": "A chemical messenger produced by glands and transported by blood.",
    "what is insulin": "A hormone that regulates blood glucose levels.",
    "what is hemoglobin": "The protein in red blood cells that carries oxygen.",
}


def solve_factual(query: str) -> str:
    q = _normalize_query(query)

    # ponytail: direct lookup first
    if q in FACTS:
        return json.dumps({"answer": FACTS[q]})

    # ponytail: fuzzy match
    for key, val in FACTS.items():
        if key in q or q in key:
            return json.dumps({"answer": val})

    return None  # let cloud/llm handle unknown facts


# ─── SUMMARIZATION SOLVER ────────────────────────────────────────────────────

def solve_summarization(query: str) -> str:
    # ponytail: extractive summarization, no LLM needed
    sentences = re.split(r'[.!?]+', query)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if len(sentences) <= 2:
        return json.dumps({"summary": query})

    # ponytail: TF scoring, take top sentences
    words = re.findall(r'\b\w+\b', query.lower())
    freq = Counter(words)
    # remove stopwords
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                 'as', 'into', 'through', 'during', 'before', 'after', 'and',
                 'but', 'or', 'nor', 'not', 'no', 'so', 'if', 'then', 'than',
                 'that', 'this', 'these', 'those', 'it', 'its', 'i', 'me',
                 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she',
                 'her', 'they', 'them', 'their', 'what', 'which', 'who', 'whom'}
    for sw in stopwords:
        freq.pop(sw, None)

    scored = []
    for s in sentences:
        s_words = set(re.findall(r'\b\w+\b', s.lower()))
        score = sum(freq.get(w, 0) for w in s_words)
        scored.append((score, s))

    scored.sort(reverse=True)
    n = max(1, len(sentences) // 3)
    summary = '. '.join(s for _, s in scored[:n])

    return json.dumps({"summary": summary})


# ─── CODE DEBUG SOLVER ───────────────────────────────────────────────────────

def solve_code_debug(query: str) -> str:
    # ponytail: extract code blocks or treat whole query as code
    code_match = re.search(r'```[\w]*\n(.*?)```', query, re.DOTALL)
    code = code_match.group(1) if code_match else query

    errors = []

    # ponytail: syntax check via ast
    try:
        ast.parse(code)
    except SyntaxError as e:
        errors.append({
            "type": "SyntaxError",
            "message": str(e.msg),
            "line": e.lineno,
            "offset": e.offset,
        })

    # ponytail: common bug patterns
    patterns = [
        (r'except\s*:', 'bare except clause - use specific exception type'),
        (r'print\s+[^(]', 'print statement without parentheses (Python 3?)'),
        (r'==\s*None', 'use "is None" instead of "== None"'),
        (r'!=\s*None', 'use "is not None" instead of "!= None"'),
        (r'except\s+Exception\s*:\s*pass', 'swallowing all exceptions silently'),
        (r'import\s+\*', 'wildcard import - import specific names'),
    ]
    for pat, msg in patterns:
        if re.search(pat, code):
            errors.append({"type": "warning", "message": msg})

    if errors:
        return json.dumps({"errors": errors, "fixed_suggestion": "See errors above"})
    return json.dumps({"status": "no errors found"})


# ─── LOGIC PUZZLE SOLVER ─────────────────────────────────────────────────────

def solve_logic(query: str) -> str:
    q = query.lower()

    # ponytail: "if X then Y" or "if X, Y" patterns
    rules = re.findall(r'if\s+(.+?)(?:\s+then|,)\s+(.+?)(?:\.|$)', q)
    if rules:
        conclusions = [cons.strip() for _, cons in rules]
        # ponytail: check for stated facts and apply rules
        facts_stated = re.findall(r'(.+?)\.(?:\s|$)', q)
        applied = []
        for fact in facts_stated:
            fact = fact.strip()
            for antecedent, consequent in rules:
                if antecedent.strip() in fact or fact in antecedent.strip():
                    applied.append(consequent.strip())
        return json.dumps({
            "rules_found": len(rules),
            "conclusions": conclusions,
            "applied": applied,
        })

    # ponytail: try itertools for small combinatorial puzzles
    numbers = re.findall(r'\d+', q)
    if ('sum' in q or 'add' in q) and numbers and len(numbers) <= 8:
        nums = [int(n) for n in numbers]
        target = None
        target_match = re.search(r'(?:sum|equals?|=\s*)(\d+)', q)
        if target_match:
            target = int(target_match.group(1))
        if target:
            for length in range(2, len(nums) + 1):
                for combo in iterproduct(nums, repeat=length):
                    if sum(combo) == target:
                        return json.dumps({"solution": list(combo), "sum": target})

    return None  # let cloud handle complex logic


# ─── CODE GENERATION SOLVER ──────────────────────────────────────────────────

# ponytail: common code templates, covers majority of code gen requests
CODE_TEMPLATES = {
    "fibonacci": """def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b""",
    "factorial": """def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result""",
    "sort": """def sort_list(lst):
    return sorted(lst)""",
    "reverse_string": """def reverse_string(s):
    return s[::-1]""",
    "is_palindrome": """def is_palindrome(s):
    s = s.lower().replace(" ", "")
    return s == s[::-1]""",
    "binary_search": """def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1""",
    "read_file": """def read_file(path):
    with open(path, 'r') as f:
        return f.read()""",
    "is_prime": """def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True""",
    "gcd": """def gcd(a, b):
    while b:
        a, b = b, a % b
    return a""",
    "lcm": """def lcm(a, b):
    return a * b // gcd(a, b)

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a""",
    "flatten": """def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result""",
    "unique": """def unique(lst):
    return list(set(lst))""",
    "word_count": """def word_count(text):
    words = text.lower().split()
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return counts""",
    "char_frequency": """def char_frequency(text):
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    return freq""",
    "capitalize_words": """def capitalize_words(text):
    return ' '.join(w.capitalize() for w in text.split())""",
    "anagram": """def is_anagram(s1, s2):
    return sorted(s1.lower()) == sorted(s2.lower())""",
    "binary_search_recursive": """def binary_search_recursive(arr, target, lo=0, hi=None):
    if hi is None:
        hi = len(arr) - 1
    if lo > hi:
        return -1
    mid = (lo + hi) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, hi)
    else:
        return binary_search_recursive(arr, target, lo, mid - 1)""",
    "linear_search": """def linear_search(arr, target):
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1""",
    "find_all": """def find_all(lst, predicate):
    return [x for x in lst if predicate(x)]""",
    "count_occurrences": """def count_occurrences(lst, target):
    return lst.count(target)""",
    "merge_sorted": """def merge_sorted(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result""",
    "frequency_count": """def frequency_count(lst):
    freq = {}
    for item in lst:
        freq[item] = freq.get(item, 0) + 1
    return freq""",
    "stack": """class Stack:
    def __init__(self):
        self.items = []
    def push(self, item):
        self.items.append(item)
    def pop(self):
        return self.items.pop() if self.items else None
    def peek(self):
        return self.items[-1] if self.items else None
    def is_empty(self):
        return len(self.items) == 0""",
    "queue": """class Queue:
    def __init__(self):
        self.items = []
    def enqueue(self, item):
        self.items.append(item)
    def dequeue(self):
        return self.items.pop(0) if self.items else None
    def is_empty(self):
        return len(self.items) == 0""",
    "linked_list_node": """class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next""",
    "binary_tree_node": """class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right""",
    "flatten_dict": """def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)""",
    "deep_merge": """def deep_merge(a, b):
    result = a.copy()
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result""",
    "memoize": """def memoize(func):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper""",
    "retry": """def retry(func, max_attempts=3):
    def wrapper(*args, **kwargs):
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise e
    return wrapper""",
    "bubble_sort": """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr""",
    "selection_sort": """def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr""",
    "insertion_sort": """def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr""",
    "merge_sort": """def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge_sorted(left, right)""",
    "quick_sort": """def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)""",
    "bst_insert": """def bst_insert(root, val):
    if root is None:
        return TreeNode(val)
    if val < root.val:
        root.left = bst_insert(root.left, val)
    else:
        root.right = bst_insert(root.right, val)
    return root""",
    "bst_search": """def bst_search(root, val):
    if root is None or root.val == val:
        return root
    if val < root.val:
        return bst_search(root.left, val)
    return bst_search(root.right, val)""",
    "bfs": """def bfs(graph, start):
    visited, queue = set(), [start]
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            queue.extend(n for n in graph[node] if n not in visited)
    return visited""",
    "dfs": """def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for next in graph[start]:
        if next not in visited:
            dfs(graph, next, visited)
    return visited""",
    "valid_parentheses": """def is_valid(s):
    stack, pairs = [], {'(': ')', '{': '}', '[': ']'}
    for c in s:
        if c in pairs:
            stack.append(c)
        else:
            if not stack or pairs[stack.pop()] != c:
                return False
    return len(stack) == 0""",
    "string_compression": """def compress(s):
    result = []
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            count += 1
        else:
            result.append(f"{s[i-1]}{count}")
            count = 1
    result.append(f"{s[-1]}{count}")
    compressed = ''.join(result)
    return compressed if len(compressed) < len(s) else s""",
    "linked_list_cycle": """def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False""",
    "reverse_linked_list": """def reverse_list(head):
    prev = None
    while head:
        nxt = head.next
        head.next = prev
        prev = head
        head = nxt
    return prev""",
    "two_sum": """def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []""",
    "max_subarray": """def max_subarray(nums):
    max_ending = max_sofar = nums[0]
    for x in nums[1:]:
        max_ending = max(x, max_ending + x)
        max_sofar = max(max_sofar, max_ending)
    return max_sofar""",
    "longest_common_prefix": """def longest_common_prefix(strs):
    if not strs:
        return ''
    for i, c in enumerate(strs[0]):
        for s in strs[1:]:
            if i >= len(s) or s[i] != c:
                return strs[0][:i]
    return strs[0]""",
    "coin_change": """def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for a in range(1, amount + 1):
        for c in coins:
            if a >= c:
                dp[a] = min(dp[a], dp[a - c] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1""",
    "knapsack": """def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][capacity]""",
    "levenshtein_distance": """def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n]""",
    "convert_temperature": """def celsius_to_fahrenheit(c):
    return c * 9/5 + 32

def fahrenheit_to_celsius(f):
    return (f - 32) * 5/9""",
}


def solve_code_gen(query: str) -> str:
    q = query.lower().replace(' ', '_')
    q_plain = query.lower()

    # ponytail: template matching — direct substring or all words match
    for name, template in CODE_TEMPLATES.items():
        if name in q:
            return json.dumps({"code": template, "language": "python"})
        words = name.split('_')
        if len(words) >= 2 and all(w in q_plain for w in words):
            return json.dumps({"code": template, "language": "python"})

    return None  # let cloud handle complex code gen


# ─── CODE EXECUTION (math/logic) ──────────────────────────────────────────────

def _exec_code(code: str, timeout: int = 5) -> str:
    """Execute Python code safely, return stdout or None."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            result = subprocess.run(
                ['python', f.name],
                capture_output=True, text=True, timeout=timeout
            )
            os.unlink(f.name)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
    except (subprocess.TimeoutExpired, Exception):
        pass
    return None


def solve_math_exec(query: str) -> str:
    """Try to solve math by generating and executing Python code."""
    q = query.lower()
    # generate a simple Python expression
    expr = q
    expr = re.sub(r'sqrt\(([^)]+)\)', r'(\1)**0.5', expr)
    expr = re.sub(r'\^', '**', expr)
    expr = re.sub(r'[^0-9\+\-\*\/\.\(\)\s\*]', '', expr)
    if expr.strip():
        code = f"print({expr})"
        result = _exec_code(code)
        if result:
            return json.dumps({"answer": result})
    return None


# ─── ROUTER ──────────────────────────────────────────────────────────────────

LOCAL_SOLVERS = {
    "math": solve_math,
    "sentiment": solve_sentiment,
    "ner": solve_ner,
    "factual": solve_factual,
    "code_debug": solve_code_debug,
    "logic": solve_logic,
    "code_gen": solve_code_gen,
}


def route(query: str) -> dict:
    task_type = classify(query)

    # ponytail: Tier 0 — deterministic solvers (0 tokens)
    if task_type in LOCAL_SOLVERS:
        result = LOCAL_SOLVERS[task_type](query)
        if result:
            return {"task": task_type, "source": "local", "answer": result}

    # ponytail: Tier 0.5 — code execution for math/logic (0 tokens)
    if task_type in ("math", "logic"):
        result = solve_math_exec(query)
        if result:
            return {"task": task_type, "source": "local_exec", "answer": result}

    # ponytail: Tier 1 — cloud fallback (token cost)
    try:
        answer = cloud(query, task_type)
        return {"task": task_type, "source": "cloud", "answer": answer}
    except Exception as e:
        return {"task": task_type, "source": "error", "answer": str(e)}


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    # ponytail: read from stdin or args
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
    else:
        query = sys.stdin.read().strip()

    if not query:
        print(json.dumps({"error": "no query provided"}))
        return

    result = route(query)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
