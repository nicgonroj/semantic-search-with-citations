"""Normalización de texto en español para la recuperación léxica.

Minúsculas, sin tildes, sin palabras vacías y con un stemming liviano que
junta singular/plural y algunas derivaciones ("contratos" ≈ "contrato",
"renovación" ≈ "renovaciones"). No pretende ser un stemmer completo: basta
para que BM25 no dependa de la forma exacta de cada palabra.
"""
from __future__ import annotations

import re
import unicodedata

STOPWORDS = set("""
a al algo algun alguna algunas alguno algunos ante antes aqui asi aun cada como con contra cual
cuales cualquier cuando cuanto cuanta cuantos cuantas de del desde donde dos durante e el ella
ellas ellos en entre era es esa esas ese eso esos esta estan estas este esto estos fue fueron ha
han hasta hay la las le les lo los mas me mi mis mucho muy ni no nos o otra otras otro otros para
pero poco por porque puede pueden puedo que quien quienes se sea segun ser si sin sobre son su sus
tambien tan tener tiene tienen todo todos tu un una unas uno unos y ya yo debe deben debo hacer
hace tengo tenemos existe existen
hago necesito quiero nuestro nuestra nuestros nuestras usted alguien
compania empresa
""".split())

# Sufijos que se recortan, del más largo al más corto. El tope de largo mínimo
# del resto evita que palabras cortas queden irreconocibles.
_SUFFIXES = ("aciones", "imientos", "imiento", "amiento", "idades", "acion", "ciones", "cion",
             "mente", "idad", "ables", "able", "ibles", "ible", "ores", "ados", "adas", "idos",
             "idas", "ado", "ada", "ido", "ida", "or", "ar", "er", "ir", "an", "en", "es", "os", "as", "s", "o", "a", "e")
_MIN_STEM = 4


def normalize(text: str) -> str:
    """Minúsculas y sin tildes (la ñ pasa a n: aceptable para buscar)."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    """Texto → lista de términos normalizados, sin palabras vacías."""
    words = re.findall(r"[a-z0-9]+", normalize(text))
    return [stem(w) for w in words if w not in STOPWORDS and len(w) > 1]


def split_sentences(text: str) -> list[str]:
    """Separa oraciones por punto final; suficiente para textos de políticas."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
