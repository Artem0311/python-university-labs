"""Потокова токенізація Unicode-тексту для пошукової системи findex.

Правила:
* Спочатку нормалізація NFC, потім casefold для нечутливості до регістру.
* Апостроф усередині слова зберігається (п'ять, don't). Символи ’ та ʼ
  приводяться до звичайного апострофа ', щоб варіанти запису збігалися.
* Дефіси та тире розділяють слова: «веб-застосунок» -> «веб», «застосунок».
* Цифри зберігаються: «2026» і «python3» є окремими токенами.
* Підкреслення і решта пунктуації розділяють токени.
* Токени повертаються по одному через генератор і re.finditer.
"""

import re
import unicodedata
from collections.abc import Iterator

_APOSTROPHE_MAP = str.maketrans({"’": "'", "ʼ": "'"})
_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)*", flags=re.UNICODE)


def tokenize(text: str) -> Iterator[str]:
    """Повертає нормалізовані токени один за одним, без списку слів."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = normalized.translate(_APOSTROPHE_MAP)

    for match in _TOKEN_PATTERN.finditer(normalized):
        yield match.group()
