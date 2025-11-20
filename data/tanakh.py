"""
Статичные данные для навигации по Танаху.
Стандартная структура 24 книг.
"""

from typing import Dict, Any, List

# Структура Танаха: книга -> количество глав
TANAKH_BOOKS: Dict[str, Dict[str, Any]] = {
    # Тора (5 книг)
    "Genesis": {
        "chapters": 50,
        "section": "Torah",
        "he_name": "בראשית",
        "ru_name": "Берешит",
        "order": 1
    },
    "Exodus": {
        "chapters": 40,
        "section": "Torah", 
        "he_name": "שמות",
        "ru_name": "Шмот",
        "order": 2
    },
    "Leviticus": {
        "chapters": 27,
        "section": "Torah",
        "he_name": "ויקרא", 
        "ru_name": "Вайикра",
        "order": 3
    },
    "Numbers": {
        "chapters": 36,
        "section": "Torah",
        "he_name": "במדבר",
        "ru_name": "Бемидбар", 
        "order": 4
    },
    "Deuteronomy": {
        "chapters": 34,
        "section": "Torah",
        "he_name": "דברים",
        "ru_name": "Дварим",
        "order": 5
    },
    
    # Невиим - Ранние пророки (4 книги)
    "Joshua": {
        "chapters": 24,
        "section": "Nevi'im",
        "he_name": "יהושע",
        "ru_name": "Йехошуа",
        "order": 6
    },
    "Judges": {
        "chapters": 21,
        "section": "Nevi'im",
        "he_name": "שופטים",
        "ru_name": "Шофтим",
        "order": 7
    },
    "I Samuel": {
        "chapters": 31,
        "section": "Nevi'im",
        "he_name": "שמואל א",
        "ru_name": "Шмуэль I",
        "order": 8
    },
    "II Samuel": {
        "chapters": 24,
        "section": "Nevi'im",
        "he_name": "שמואל ב",
        "ru_name": "Шмуэль II",
        "order": 9
    },
    "I Kings": {
        "chapters": 22,
        "section": "Nevi'im",
        "he_name": "מלכים א",
        "ru_name": "Млахим I",
        "order": 10
    },
    "II Kings": {
        "chapters": 25,
        "section": "Nevi'im",
        "he_name": "מלכים ב",
        "ru_name": "Млахим II",
        "order": 11
    },
    
    # Невиим - Поздние пророки (15 книг, но объединены в 4)
    "Isaiah": {
        "chapters": 66,
        "section": "Nevi'im",
        "he_name": "ישעיהו",
        "ru_name": "Йешаяху",
        "order": 12
    },
    "Jeremiah": {
        "chapters": 52,
        "section": "Nevi'im",
        "he_name": "ירמיהו",
        "ru_name": "Йирмеяху",
        "order": 13
    },
    "Ezekiel": {
        "chapters": 48,
        "section": "Nevi'im",
        "he_name": "יחזקאל",
        "ru_name": "Йехезкель",
        "order": 14
    },
    "Hosea": {
        "chapters": 14,
        "section": "Nevi'im",
        "he_name": "הושע",
        "ru_name": "Ошеа",
        "order": 15
    },
    "Joel": {
        "chapters": 4,
        "section": "Nevi'im",
        "he_name": "יואל",
        "ru_name": "Йоэль",
        "order": 16
    },
    "Amos": {
        "chapters": 9,
        "section": "Nevi'im",
        "he_name": "עמוס",
        "ru_name": "Амос",
        "order": 17
    },
    "Obadiah": {
        "chapters": 1,
        "section": "Nevi'im",
        "he_name": "עובדיה",
        "ru_name": "Овадья",
        "order": 18
    },
    "Jonah": {
        "chapters": 4,
        "section": "Nevi'im",
        "he_name": "יונה",
        "ru_name": "Йона",
        "order": 19
    },
    "Micah": {
        "chapters": 7,
        "section": "Nevi'im",
        "he_name": "מיכה",
        "ru_name": "Миха",
        "order": 20
    },
    "Nahum": {
        "chapters": 3,
        "section": "Nevi'im",
        "he_name": "נחום",
        "ru_name": "Нахум",
        "order": 21
    },
    "Habakkuk": {
        "chapters": 3,
        "section": "Nevi'im",
        "he_name": "חבקוק",
        "ru_name": "Хавакук",
        "order": 22
    },
    "Zephaniah": {
        "chapters": 3,
        "section": "Nevi'im",
        "he_name": "צפניה",
        "ru_name": "Цфанья",
        "order": 23
    },
    "Haggai": {
        "chapters": 2,
        "section": "Nevi'im",
        "he_name": "חגי",
        "ru_name": "Хагай",
        "order": 24
    },
    "Zechariah": {
        "chapters": 14,
        "section": "Nevi'im",
        "he_name": "זכריה",
        "ru_name": "Зхарья",
        "order": 25
    },
    "Malachi": {
        "chapters": 3,
        "section": "Nevi'im",
        "he_name": "מלאכי",
        "ru_name": "Малахи",
        "order": 26
    },
    
    # Ктувим (11 книг)
    "Psalms": {
        "chapters": 150,
        "section": "Ketuvim",
        "he_name": "תהלים",
        "ru_name": "Тегилим",
        "order": 27
    },
    "Proverbs": {
        "chapters": 31,
        "section": "Ketuvim",
        "he_name": "משלי",
        "ru_name": "Мишлей",
        "order": 28
    },
    "Job": {
        "chapters": 42,
        "section": "Ketuvim",
        "he_name": "איוב",
        "ru_name": "Иов",
        "order": 29
    },
    "Song of Songs": {
        "chapters": 8,
        "section": "Ketuvim",
        "he_name": "שיר השירים",
        "ru_name": "Шир Гашириим",
        "order": 30
    },
    "Ruth": {
        "chapters": 4,
        "section": "Ketuvim",
        "he_name": "רות",
        "ru_name": "Рут",
        "order": 31
    },
    "Lamentations": {
        "chapters": 5,
        "section": "Ketuvim",
        "he_name": "איכה",
        "ru_name": "Эйха",
        "order": 32
    },
    "Ecclesiastes": {
        "chapters": 12,
        "section": "Ketuvim",
        "he_name": "קהלת",
        "ru_name": "Когелет",
        "order": 33
    },
    "Esther": {
        "chapters": 10,
        "section": "Ketuvim",
        "he_name": "אסתר",
        "ru_name": "Эстер",
        "order": 34
    },
    "Daniel": {
        "chapters": 12,
        "section": "Ketuvim",
        "he_name": "דניאל",
        "ru_name": "Даниэль",
        "order": 35
    },
    "Ezra": {
        "chapters": 10,
        "section": "Ketuvim",
        "he_name": "עזרא",
        "ru_name": "Эзра",
        "order": 36
    },
    "Nehemiah": {
        "chapters": 13,
        "section": "Ketuvim",
        "he_name": "נחמיה",
        "ru_name": "Нехемья",
        "order": 37
    },
    "I Chronicles": {
        "chapters": 29,
        "section": "Ketuvim",
        "he_name": "דברי הימים א",
        "ru_name": "Диврей Гаямим I",
        "order": 38
    },
    "II Chronicles": {
        "chapters": 36,
        "section": "Ketuvim",
        "he_name": "דברי הימים ב",
        "ru_name": "Диврей Гаямим II",
        "order": 39
    }
}

# Разделы Танаха
TANAKH_SECTIONS = [
    {"name": "Torah", "he_name": "תורה", "ru_name": "Тора", "books": 5},
    {"name": "Nevi'im", "he_name": "נביאים", "ru_name": "Невиим", "books": 8},
    {"name": "Ketuvim", "he_name": "כתובים", "ru_name": "Ктувим", "books": 11}
]

def get_book_info(book_name: str) -> Dict[str, Any]:
    """Получить информацию о книге Танаха."""
    if book_name not in TANAKH_BOOKS:
        return {"ok": False, "error": f"Unknown book: {book_name}"}
    
    info = TANAKH_BOOKS[book_name]
    
    return {
        "ok": True,
        "corpus": "Танах",
        "corpus_en": "Tanakh",
        "section": info["section"],
        "book": book_name,
        "he_name": info["he_name"],
        "ru_name": info["ru_name"],
        "chapters": info["chapters"],
        "chapter_range": f"1-{info['chapters']}",
        "order": info["order"]
    }

def get_books_by_section(section_name: str) -> List[str]:
    """Получить список книг по разделу."""
    return [
        book for book, info in TANAKH_BOOKS.items()
        if info["section"] == section_name
    ]

def get_all_books() -> List[str]:
    """Получить список всех книг Танаха."""
    return list(TANAKH_BOOKS.keys())

def is_valid_chapter(book_name: str, chapter_num: int) -> bool:
    """Проверить, существует ли глава в книге."""
    if book_name not in TANAKH_BOOKS:
        return False
    
    return 1 <= chapter_num <= TANAKH_BOOKS[book_name]["chapters"]

def get_chapter_reference(book_name: str, chapter_num: int, verse_num: int = None) -> str:
    """Получить полную ссылку на главу или стих."""
    if verse_num:
        return f"{book_name} {chapter_num}:{verse_num}"
    return f"{book_name} {chapter_num}"

































