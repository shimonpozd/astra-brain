"""
Статичные данные для навигации по Талмуду Вавилонскому.
Данные основаны на стандартной структуре издания Вильна.
"""

from typing import Dict, Any, Tuple

# Структура Талмуда Вавилонского: трактат -> (начальная_страница, конечная_страница)
TALMUD_BAVLI_TRACTATES: Dict[str, Dict[str, Any]] = {
    # Седер Зраим
    "Berakhot": {
        "pages": (2, 64),
        "order": "Zeraim",
        "he_name": "ברכות",
        "ru_name": "Благословения"
    },
    
    # Седер Моэд  
    "Shabbat": {
        "pages": (2, 157),
        "order": "Moed", 
        "he_name": "שבת",
        "ru_name": "Суббота"
    },
    "Eruvin": {
        "pages": (2, 105),
        "order": "Moed",
        "he_name": "עירובין", 
        "ru_name": "Смешения"
    },
    "Pesachim": {
        "pages": (2, 121),
        "order": "Moed",
        "he_name": "פסחים",
        "ru_name": "Песах"
    },
    "Shekalim": {
        "pages": (2, 22),
        "order": "Moed",
        "he_name": "שקלים",
        "ru_name": "Шекели"
    },
    "Yoma": {
        "pages": (2, 88),
        "order": "Moed",
        "he_name": "יומא",
        "ru_name": "День [Искупления]"
    },
    "Sukkah": {
        "pages": (2, 56),
        "order": "Moed",
        "he_name": "סוכה",
        "ru_name": "Сукка"
    },
    "Beitzah": {
        "pages": (2, 40),
        "order": "Moed",
        "he_name": "ביצה",
        "ru_name": "Яйцо"
    },
    "Rosh Hashanah": {
        "pages": (2, 35),
        "order": "Moed",
        "he_name": "ראש השנה",
        "ru_name": "Новый год"
    },
    "Taanit": {
        "pages": (2, 31),
        "order": "Moed",
        "he_name": "תענית",
        "ru_name": "Пост"
    },
    "Megillah": {
        "pages": (2, 32),
        "order": "Moed",
        "he_name": "מגילה",
        "ru_name": "Свиток"
    },
    "Moed Katan": {
        "pages": (2, 29),
        "order": "Moed",
        "he_name": "מועד קטן",
        "ru_name": "Малый праздник"
    },
    "Chagigah": {
        "pages": (2, 27),
        "order": "Moed",
        "he_name": "חגיגה",
        "ru_name": "Праздничное жертвоприношение"
    },
    
    # Седер Нашим
    "Yevamot": {
        "pages": (2, 122),
        "order": "Nashim",
        "he_name": "יבמות",
        "ru_name": "Левиратный брак"
    },
    "Ketubot": {
        "pages": (2, 112),
        "order": "Nashim",
        "he_name": "כתובות",
        "ru_name": "Брачные договоры"
    },
    "Nedarim": {
        "pages": (2, 91),
        "order": "Nashim",
        "he_name": "נדרים",
        "ru_name": "Обеты"
    },
    "Nazir": {
        "pages": (2, 66),
        "order": "Nashim",
        "he_name": "נזיר",
        "ru_name": "Назорей"
    },
    "Sotah": {
        "pages": (2, 49),
        "order": "Nashim",
        "he_name": "סוטה",
        "ru_name": "Изменившая жена"
    },
    "Gittin": {
        "pages": (2, 90),
        "order": "Nashim",
        "he_name": "גיטין",
        "ru_name": "Разводы"
    },
    "Kiddushin": {
        "pages": (2, 82),
        "order": "Nashim",
        "he_name": "קידושין",
        "ru_name": "Обручение"
    },
    
    # Седер Незикин
    "Bava Kamma": {
        "pages": (2, 119),
        "order": "Nezikin",
        "he_name": "בבא קמא",
        "ru_name": "Первые врата"
    },
    "Bava Metzia": {
        "pages": (2, 119),
        "order": "Nezikin",
        "he_name": "בבא מציעא",
        "ru_name": "Средние врата"
    },
    "Bava Batra": {
        "pages": (2, 176),
        "order": "Nezikin",
        "he_name": "בבא בתרא",
        "ru_name": "Последние врата"
    },
    "Sanhedrin": {
        "pages": (2, 113),
        "order": "Nezikin",
        "he_name": "סנהדרין",
        "ru_name": "Синедрион"
    },
    "Makkot": {
        "pages": (2, 24),
        "order": "Nezikin",
        "he_name": "מכות",
        "ru_name": "Удары"
    },
    "Shevuot": {
        "pages": (2, 49),
        "order": "Nezikin",
        "he_name": "שבועות",
        "ru_name": "Клятвы"
    },
    "Eduyot": {
        "pages": (2, 8),
        "order": "Nezikin",
        "he_name": "עדויות",
        "ru_name": "Свидетельства"
    },
    "Avodah Zarah": {
        "pages": (2, 76),
        "order": "Nezikin",
        "he_name": "עבודה זרה",
        "ru_name": "Идолопоклонство"
    },
    "Avot": {
        "pages": (2, 6),
        "order": "Nezikin",
        "he_name": "אבות",
        "ru_name": "Отцы"
    },
    "Horayot": {
        "pages": (2, 14),
        "order": "Nezikin",
        "he_name": "הוריות",
        "ru_name": "Постановления"
    },
    
    # Седер Кодашим
    "Zevachim": {
        "pages": (2, 120),
        "order": "Kodashim",
        "he_name": "זבחים",
        "ru_name": "Жертвоприношения"
    },
    "Menachot": {
        "pages": (2, 110),
        "order": "Kodashim",
        "he_name": "מנחות",
        "ru_name": "Хлебные приношения"
    },
    "Hullin": {
        "pages": (2, 142),
        "order": "Kodashim",
        "he_name": "חולין",
        "ru_name": "Будничная пища"
    },
    "Bekhorot": {
        "pages": (2, 61),
        "order": "Kodashim",
        "he_name": "בכורות",
        "ru_name": "Первенцы"
    },
    "Arakhin": {
        "pages": (2, 34),
        "order": "Kodashim",
        "he_name": "ערכין",
        "ru_name": "Оценки"
    },
    "Temurah": {
        "pages": (2, 34),
        "order": "Kodashim",
        "he_name": "תמורה",
        "ru_name": "Замена"
    },
    "Keritot": {
        "pages": (2, 28),
        "order": "Kodashim",
        "he_name": "כריתות",
        "ru_name": "Отсечения"
    },
    "Meilah": {
        "pages": (2, 22),
        "order": "Kodashim",
        "he_name": "מעילה",
        "ru_name": "Использование святыни"
    },
    "Kinnim": {
        "pages": (2, 4),
        "order": "Kodashim",
        "he_name": "קינים",
        "ru_name": "Гнезда"
    },
    "Tamid": {
        "pages": (2, 10),
        "order": "Kodashim",
        "he_name": "תמיד",
        "ru_name": "Постоянное [жертвоприношение]"
    },
    "Midot": {
        "pages": (2, 4),
        "order": "Kodashim",
        "he_name": "מדות",
        "ru_name": "Размеры"
    },
    
    # Седер Тохорот
    "Kelim": {
        "pages": (2, 30),
        "order": "Tohorot",
        "he_name": "כלים",
        "ru_name": "Сосуды"
    },
    "Oholot": {
        "pages": (2, 18),
        "order": "Tohorot",
        "he_name": "אהלות",
        "ru_name": "Шатры"
    },
    "Negaim": {
        "pages": (2, 14),
        "order": "Tohorot",
        "he_name": "נגעים",
        "ru_name": "Проказа"
    },
    "Parah": {
        "pages": (2, 12),
        "order": "Tohorot",
        "he_name": "פרה",
        "ru_name": "Корова"
    },
    "Tohorot": {
        "pages": (2, 10),
        "order": "Tohorot",
        "he_name": "טהרות",
        "ru_name": "Чистоты"
    },
    "Mikvaot": {
        "pages": (2, 10),
        "order": "Tohorot",
        "he_name": "מקואות",
        "ru_name": "Бассейны"
    },
    "Niddah": {
        "pages": (2, 73),
        "order": "Tohorot",
        "he_name": "נדה",
        "ru_name": "Отлученная"
    },
    "Makhshirin": {
        "pages": (2, 6),
        "order": "Tohorot",
        "he_name": "מכשירין",
        "ru_name": "Подготавливающие"
    },
    "Zavim": {
        "pages": (2, 5),
        "order": "Tohorot",
        "he_name": "זבים",
        "ru_name": "Истечения"
    },
    "Tevul Yom": {
        "pages": (2, 4),
        "order": "Tohorot",
        "he_name": "טבול יום",
        "ru_name": "Окунувшийся днем"
    },
    "Yadayim": {
        "pages": (2, 4),
        "order": "Tohorot",
        "he_name": "ידים",
        "ru_name": "Руки"
    },
    "Uktzin": {
        "pages": (2, 3),
        "order": "Tohorot",
        "he_name": "עוקצין",
        "ru_name": "Стебли"
    }
}

# Порядок седеров
TALMUD_ORDERS = [
    {"name": "Zeraim", "he_name": "זרעים", "ru_name": "Семена"},
    {"name": "Moed", "he_name": "מועד", "ru_name": "Праздники"},
    {"name": "Nashim", "he_name": "נשים", "ru_name": "Женщины"},
    {"name": "Nezikin", "he_name": "נזיקין", "ru_name": "Ущербы"},
    {"name": "Kodashim", "he_name": "קדשים", "ru_name": "Святыни"},
    {"name": "Tohorot", "he_name": "טהרות", "ru_name": "Чистоты"}
]

def get_tractate_info(tractate_name: str) -> Dict[str, Any]:
    """Получить информацию о трактате."""
    if tractate_name not in TALMUD_BAVLI_TRACTATES:
        return {"ok": False, "error": f"Unknown tractate: {tractate_name}"}
    
    info = TALMUD_BAVLI_TRACTATES[tractate_name]
    start_page, end_page = info["pages"]
    
    return {
        "ok": True,
        "corpus": "Талмуд Вавилонский",
        "corpus_en": "Talmud Bavli",
        "order": info["order"],
        "tractate": tractate_name,
        "he_name": info["he_name"],
        "ru_name": info["ru_name"],
        "page_range": f"{start_page}a-{end_page}b",
        "total_pages": (end_page - start_page + 1) * 2,  # a и b стороны
        "first_page": start_page,
        "last_page": end_page,
        "pages_tuple": (start_page, end_page)
    }

def get_tractates_by_order(order_name: str) -> List[str]:
    """Получить список трактатов по седеру."""
    return [
        tractate for tractate, info in TALMUD_BAVLI_TRACTATES.items()
        if info["order"] == order_name
    ]

def get_all_tractates() -> List[str]:
    """Получить список всех трактатов."""
    return list(TALMUD_BAVLI_TRACTATES.keys())

def is_valid_page(tractate_name: str, page_num: int, amud: str = "a") -> bool:
    """Проверить, существует ли страница в трактате."""
    if tractate_name not in TALMUD_BAVLI_TRACTATES:
        return False
    
    start_page, end_page = TALMUD_BAVLI_TRACTATES[tractate_name]["pages"]
    return start_page <= page_num <= end_page and amud in ["a", "b"]

def get_page_reference(tractate_name: str, page_num: int, amud: str = "a") -> str:
    """Получить полную ссылку на страницу."""
    return f"{tractate_name} {page_num}{amud}"


































