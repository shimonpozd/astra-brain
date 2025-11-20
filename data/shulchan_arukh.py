"""
Статичные данные для навигации по Шулхан Аруху.
Структура 4 частей с симанами.
"""

from typing import Dict, Any, List

# Структура Шулхан Аруха: часть -> количество симанов
SHULCHAN_ARUKH_SECTIONS: Dict[str, Dict[str, Any]] = {
    "Orach Chaim": {
        "simanim": 697,
        "he_name": "אורח חיים",
        "ru_name": "Орах Хаим",
        "description": "Повседневные законы молитвы, благословений и праздников",
        "order": 1
    },
    "Yoreh De'ah": {
        "simanim": 403,
        "he_name": "יורה דעה", 
        "ru_name": "Йорэ Деа",
        "description": "Законы кашрута, траура, благотворительности",
        "order": 2
    },
    "Even HaEzer": {
        "simanim": 178,
        "he_name": "אבן העזר",
        "ru_name": "Эвен Гаэзер", 
        "description": "Законы брака, развода и семейных отношений",
        "order": 3
    },
    "Choshen Mishpat": {
        "simanim": 427,
        "he_name": "חושן משפט",
        "ru_name": "Хошен Мишпат",
        "description": "Гражданские и коммерческие законы",
        "order": 4
    }
}

# Популярные подразделы Орах Хаим (для более детальной навигации)
ORACH_CHAIM_TOPICS = {
    "Благословения": {"simanim": (1, 230), "description": "Законы благословений"},
    "Молитва": {"simanim": (89, 232), "description": "Законы молитвы"},
    "Шабат": {"simanim": (242, 344), "description": "Законы Шабата"},
    "Праздники": {"simanim": (345, 581), "description": "Законы праздников"},
    "Пост": {"simanim": (549, 581), "description": "Законы постов"},
    "Рош Хашана": {"simanim": (582, 603), "description": "Законы Рош Хашана"},
    "Йом Кипур": {"simanim": (604, 624), "description": "Законы Йом Кипур"},
    "Сукка": {"simanim": (625, 669), "description": "Законы Суккот"},
    "Лулав": {"simanim": (645, 669), "description": "Законы четырех видов"},
    "Ханука": {"simanim": (670, 684), "description": "Законы Хануки"},
    "Пурим": {"simanim": (685, 697), "description": "Законы Пурима"}
}

def get_section_info(section_name: str) -> Dict[str, Any]:
    """Получить информацию о части Шулхан Аруха."""
    if section_name not in SHULCHAN_ARUKH_SECTIONS:
        return {"ok": False, "error": f"Unknown section: {section_name}"}
    
    info = SHULCHAN_ARUKH_SECTIONS[section_name]
    
    return {
        "ok": True,
        "corpus": "Шулхан Арух",
        "corpus_en": "Shulchan Arukh",
        "section": section_name,
        "he_name": info["he_name"],
        "ru_name": info["ru_name"],
        "description": info["description"],
        "simanim": info["simanim"],
        "siman_range": f"1-{info['simanim']}",
        "order": info["order"]
    }

def get_all_sections() -> List[str]:
    """Получить список всех частей Шулхан Аруха."""
    return list(SHULCHAN_ARUKH_SECTIONS.keys())

def is_valid_siman(section_name: str, siman_num: int) -> bool:
    """Проверить, существует ли симан в части."""
    if section_name not in SHULCHAN_ARUKH_SECTIONS:
        return False
    
    return 1 <= siman_num <= SHULCHAN_ARUKH_SECTIONS[section_name]["simanim"]

def get_siman_reference(section_name: str, siman_num: int, seif_num: int = None) -> str:
    """Получить полную ссылку на симан или сеиф."""
    base_ref = f"Shulchan Arukh, {section_name} {siman_num}"
    if seif_num:
        return f"{base_ref}:{seif_num}"
    return base_ref

def get_orach_chaim_topic_info(topic_name: str) -> Dict[str, Any]:
    """Получить информацию о разделе Орах Хаим."""
    if topic_name not in ORACH_CHAIM_TOPICS:
        return {"ok": False, "error": f"Unknown topic: {topic_name}"}
    
    info = ORACH_CHAIM_TOPICS[topic_name]
    start_siman, end_siman = info["simanim"]
    
    return {
        "ok": True,
        "topic": topic_name,
        "description": info["description"],
        "siman_range": (start_siman, end_siman),
        "siman_count": end_siman - start_siman + 1,
        "first_siman": start_siman,
        "last_siman": end_siman
    }

def find_topic_by_siman(siman_num: int) -> str:
    """Найти раздел Орах Хаим по номеру симана."""
    for topic, info in ORACH_CHAIM_TOPICS.items():
        start_siman, end_siman = info["simanim"]
        if start_siman <= siman_num <= end_siman:
            return topic
    return "Неизвестный раздел"

































