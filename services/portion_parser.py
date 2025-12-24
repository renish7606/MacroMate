import re

# Unit → grams mapping
UNIT_GRAMS = {
    # Pizza
    "slice": 110,
    "slices": 110,
    "pizza": 900,

    # Ice cream
    "scoop": 70,
    "scoops": 70,

    # Rice
    "bowl": 180,
    "bowls": 180,

    # Bread
    "slice bread": 30,

    # Eggs
    "egg": 60,
    "eggs": 60,

    # Weight units
    "g": 1,
    "gram": 1,
    "grams": 1,
    "kg": 1000,
}
    

TEXT_NUMBERS = {
    "half": 0.5,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
}


def parse_quantity(text: str, food_name: str) -> int:
    """
    Convert natural language portion text into grams
    """
    text = text.lower().strip()

    quantity = None
    unit = None

    # 1️⃣ Find numeric quantity (e.g. 1.5, 2)
    num_match = re.search(r"\d+(\.\d+)?", text)
    if num_match:
        quantity = float(num_match.group())

    # 2️⃣ Find textual quantity (half, one, two)
    if quantity is None:
        for word, value in TEXT_NUMBERS.items():
            if word in text:
                quantity = value
                break

    # Default quantity
    if quantity is None:
        quantity = 1

    # 3️⃣ Find unit
    for u in UNIT_GRAMS:
        if u in text:
            unit = u
            break

    # 4️⃣ Calculate grams
    if unit:
        grams = quantity * UNIT_GRAMS[unit]
    else:
        grams = 100  # safe fallback

    return int(grams)
