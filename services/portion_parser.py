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
    "gm": 1,
    "gms": 1,
    "gram": 1,
    "grams": 1,
    "ml": 1,
    "kg": 1000,
}

# Food item piece-size estimates (grams per piece/unit)
FOOD_ITEM_GRAMS = {
    "samosa": 70,
    "samosas": 70,
    "egg": 60,
    "eggs": 60,
    "burger": 250,
    "burgers": 250,
    "pizza": 900,
    "slice": 110,
    "scoop": 70,
    "vadapav": 140,
    "vada_pav": 140,
    "vada": 140,
    "pav": 140,
    "cheesecake": 125,
}

TEXT_NUMBERS = {
    "half": 0.5,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _singular(token: str) -> str:
    if token.endswith("es") and len(token) > 3:
        return token[:-2]
    if token.endswith("s") and len(token) > 2:
        return token[:-1]
    return token


def parse_quantity(text: str, food_name: str) -> int:
    """
    Convert natural language portion text into grams
    """
    text = text.lower().strip()
    normalized_food = (food_name or "").lower().strip()

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

    # 3️⃣ Find explicit unit via word boundaries
    tokens = re.findall(r"[a-zA-Z]+", text)
    for token in tokens:
        if token in UNIT_GRAMS:
            unit = token
            break

    # 4️⃣ Calculate grams
    if unit:
        grams = quantity * UNIT_GRAMS[unit]
    elif normalized_food and normalized_food.replace(" ", "_") in ("ice_cream", "ice cream"):
        # Natural text often says "2 ice cream"; treat as scoops for this food.
        grams = quantity * 70
    elif normalized_food:
        # If user writes "i ate 4 samosa" (count + food name), map as pieces.
        food_tokens = [normalized_food, normalized_food.replace("_", " ")]
        matched_piece_unit = None
        for ft in food_tokens:
            for word in ft.split():
                sw = _singular(word)
                if word in FOOD_ITEM_GRAMS:
                    matched_piece_unit = word
                    break
                if sw in FOOD_ITEM_GRAMS:
                    matched_piece_unit = sw
                    break
            if matched_piece_unit:
                break

        if matched_piece_unit and (
            re.search(rf"\b{re.escape(matched_piece_unit)}\b", text)
            or re.search(rf"\b{re.escape(matched_piece_unit)}s\b", text)
        ):
            grams = quantity * FOOD_ITEM_GRAMS[matched_piece_unit]
        else:
            # Fallback for count-based phrasing (e.g. "I ate 3 cheesecake").
            grams = quantity * 100
    else:
        grams = quantity * 100

    return int(grams)
