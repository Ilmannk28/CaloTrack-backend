import random

CATEGORY_PRIORITY = {
    "carbohydrate": 0,
    "protein": 1,
    "vegetable": 2,
    "fruit": 3,
    "dairy": 4,
    "fat": 5,
    "beverage": 6,
    "snack": 7
}

def find_meal_combinations(food_items, calorie_target, limit=30):
    results = []
    n = len(food_items)
    items = food_items.copy()
    random.shuffle(items)

    def backtrack(start, current_combo, current_cal, used_categories):
        if len(results) >= limit:
            return
        if calorie_target * 0.9 <= current_cal <= calorie_target:
            results.append(list(current_combo))
        if current_cal >= calorie_target or start == n:
            return
        for i in range(start, n):
            name, cal, category = items[i]
            if category in used_categories:
                continue
            if current_cal + cal <= calorie_target:
                current_combo.append((name, 1, cal, category))
                used_categories.add(category)
                backtrack(i + 1, current_combo, current_cal + cal, used_categories)
                current_combo.pop()
                used_categories.remove(category)

    backtrack(0, [], 0, set())
    return results if results else [[]]

def select_random_meal(food_items, calorie_target):
    combos = find_meal_combinations(food_items, calorie_target)
    return random.choice(combos) if combos else []

def generate_meal_plan(calorie_goal: int):
    foods = {
        "breakfast": [
            ("Roti Gandum", 200, "carbohydrate"),
            ("Telur Rebus", 150, "protein"),
            ("Yogurt", 150, "dairy"),
            ("Pisang", 100, "fruit"),
            ("Oatmeal", 250, "carbohydrate"),
            ("Smoothie Buah", 200, "fruit"),
            ("Keju Cottage", 120, "dairy"),
            ("Granola Bar", 180, "snack"),
            ("Sereal Gandum", 220, "carbohydrate"),
            ("Avocado Toast", 300, "fat"),
            ("Toast dengan Selai Kacang", 280, "fat"),
            ("Kopi Susu", 90, "beverage"),
            ("Buah Stroberi", 50, "fruit"),
            ("Susu Almond", 40, "dairy")
        ],
        "lunch": [
            ("Nasi Putih", 400, "carbohydrate"),
            ("Ayam Panggang", 350, "protein"),
            ("Sayur Bayam", 100, "vegetable"),
            ("Buah Apel", 80, "fruit"),
            ("Ikan Bakar", 300, "protein"),
            ("Tahu Tempe", 250, "protein"),
            ("Salad Sayuran", 150, "vegetable"),
            ("Pasta dengan Saus Tomat", 400, "carbohydrate"),
            ("Quinoa Salad", 350, "carbohydrate"),
            ("Sushi (3 potong)", 200, "carbohydrate"),
            ("Rendang Daging", 450, "protein"),
            ("Sop Ayam", 280, "protein"),
            ("Kentang Rebus", 150, "carbohydrate"),
            ("Sayur Lodeh", 180, "vegetable")
        ],
        "dinner": [
            ("Nasi Merah", 350, "carbohydrate"),
            ("Ikan Panggang", 300, "protein"),
            ("Tumis Brokoli", 120, "vegetable"),
            ("Buah Jeruk", 70, "fruit"),
            ("Daging Sapi Panggang", 400, "protein"),
            ("Sup Ayam", 250, "protein"),
            ("Sayur Campur", 150, "vegetable"),
            ("Pizza Sayuran", 300, "carbohydrate"),
            ("Pasta dengan Saus Pesto", 450, "carbohydrate"),
            ("Kari Ayam", 350, "protein"),
            ("Salad Kacang Merah", 200, "vegetable"),
            ("Vegetable Stir Fry", 220, "vegetable"),
            ("Kroket Kentang", 180, "carbohydrate"),
            ("Tahu Goreng", 220, "protein")
        ],
        "snack": [
            ("Kacang Almond (25gr)", 150, "snack"),
            ("Biskuit Gandum", 120, "snack"),
            ("Smoothie Buah", 180, "snack"),
            ("Coklat Hitam", 200, "snack"),
            ("Popcorn (tanpa mentega)", 100, "snack"),
            ("Buah Kering", 150, "snack"),
            ("Yogurt dengan Madu", 200, "snack"),
            ("Keripik Sayuran", 130, "snack"),
            ("Granola", 200, "snack"),
            ("Kue Beras", 90, "snack"),
            ("Edamame", 160, "snack"),
            ("Juice Jeruk", 110, "beverage"),
            ("Crispy Chickpeas", 170, "snack"),
            ("Energy Bar", 210, "snack")
        ]
    }

    breakfast_target = int(calorie_goal * 0.25)
    lunch_target = int(calorie_goal * 0.35)
    dinner_target = int(calorie_goal * 0.30)
    snack_target = calorie_goal - (breakfast_target + lunch_target + dinner_target)

    breakfast_items = select_random_meal(foods["breakfast"], breakfast_target)
    lunch_items = select_random_meal(foods["lunch"], lunch_target)
    dinner_items = select_random_meal(foods["dinner"], dinner_target)
    snack_items = select_random_meal(foods["snack"], max(0, snack_target))

    result = {
        "breakfast": [{"name": i[0], "calories": i[2], "category": i[3]} for i in breakfast_items],
        "lunch": [{"name": i[0], "calories": i[2], "category": i[3]} for i in lunch_items],
        "dinner": [{"name": i[0], "calories": i[2], "category": i[3]} for i in dinner_items],
        "snack": [{"name": i[0], "calories": i[2], "category": i[3]} for i in snack_items],
    }

    result["total_calories"] = sum(i["calories"] for meals in result.values() if isinstance(meals, list) for i in meals)
    return result
