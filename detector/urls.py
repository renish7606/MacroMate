from django.urls import path
from .views import reset_analysis
from .views import save_to_history
from .views import (
    index,
    get_food_suggestions_api,
    meal_history,
    confirm_food_and_log,
)

urlpatterns = [
    path("", index, name="index"),
    path("reset/", reset_analysis, name="reset_analysis"),
    path("history/", meal_history, name="meal_history"),
    path("save-history/", save_to_history, name="save_history"),
    path("api/confirm-food/", confirm_food_and_log),
    path("api/food-suggestions/", get_food_suggestions_api),
]