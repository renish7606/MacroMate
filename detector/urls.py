# from django.urls import path
# from .views import reset_analysis
# from .views import save_to_history
# from .views import download_history_pdf
# from .views import (
#     index,
#     get_food_suggestions_api,
#     meal_history,
#     confirm_food_and_log,
# )

# urlpatterns = [
#     path("", index, name="index"),
#     path("reset/", reset_analysis, name="reset_analysis"),
#     path("history/", meal_history, name="meal_history"),
#     path("save-history/", save_to_history, name="save_history"),
#     path("api/confirm-food/", views.confirm_food_and_log, name="confirm_food"),
#     path("api/food-suggestions/", get_food_suggestions_api),
#     path("history/download/", download_history_pdf, name="download_history_pdf"),
# ]

from django.urls import path
from .views import (
    index,
    reset_analysis,
    save_to_history,
    download_history_pdf,
    get_food_suggestions_api,
    meal_history,
    confirm_food_and_log,
    parse_portion_api,
)

urlpatterns = [
    path("", index, name="index"),
    path("reset/", reset_analysis, name="reset_analysis"),
    path("history/", meal_history, name="meal_history"),
    path("save-history/", save_to_history, name="save_to_history"),
    path("api/confirm-food/", confirm_food_and_log, name="confirm_food"),
    path("api/food-suggestions/", get_food_suggestions_api),
    path("history/download/", download_history_pdf, name="download_history_pdf"),
    path("api/parse-portion/", parse_portion_api , name="parse_portion_api"),
]
