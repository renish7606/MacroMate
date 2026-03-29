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
    dashboard,
    upload_food,
    analysis,
    assistant,
    assistant_chat,
    clear_assistant_history,
    profile,
    reset_analysis,
    save_to_history,
    download_history_pdf,
    download_today_pdf,
    get_food_suggestions_api,
    meal_history,
    delete_history_item,
    confirm_food_and_log,
    parse_portion_api,
    manual_food_lookup,
)

urlpatterns = [
    path("", dashboard, name="dashboard"),
    path("upload/", upload_food, name="upload_food"),
    path("analysis/", analysis, name="analysis"),
    path("assistant/", assistant, name="assistant"),
    path("assistant/chat/", assistant_chat, name="assistant_chat"),
    path("assistant/clear-history/", clear_assistant_history, name="clear_assistant_history"),
    path("profile/", profile, name="profile"),
    path("reset/", reset_analysis, name="reset_analysis"),
    path("history/", meal_history, name="meal_history"),
    path("history/delete/<int:meal_id>/", delete_history_item, name="delete_history_item"),
    path("save-history/", save_to_history, name="save_to_history"),
    path("api/confirm-food/", confirm_food_and_log, name="confirm_food"),
    path("api/food-suggestions/", get_food_suggestions_api),
    path("history/download/", download_history_pdf, name="download_history_pdf"),
    path("history/today-pdf/", download_today_pdf, name="download_today_pdf"),
    path("api/parse-portion/", parse_portion_api, name="parse_portion_api"),
    path("manual-food/", manual_food_lookup, name="manual_food_lookup"),
]
