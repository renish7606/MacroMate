from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import FoodHistory, UserProfile, Exercise, MealLog

admin.site.register(FoodHistory)
admin.site.register(UserProfile)
admin.site.register(Exercise)
admin.site.register(MealLog)