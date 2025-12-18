from django.db import models
from django.db.models import Sum
from django.utils import timezone
from django.contrib.auth.models import User

class MealLog(models.Model):
    """Model to store meal/food entries for calorie tracking"""
    
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="meal_logs",
        help_text="Owner of this meal entry"
    )
    food_name = models.CharField(max_length=200, help_text="Name of the food item")
    calories = models.IntegerField(help_text="Calories for this food entry")
    created_at = models.DateTimeField(auto_now_add=True, help_text="When this meal was logged")
    
    class Meta:
        ordering = ['-created_at']  # Most recent first
        verbose_name = "Meal Log Entry"
        verbose_name_plural = "Meal Log Entries"
    
    def __str__(self):
        created_local = timezone.localtime(self.created_at)
        return f"{self.food_name} - {self.calories} kcal ({created_local.strftime('%Y-%m-%d %H:%M')})"
    
    @classmethod
    def get_todays_meals(cls):
        """Get all meals logged today"""
        today = timezone.localdate()
        return cls.objects.filter(created_at__date=today)
    
    @classmethod
    def get_todays_total_calories(cls, user=None):
        """Calculate total calories consumed today for a specific user"""
        qs = cls.get_todays_meals()
        if user:
            qs = qs.filter(user=user)
        total = qs.aggregate(total=Sum('calories'))['total']
        return total or 0
    
    def get_time_display(self):
        """Get formatted time for display"""
        created_local = timezone.localtime(self.created_at)
        return created_local.strftime('%I:%M %p')
    
    def get_date_display(self):
        """Get formatted date for display"""
        created_local = timezone.localtime(self.created_at)
        return created_local.strftime('%B %d, %Y')

class FoodHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    food = models.CharField(max_length=100)
    calories = models.FloatField()
    image = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.food} - {self.calories} kcal"