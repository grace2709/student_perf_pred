# models.py
from django.db import models
from django.contrib.auth.models import User
from django.db import models


class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    predicted_cgpa = models.FloatField()
    predicted_performance = models.CharField(max_length=100)
    classification = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.predicted_cgpa} - {self.predicted_performance} - {self.classification}"
# Create your models here.


class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)