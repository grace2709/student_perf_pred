from django.urls import path
from .views import *  # Import the view function

urlpatterns = [
    path("login/", login_view, name="login"),
    path("landing_page/", landing_page, name="landing_page"),
    path("predict_cgpa/", predict_cgpa, name="predict_cgpa"),
    path("result/", result, name="result"),
    path("predict/", predict_performance, name="predict"),
    path("student_aid/", student_aid, name="student_aid"),
    path("logout/", logout_view, name="logout"),
    path('predictions/', predictions_view, name='predictions'),
path('predict/<int:file_id>/', predict, name='predict'),
]
