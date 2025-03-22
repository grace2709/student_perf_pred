from django.shortcuts import render
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required

# views.py
from .models import Prediction


# views.py

from .models import Prediction

@login_required
def predictions_view(request):
    predictions = Prediction.objects.filter(user=request.user).order_by('-created_at')
    return render(request, "predictor/predictions.html", {"predictions": predictions})
def predict_cgpa(request):
    if request.method == "POST":
        try:
            # Retrieve input values from the form
            data = {
                "Peer_influence_on_student_perfromance": int(request.POST.get("Peer_influence_on_student_perfromance")),
                "time_management_skills": request.POST.get("time_management_skills"),
            }

            # Convert input to DataFrame
            new_data = pd.DataFrame([data])

            # Encode categorical features
            categorical_columns = ["Education_motivation_level", "Internet_access", "time_management_skills"]
            for col in categorical_columns:
                if col in new_data.columns:
                    if new_data[col].iloc[0] in encoders[col].classes_:
                        new_data[col] = encoders[col].transform(new_data[col].astype(str))
                    else:
                        most_frequent_label = encoders[col].classes_[0]
                        new_data[col] = encoders[col].transform([most_frequent_label])

            # Make prediction
            predicted_cgpa = gbr.predict(new_data)[0]
            if predicted_cgpa > 5:
                predicted_cgpa = random.choice([5.0, 4.8, 4.95, 4.9])

            # Determine classification
            classification = "First Class" if predicted_cgpa >= 4.5 else "Second Class Upper"

            # Save prediction to database
            Prediction.objects.create(
                user=request.user,
                predicted_cgpa=predicted_cgpa,
                predicted_performance="N/A",
                classification=classification
            )

            return render(request, "predictor/result.html", {"prediction": round(predicted_cgpa, 2), "classification": classification})

        except Exception as e:
            return render(request, "predictor/index.html", {"error": str(e)})

    return render(request, "predictor/index.html")

def predict_performance(request):
    if request.method == "POST":
        try:
            # Retrieve input values from the form
            data = {
                "current_cgpa": float(request.POST.get("current_cgpa", 0)),
                "100_level_cgpa": float(request.POST.get("100_level_cgpa", 0)),
                "Assignment_performance": int(request.POST.get("Assignment_performance", 0)),
                "classAttendance": int(request.POST.get("classAttendance", 0)),
                "study_hours": request.POST.get("study_hours", "20"),  # Default to "20" if missing
                "Education_motivation_level": request.POST.get("Education_motivation_level", "Medium"),
                "class_participation": int(request.POST.get("class_participation", 0)),
                "Internet_access": request.POST.get("Internet_access", "Yes"),
                "Peer_influence_on_student_perfromance": int(request.POST.get("Peer_influence_on_student_perfromance", 0)),
                "time_management_skills": request.POST.get("time_management_skills", "Average"),
            }

            # Convert input data to DataFrame
            new_data = pd.DataFrame([data])

            # Capitalize categorical values for encoding consistency
            new_data["Education_motivation_level"] = new_data["Education_motivation_level"].str.capitalize()

            # Encode categorical data
            for col in rate_encoders.keys():  # Use saved encoder keys
                if col in new_data.columns:
                    try:
                        new_data[col + "_encoder"] = rate_encoders[col].transform(new_data[col].astype(str))
                    except ValueError:
                        # If unseen value appears, assign a default encoded value
                        new_data[col + "_encoder"] = rate_encoders[col].transform(["Medium"])[0]

            # Drop original categorical columns to match training format
            new_data_processed = new_data.drop(columns=rate_encoders.keys(), errors="ignore")

            # Make prediction
            predicted_performance = model.predict(new_data_processed)[0]  # Directly use model output

            if predicted_performance == "Medium":
                predicted_performance = "Average"

            # Determine classification
            classification = "First Class" if data["current_cgpa"] >= 4.5 else "Second Class Upper"

            # Save prediction to database
            Prediction.objects.create(
                user=request.user,
                predicted_cgpa=data["current_cgpa"],
                predicted_performance=predicted_performance,
                classification=classification
            )

            # Render the result template
            return render(request, "predictor/result.html", {"prediction": predicted_performance, "classification": classification})

        except Exception as e:
            return render(request, "predictor/result.html", {"error": f"Error: {str(e)}"})

    return render(request, "predictor/predict.html")

def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("landing_page")
        else:
            return render(request, "predictor/login.html", {"error": "Invalid username or password"})
    return render(request, "predictor/login.html")

@login_required
def landing_page(request):
    return render(request, "predictor/landing_page.html")

from django.urls import path
from .views import login_view, landing_page

from django.contrib.auth import logout
from django.shortcuts import redirect

def logout_view(request):
    logout(request)
    return redirect('login')


# Create your views here.
import pickle
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
import random
import numpy as np

# Load the trained model and encoders
with open("gb_student_performance_model.pkl", "rb") as model_file:
    gbr = pickle.load(model_file)

with open("gb_label_encoders.pkl", "rb") as encoder_file:
    encoders = pickle.load(encoder_file)

# Load the trained model and encoders
with open("re_performance_model.pkl", "rb") as rate_model_file:
    model = pickle.load(rate_model_file)

with open("re_encoders.pkl", "rb") as rate_encoder_file:
    rate_encoders = pickle.load(rate_encoder_file)

# Define categorical columns for encoding
label_encoder_df = ["Education_motivation_level", "Internet_access", "time_management_skills"]

def predict_cgpa(request):
    if request.method == "POST":
        try:
            # Get user input
            data = {
                "100_level_cgpa": float(request.POST.get("100_level_cgpa")),
                "Assignment_performance": int(request.POST.get("Assignment_performance")),
                "classAttendance": int(request.POST.get("classAttendance")),
                "study_hours": int(request.POST.get("study_hours")),
             
                "Education_motivation_level": request.POST.get("Education_motivation_level"),
             
                "class_participation": int(request.POST.get("class_participation")),
                "Internet_access": request.POST.get("Internet_access"),
             
                "Peer_influence_on_student_perfromance": int(request.POST.get("Peer_influence_on_student_perfromance")),
                "time_management_skills": request.POST.get("time_management_skills"),
               
                
            }

            # Convert input to DataFrame
            new_data = pd.DataFrame([data])

           

            # Encode categorical features
            categorical_columns = ["Education_motivation_level", "Internet_access", "time_management_skills"]
            for col in categorical_columns:
                if col in new_data.columns:
                    if new_data[col].iloc[0] in encoders[col].classes_:
                        new_data[col] = encoders[col].transform(new_data[col].astype(str))
                    else:
                        most_frequent_label = encoders[col].classes_[0]
                        new_data[col] = encoders[col].transform([most_frequent_label])

            # Make prediction
            predicted_cgpa = gbr.predict(new_data)[0]
            if predicted_cgpa > 5:
                predicted_cgpa = random.choice([5.0, 4.8, 4.95, 4.9])

            return render(request, "predictor/result.html", {"prediction": round(predicted_cgpa, 2)})

        except Exception as e:
            return render(request, "predictor/index.html", {"error": str(e)})

    return render(request, "predictor/index.html")

def result(request):
    return render("predictor/result.html")


def predict_performance(request):
    if request.method == "POST":
        try:
            # Retrieve input values from the form
            data = {
                "current_cgpa": float(request.POST.get("current_cgpa", 0)),
                "100_level_cgpa": float(request.POST.get("100_level_cgpa", 0)),
                "Assignment_performance": int(request.POST.get("Assignment_performance", 0)),
                "classAttendance": int(request.POST.get("classAttendance", 0)),
                "study_hours": request.POST.get("study_hours", "20"),  # Default to "20" if missing
                "Education_motivation_level": request.POST.get("Education_motivation_level", "Medium"),
                "class_participation": int(request.POST.get("class_participation", 0)),
                "Internet_access": request.POST.get("Internet_access", "Yes"),
                "Peer_influence_on_student_perfromance": int(request.POST.get("Peer_influence_on_student_perfromance", 0)),
                "time_management_skills": request.POST.get("time_management_skills", "Average"),
            }

            # Convert input data to DataFrame
            new_data = pd.DataFrame([data])

            # Capitalize categorical values for encoding consistency
            new_data["Education_motivation_level"] = new_data["Education_motivation_level"].str.capitalize()

            # Encode categorical data
            for col in rate_encoders.keys():  # Use saved encoder keys
                if col in new_data.columns:
                    try:
                        new_data[col + "_encoder"] = rate_encoders[col].transform(new_data[col].astype(str))
                    except ValueError:
                        # If unseen value appears, assign a default encoded value
                        new_data[col + "_encoder"] = rate_encoders[col].transform(["Medium"])[0]

            # Drop original categorical columns to match training format
            new_data_processed = new_data.drop(columns=rate_encoders.keys(), errors="ignore")

            # Make prediction
            predicted_performance = model.predict(new_data_processed)[0]  # Directly use model output

            if predicted_performance == "Medium":
                predicted_performance = "Average"

            # Debugging output
            print(f"Predicted Performance: {predicted_performance}")

            # Render the result template
            return render(request, "predictor/result.html", {"prediction": predicted_performance})

        except Exception as e:
            return render(request, "predictor/result.html", {"error": f"Error: {str(e)}"})

    return render(request, "predictor/predict.html")


def student_aid(request):
    return render(request, "predictor/student_aid.html")




import pandas as pd
import numpy as np
import joblib
from django.shortcuts import render, redirect
from django.conf import settings
from .forms import UploadFileForm
from .models import UploadedFile
from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy.stats

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = UploadedFile(file=request.FILES['file'])
            uploaded_file.save()
            return redirect('predict', file_id=uploaded_file.id)
    else:
        form = UploadFileForm()
    return render(request, 'predictor/upload.html', {'form': form})

def predict(request, file_id):
    uploaded_file = UploadedFile.objects.get(id=file_id)
    file_path = uploaded_file.file.path

    # Load the uploaded file
    data = pd.read_csv(file_path)

    # Load the pre-trained model
    model = joblib.load(settings.MODEL_PATH)

    # Make predictions
    X = data.drop(columns=['Matriculation Number', 'Actual CGPA'], errors='ignore')
    predictions = model.predict(X)

    # Calculate error metrics
    if 'Actual CGPA' in data.columns:
        actual = data['Actual CGPA']
        mae = mean_absolute_error(actual, predictions)
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        spearman_corr, _ = scipy.stats.spearmanr(actual, predictions)
    else:
        mae = mse = rmse = spearman_corr = None

    # Add predictions to the data
    data['Predicted CGPA'] = predictions
    data['Predicted Class of Degree'] = ['First Class' if x >= 4.5 else 'Second Class Upper' if x >= 3.5 else 'Second Class Lower' if x >= 2.5 else 'Third Class' for x in predictions]

    context = {
        'data': data.to_dict(orient='records'),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'spearman_corr': spearman_corr,
    }
    return render(request, 'predictor/results.html', context)