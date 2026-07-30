from django.shortcuts import render
import joblib
from preprocess import preprocess_text
import pandas as pd
from django.http import HttpResponse

# Load model only once
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")


def home(request):

    prediction = None
    confidence = None

    if request.method == "POST":

        # ---------- SINGLE REVIEW ----------
        if "review" in request.POST and request.POST.get("review").strip():

            review = request.POST.get("review")

            clean_review = preprocess_text(review)

            vector = tfidf.transform([clean_review])

            prediction = model.predict(vector)[0]

            probability = model.predict_proba(vector)

            confidence = min(round(max(probability[0]) * 100, 2), 99.99)

        # ---------- CSV UPLOAD ----------
        elif "csv_submit" in request.POST:

            csv_file = request.FILES.get("csv_file")

            if not csv_file:

                return render(request, "index.html", {
                    "error": "Please select a CSV file."
                })
            try:

                df = pd.read_csv(csv_file)

            except Exception:

                return render(request, "index.html", {
                    "error": "Invalid CSV file."
                })

            if "review" not in df.columns:

                return render(request, "index.html", {
                    "error": "CSV must contain a column named 'review'."
                })
            df["clean_review"] = df["review"].apply(preprocess_text)

            vectors = tfidf.transform(df["clean_review"])

            df["prediction"] = model.predict(vectors)

            df.drop(columns=["clean_review"], inplace=True)

            response = HttpResponse(content_type="text/csv")

            response["Content-Disposition"] = 'attachment; filename="predicted_reviews.csv"'

            df.to_csv(response, index=False)

            return response

    return render(request, "index.html", {
        "prediction": prediction,
        "confidence": confidence,
        "error":None
    })
