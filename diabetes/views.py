import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Information
from .serializers import InformationSerializer
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


class InfoListAPI(APIView):
    def get(self, request):
        queryset = Information.objects.all()
        print(queryset)
        serializer = InformationSerializer(queryset, many=True)
        return Response(serializer.data)


@api_view(["POST", "GET"])
@csrf_exempt
def index(request):
    if request.method == "POST":
        serializer = InformationSerializer(data=request.data)
        if serializer.is_valid():
            glucose = serializer.validated_data['glucose']
            age = serializer.validated_data['age']
            bmi = float(serializer.validated_data['bmi'])
            bloodPressure = serializer.validated_data['bloodPressure']
            pregnancies = int(serializer.validated_data['pregnancies'])
            weight = int(serializer.validated_data['weight'])
            height = int(serializer.validated_data['height'])
            skinThickness = serializer.validated_data['skinThickness']
            insulin = serializer.validated_data['insulin']
            diabetesPedigreeFn = serializer.validated_data['diabetesPedigreeFn']

            diabetes_dataset = pd.read_csv(r"/Users/abdulaziz/Coding/diabetes/diabetes.csv")
            diabetes_dataset.head()
            diabetes_dataset.shape
            diabetes_dataset.describe()

            diabetes_dataset['Outcome'].value_counts()

            diabetes_dataset.groupby('Outcome').mean()

            X = diabetes_dataset.drop(columns='Outcome', axis=1)
            Y = diabetes_dataset['Outcome']

            scaler = StandardScaler()

            scaler.fit(X)

            standardized_data = scaler.transform(X)

            X = standardized_data
            Y = diabetes_dataset['Outcome']

            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.2, stratify=Y, random_state=2)

            classifier = svm.SVC(kernel='linear')

            classifier.fit(X_train, Y_train)

            X_train_prediction = classifier.predict(X_train)
            training_data_accuracy = accuracy_score(
                X_train_prediction, Y_train)

            print('Accuracy score of the training data : ',
                  training_data_accuracy)

            # accuracy score on the test data
            X_test_prediction = classifier.predict(X_test)
            test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

            print('Accuracy score of the test data : ', test_data_accuracy)

            # input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
            input_data = (pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFn, age)
            print(input_data)
            input_data_as_numpy_array = np.asarray(input_data)

            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

            std_data = scaler.transform(input_data_reshaped)

            prediction = classifier.predict(std_data)
            result = ''

            if (prediction[0] == 0):
                result = 'Non Diabetic'
            else:
                result = 'Diabetic'

            print(result)
            return Response({'status': 'success', 'result': result})
        else:
            return Response(serializer.errors)
    else:
        return render(request, 'index.html')
