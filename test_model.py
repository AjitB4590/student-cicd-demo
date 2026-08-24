import joblib


def test_model_file_exists():
    model = joblib.load("model.pkl")
    assert model is not None


def test_model_prediction():
    model = joblib.load("model.pkl")

    student = [[
        8.5,
        90,
        85,
        3,
        1
    ]]

    prediction = model.predict(student)

    assert prediction[0] in [0, 1]