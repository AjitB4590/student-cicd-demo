import joblib

# Load trained model
model = joblib.load("model.pkl")

print("Student Placement Prediction")

cgpa = float(input("Enter CGPA: "))
attendance = float(input("Enter Attendance: "))
coding_score = float(input("Enter Coding Score: "))
projects = int(input("Enter Number of Projects: "))
internship = int(input("Enter Internship (1=Yes, 0=No): "))

student = [[
    cgpa,
    attendance,
    coding_score,
    projects,
    internship
]]

prediction = model.predict(student)

if prediction[0] == 1:
    print("Prediction: Student is likely to be PLACED")
else:
    print("Prediction: Student is likely to be NOT PLACED")