def grade_from_mark(mark):
    if mark >= 90:
        return 'A+'
    elif mark >= 80:
        return 'A'
    elif mark >= 70:
        return 'B'
    elif mark >= 60:
        return 'C'
    elif mark >= 50:
        return 'D'
    else:
        return 'F'

num_subjects = int(input("Enter the number of subjects: "))

subjects = []
marks = []

for i in range(num_subjects):
    while True:
        subject = input(f"Enter name of subject {i+1}: ")
        if subject.replace(' ', '').isalpha():
            break
        print("Invalid subject name! Use letters only.")
    mark = float(input(f"Enter marks for {subject}: "))
    subjects.append(subject)
    marks.append(mark)


results = []
for i in range(len(subjects)):
    subject = subjects[i]
    mark = marks[i]
    grade = grade_from_mark(mark)
    results.append({'Subject': subject, 'Marks': mark, 'Grade': grade})

average_mark = sum(marks) / len(marks)
overall_grade = grade_from_mark(average_mark)
passed_all_subjects = all(mark >= 50 for mark in marks)

for res in results:
    print(f"{res['Subject']}: {res['Marks']} marks, Grade: {res['Grade']}")
print(f"\nAverage Mark: {average_mark:.2f}")
print(f"Overall Grade: {overall_grade}")
print(f"Passed All Subjects: {'Yes' if passed_all_subjects else 'No'}")
