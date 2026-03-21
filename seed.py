from app import app, db, bcrypt
from models import User, Course

with app.app_context():
    db.drop_all()
    db.create_all()

    # Create Faculty
    hashed_pw = bcrypt.generate_password_hash('password123').decode('utf-8')
    faculty = User(username='teacher1', password=hashed_pw, role='faculty')
    
    # Create Student
    student = User(username='student2', password=hashed_pw, role='student')
    
    db.session.add(faculty)
    db.session.add(student)
    db.session.commit()

    # Create a Course and enroll the student
    course = Course(name='Digital Signal Processing', code='DSP-101', faculty_id=faculty.id)
    db.session.add(course)
    db.session.commit()

    student.enrolled_courses.append(course)
    db.session.commit()

    print("✅ Database Seeded!")
    print("Login with: student1 / password123")