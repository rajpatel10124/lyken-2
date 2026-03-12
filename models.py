from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

# Helper table for Many-to-Many relationship
enrollments = db.Table('enrollments',
    db.Column('student_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('course_id', db.Integer, db.ForeignKey('course.id'), primary_key=True)
)

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(10), nullable=False) # 'admin', 'faculty', 'student'
    
    enrolled_courses = db.relationship('Course', secondary=enrollments, 
                                     backref=db.backref('students', lazy='dynamic'))

class Course(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    code = db.Column(db.String(20), unique=True, nullable=False)
    faculty_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    faculty = db.relationship('User', backref=db.backref('managed_courses', lazy=True))
    assignments = db.relationship('Assignment', backref='course', lazy=True)
    
class Assignment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    title = db.Column(db.String(100), nullable=False)
    instructions = db.Column(db.Text)
    deadline = db.Column(db.DateTime, nullable=False) 
    attempt_limit = db.Column(db.Integer, default=3)
    question_file = db.Column(db.String(255), nullable=True) 
    is_published = db.Column(db.Boolean, default=True) 
    
    submissions = db.relationship('Submission', backref='assignment', lazy=True)

class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    assignment_id = db.Column(db.Integer, db.ForeignKey('assignment.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    
    filename = db.Column(db.String(100))
    # CRITICAL: Stores the OCR/Extracted text for AI comparison
    text_content = db.Column(db.Text, nullable=True) 
    content_hash = db.Column(db.String(64), nullable=True) 
    
    score = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(20)) # 'accepted' or 'rejected'
    reason = db.Column(db.String(255)) 
    
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Standardized backref to 'author' to match logic in app.py
    author = db.relationship('User', backref=db.backref('submissions', lazy=True))
