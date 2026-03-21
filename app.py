from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from models import db, User, Course, Submission, Assignment
import logic  # Importing your FAISS version
import datetime
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-key-123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///university.db'
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static/uploads')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(id): 
    return db.session.get(User, int(id))

# --- HELPER: INITIALIZE FAISS ON STARTUP ---
def sync_vector_engine():
    """Loads all existing text from DB into the FAISS index."""
    with app.app_context():
        # Get all submissions that have extracted text
        all_subs = Submission.query.all()
        texts = [s.text_content for s in all_subs if s.text_content]
        if texts:
            logic.build_index(texts)
            print(f"FAISS Index synchronized with {len(texts)} documents.")

# --- AUTH ROUTES (Unchanged) ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        role = request.form.get('role')
        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('signup'))
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, password=hashed_password, role=role)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and bcrypt.check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

# --- DASHBOARD & COURSE MGMT ---

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'faculty':
        courses = Course.query.filter_by(faculty_id=current_user.id).all()
        return render_template('dashboard.html', courses=courses)
    else:
        enrolled_courses = current_user.enrolled_courses 
        all_available = Course.query.all()
        return render_template('dashboard.html', 
                               courses=enrolled_courses, 
                               all_available_courses=all_available)

@app.route('/create_course', methods=['GET', 'POST'])
@login_required
def create_course():
    if current_user.role != 'faculty':
        flash("Only faculty can create courses.", "danger")
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        name = request.form.get('name')
        code = request.form.get('code')
        new_course = Course(name=name, code=code, faculty_id=current_user.id)
        db.session.add(new_course)
        db.session.commit()
        flash(f"Course '{name}' created successfully!", "success")
        return redirect(url_for('dashboard'))
    return render_template('create_course.html')

@app.route('/enroll/<int:course_id>')
@login_required
def enroll(course_id):
    course = db.get_or_404(Course, course_id)
    if course not in current_user.enrolled_courses:
        current_user.enrolled_courses.append(course)
        db.session.commit()
        flash(f"Successfully enrolled in {course.name}!", "success")
    else:
        flash("You are already enrolled in this course.", "info")
    return redirect(url_for('dashboard'))

# --- ASSIGNMENT & SUBMISSION MGMT ---

@app.route('/course/<int:course_id>/create_assignment', methods=['GET', 'POST'])
@login_required
def create_assignment(course_id):
    if current_user.role != 'faculty':
        return redirect(url_for('dashboard'))
    
    course = Course.query.get_or_404(course_id)
    
    if request.method == 'POST':
        title = request.form.get('title')
        instructions = request.form.get('instructions')
        deadline_str = request.form.get('deadline')
        
        try:
            deadline = datetime.datetime.strptime(deadline_str, '%Y-%m-%dT%H:%M')
        except ValueError:
            flash("Invalid date format.", "danger")
            return render_template('create_assignment.html', course=course)

        files = request.files.getlist('question_files')
        filenames = []
        
        for file in files:
            if file and file.filename != '':
                timestamp = int(datetime.datetime.now().timestamp())
                secure_name = f"Q_{course_id}_{timestamp}_{file.filename}"
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_name))
                filenames.append(secure_name)

        new_assign = Assignment(
            title=title,
            deadline=deadline,
            instructions=instructions,
            course_id=course.id,
            question_file=",".join(filenames) if filenames else None, 
            attempt_limit=int(request.form.get('attempt_limit', 3)),
            is_published=True
        )
        db.session.add(new_assign)
        db.session.commit()
        
        flash(f"Assignment created with {len(filenames)} resource files!", "success")
        return redirect(url_for('dashboard'))
        
    return render_template('create_assignment.html', course=course)
@app.route('/edit_assignment/<int:assignment_id>', methods=['GET', 'POST'])
@login_required
def edit_assignment(assignment_id):
    assignment = Assignment.query.get_or_404(assignment_id)
    if current_user.role != 'faculty' or assignment.course.faculty_id != current_user.id:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        assignment.title = request.form.get('title')
        assignment.instructions = request.form.get('instructions')
        db.session.commit()
        flash("Assignment updated!", "success")
        return redirect(url_for('course_page', course_id=assignment.course_id))
    return render_template('edit_assignment.html', assignment=assignment)

@app.route('/course/<int:course_id>')
@login_required
def course_page(course_id):
    course = db.get_or_404(Course, course_id)
    assignments = Assignment.query.filter_by(course_id=course_id).order_by(Assignment.deadline.asc()).all()
    
    return render_template('course_page.html', 
                           course=course, 
                           assignments=assignments, 
                           now=datetime.datetime.now(),
                           Submission=Submission)

# --- THE UPDATED SUBMISSION ROUTE ---

@app.route('/submit/<int:assignment_id>', methods=['GET', 'POST'])
@login_required
def submit(assignment_id):
    assignment = db.get_or_404(Assignment, assignment_id)
    attempts_made = Submission.query.filter_by(user_id=current_user.id, assignment_id=assignment_id).count()

    if attempts_made >= assignment.attempt_limit:
        flash(f"No attempts remaining. (Limit: {assignment.attempt_limit})", "danger")
        return redirect(url_for('course_page', course_id=assignment.course_id))

    if request.method == 'POST':
        file = request.files.get('file') 
        if file:
            filename = f"S_{assignment_id}_{current_user.id}_{int(datetime.datetime.now().timestamp())}_{file.filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # 1. Extract Text and Hash using advanced Logic
            # Note: logic.extract_text handles OCR and PDF automatically now
            extracted_text, binary_content, file_hash = logic.extract_text(file_path)
            
            if not extracted_text or len(extracted_text) < 5:
                flash("Could not read enough text. Ensure the image is clear and contains text.", "warning")
                return redirect(request.url)

            # 2. Check for EXACT duplicate hash (Fastest)
            existing_duplicate = Submission.query.filter_by(
                content_hash=file_hash, 
                assignment_id=assignment_id
            ).filter(Submission.user_id != current_user.id).first()

            max_score = 0.0
            reason = "Original Work"

            if existing_duplicate:
                max_score = 1.0
                reason = f"Exact duplicate of {existing_duplicate.author.username}'s file"
            else:
                # 3. Vector Similarity Check (Handwritten vs computerized vs other students)
                # Fetch all other students' texts in this course
                others = Submission.query.filter(
                    Submission.assignment_id == assignment_id,
                    Submission.user_id != current_user.id
                ).all()

                for other in others:
                    if other.text_content:
                        sim_score = logic.hybrid_similarity(extracted_text, other.text_content)
                        if sim_score > max_score:
                            max_score = sim_score
                            reason = f"High similarity with {other.author.username}"

            # 4. Final Decision Logic
            final_status = 'accepted'
            flash_category = "success"
            
            if max_score > 0.4:  # 40% threshold for AI similarity
                final_status = 'rejected'
                flash_category = "danger"
                flash_msg = f"Submission REJECTED: {reason} ({int(max_score*100)}%)"
            else:
                flash_msg = f"Submission Successful! Similarity Score: {int(max_score*100)}%."

            # 5. Save to Database (including the cleaned text for future indexing)
            new_sub = Submission(
                assignment_id=assignment_id,
                user_id=current_user.id,
                course_id=assignment.course_id,
                filename=filename,
                text_content=extracted_text, # Save text for next comparisons
                content_hash=file_hash,
                score=max_score,
                status=final_status,
                reason=reason,
                timestamp=datetime.datetime.now()
            )
            db.session.add(new_sub)
            db.session.commit()
            
            # Note: FAISS index sync removed for performance - rebuild manually if needed
            
            flash(flash_msg, flash_category)
            return redirect(url_for('course_page', course_id=assignment.course_id))

    return render_template('upload.html', assignment=assignment, attempts_made=attempts_made)

# --- REPORTS & PUBLISHING ---

@app.route('/course/<int:course_id>/reports')
@login_required
def view_reports(course_id):
    course = db.get_or_404(Course, course_id)
    assignments = Assignment.query.filter_by(course_id=course_id).all()
    total_subs = Submission.query.filter_by(course_id=course_id).count()
    rejected_subs = Submission.query.filter_by(course_id=course_id, status='rejected').count()
    
    return render_template('reports.html', 
                           course=course, 
                           assignments=assignments, 
                           total=total_subs, 
                           rejected=rejected_subs)

@app.route('/toggle_publish/<int:assignment_id>')
@login_required
def toggle_publish(assignment_id):
    if current_user.role != 'faculty': return redirect(url_for('dashboard'))
    assign = Assignment.query.get_or_404(assignment_id)
    assign.is_published = not assign.is_published
    db.session.commit()
    flash(f"Status updated to {'Published' if assign.is_published else 'Hidden'}", "info")
    return redirect(url_for('view_reports', course_id=assign.course_id))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Pre-load the FAISS index so it's ready for the first request
        sync_vector_engine()
    app.run(debug=True, use_reloader=False)