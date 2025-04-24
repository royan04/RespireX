from app import db, User, app  # make sure to import your Flask app
from werkzeug.security import generate_password_hash

with app.app_context():
    doctor = User(email='doctor1@gmail.com', password=generate_password_hash('doc@123'), role='doctor')
    patient = User(email='patient1@gmail.com', password=generate_password_hash('pat@123'), role='patient')

    db.session.add_all([doctor, patient])
    db.session.commit()

    print("Users added!")
