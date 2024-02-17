import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QComboBox, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import firebase_admin
from firebase_admin import firestore, credentials
import ui

class TesterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tester GUI")
        self.setGeometry(100, 100, 600, 400)
        self.setWindowIcon(QIcon('assets/logo_carl.png'))  # Set the path to your icon
        self.setStyleSheet(
            "QMainWindow {background-color: #333; color: #fff;} "
            "QLabel, QComboBox, QPushButton {background-color: #555; color: #fff; border: 1px solid #777;} "
            "QComboBox::drop-down {background-color: #555; border: 1px solid #777;}"
            "QComboBox::down-arrow {image: url(arrow_down.png);}"
        )


        cred = credentials.Certificate(ui.extract_api_keys('./assets/logo_carl_backup.png'))
        firebase_admin.initialize_app(cred)

        self.instructor_label = QLabel("Instructor Name:", self)
        self.instructor_label.move(20, 20)
        self.instructor_combobox = QComboBox(self)
        self.instructor_combobox.setGeometry(150, 20, 200, 30)
        self.instructor_combobox.addItems(self.get_instructor_names())
        self.instructor_combobox.currentIndexChanged.connect(self.update_class_combobox)

        self.class_label = QLabel("Class Name:", self)
        self.class_label.move(20, 70)
        self.class_combobox = QComboBox(self)
        self.class_combobox.setGeometry(150, 70, 200, 30)

        self.start_button = QPushButton("Start Test.py", self)
        self.start_button.setGeometry(20, 120, 120, 30)
        self.start_button.clicked.connect(self.start_test)

        self.stop_button = QPushButton("Stop Test.py", self)
        self.stop_button.setGeometry(160, 120, 120, 30)
        self.stop_button.clicked.connect(self.stop_test)

        self.upload_button = QPushButton("Upload PDF", self)
        self.upload_button.setGeometry(300, 120, 120, 30)
        self.upload_button.clicked.connect(self.upload_pdf)

        # Process object to hold the running test.py process
        self.test_process = None

    def get_instructor_names(self):
        dbase = firestore.client()
        instructors_ref = dbase.collection('Instructor')
        instructor_docs = instructors_ref.stream()
        instructor_names = [doc.id for doc in instructor_docs]
        return instructor_names

    def get_class_names(self, instructor_name):
        dbase = firestore.client()
        instructor_ref = dbase.collection('Instructor').document(instructor_name)
        class_docs = instructor_ref.collection('Class').stream()
        class_names = [doc.id for doc in class_docs]
        return class_names

    def update_class_combobox(self):
        selected_instructor = self.instructor_combobox.currentText()
        if selected_instructor:
            self.class_names = self.get_class_names(selected_instructor)
            self.class_combobox.clear()
            self.class_combobox.addItems(self.class_names)

    def start_test(self):
        instructor_name = self.instructor_combobox.currentText()
        class_name = self.class_combobox.currentText()
        activate_command = r'c:\Users\Jared\Downloads\CARL-Prototype\.venv\Scripts\Activate.ps1'
        test_command = f'python script.py --model tiny --energy_threshold 100 --instructor "{instructor_name}" --class_name "{class_name}"'
        full_command = f'powershell -Command "{activate_command}; {test_command}"'
        self.test_process = subprocess.Popen(full_command, shell=True)

    def stop_test(self):
        if self.test_process and self.test_process.poll() is None:
            try:
                self.test_process.send_signal(subprocess.signal.CTRL_C_EVENT)
                self.test_process.wait()
            except KeyboardInterrupt:
                pass

    def upload_pdf(self):
        selected_instructor = self.instructor_combobox.currentText()
        selected_class = self.class_combobox.currentText()
        db = firestore.client()

        if selected_instructor and selected_class:
            instructor_ref = db.collection('Instructor').document(selected_instructor)
            instructor_doc = instructor_ref.get()

            if instructor_doc.exists:
                class_ref = instructor_ref.collection('Class').document(selected_class)
                class_doc = class_ref.get()

                if class_doc.exists:
                    instructor_id = instructor_doc.id
                    class_id = class_doc.id

                    file_path, _ = QFileDialog.getOpenFileName(self, "Choose PDF File", "", "PDF files (*.pdf)")

                    if file_path:
                        materials_ref = class_ref.collection('Materials').add({'file_path': file_path})
                        print("PDF file uploaded successfully.")

                else:
                    print(f"Class '{selected_class}' does not exist.")
            else:
                print(f"Instructor '{selected_instructor}' does not exist.")
        else:
            new_collection_ref = db.collection('NewCollection').add({})
            new_collection_id = new_collection_ref[1].id
            file_path, _ = QFileDialog.getOpenFileName(self, "Choose PDF File", "", "PDF files (*.pdf)")

            if file_path:
                new_collection_ref.add({'file_path': file_path})
                print("PDF file uploaded to a new collection successfully.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    tester_app = TesterApp()
    tester_app.show()
    sys.exit(app.exec_())
