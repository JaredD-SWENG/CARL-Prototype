import subprocess
import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Combobox
from threading import Thread
import firebase_admin
from firebase_admin import firestore,credentials, db 

class TesterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tester GUI")
        
        cred = credentials.Certificate("C:\\Users\\Jared\\Downloads\\CARL-Prototype\\carl-9b3f3-firebase-adminsdk-9ta75-9b99c0622a.json")
        #cred = credentials.Certificate("/home/raspberry/CARL-Prototype/carl-9b3f3-firebase-adminsdk-9ta75-9b99c0622a.json")
        firebase_admin.initialize_app(cred)
        
        # Combobox for instructor name
        self.instructor_label = tk.Label(root, text="Instructor Name:")
        self.instructor_label.pack()
        # Populate instructor names from Firebase
        self.instructor_names = self.get_instructor_names()
        self.instructor_combobox = Combobox(root, values=self.instructor_names)
        self.instructor_combobox.pack(pady=5)


        # Combobox for class name
        self.class_label = tk.Label(root, text="Class Name:")
        self.class_label.pack()
        # Initially, classes will be empty until an instructor is selected
        self.class_names = []
        self.class_combobox = Combobox(root, values=self.class_names)
        self.class_combobox.pack(pady=5)
        # Bind a callback function when an instructor is selected
        self.instructor_combobox.bind("<<ComboboxSelected>>", self.update_class_combobox)

        # Button to start running test.py
        self.start_button = tk.Button(root, text="Start Test.py", command=self.start_test)
        self.start_button.pack(pady=10)

        # Button to stop running test.py
        self.stop_button = tk.Button(root, text="Stop Test.py", command=self.stop_test)
        self.stop_button.pack(pady=10)

        # Button to upload PDF files
        self.upload_button = tk.Button(root, text="Upload PDF", command=self.upload_pdf)
        self.upload_button.pack(pady=10)

        # Process object to hold the running test.py process
        self.test_process = None

    
    def get_instructor_names(self):
        dbase = firestore.client()
        # Retrieve the document IDs of all instructors from Firebase
        instructors_ref = dbase.collection('Instructor')
        instructor_docs = instructors_ref.stream()
        instructor_names = [doc.id for doc in instructor_docs]
        return instructor_names
    
    def get_class_names(self, instructor_name):
        dbase = firestore.client()
        # Retrieve the document IDs of all classes for a specific instructor from Firebase
        instructor_ref = dbase.collection('Instructor').document(instructor_name)
        class_docs = instructor_ref.collection('Class').stream()
        class_names = [doc.id for doc in class_docs]
        return class_names

    def update_class_combobox(self, event):
        # Update the class combobox based on the selected instructor
        selected_instructor = self.instructor_combobox.get()
        if selected_instructor:
            self.class_names = self.get_class_names(selected_instructor)
            self.class_combobox['values'] = self.class_names
            # Set the class combobox to the first class (if available)
            if self.class_names:
                self.class_combobox.set(self.class_names[0])

    def start_test(self):
        instructor_name = self.instructor_combobox.get()
        class_name = self.class_combobox.get()

        # # For Pi:
        # activate_command = 'source /home/raspberry/CARL-Prototype/venv/bin/activate'
        # test_command = f'python /home/raspberry/CARL-Prototype/script.py --model tiny --energy_threshold 100 --instructor "{instructor_name}" --class_name "{class_name}"'
        # full_command = f'{activate_command} && {test_command}'

        # self.test_process = subprocess.Popen(full_command, shell=True, executable='/bin/bash')

        activate_command = r'c:\Users\Jared\Downloads\CARL-Prototype\.venv\Scripts\Activate.ps1'
        test_command = f'python script.py --model tiny --energy_threshold 100 --instructor "{instructor_name}" --class_name "{class_name}"'
        full_command = f'powershell -Command "{activate_command}; {test_command}"'

        self.test_process = subprocess.Popen(full_command, shell=True)

    def stop_test(self):
        # Terminate the running process if it exists
        if self.test_process and self.test_process.poll() is None:
            try:
                # Send a signal to gracefully terminate the script.py process
                self.test_process.send_signal(subprocess.signal.CTRL_C_EVENT)
                self.test_process.wait()  # Wait for the process to finish
            except KeyboardInterrupt:
                # Handle KeyboardInterrupt (e.g., user pressing Ctrl+C in the console)
                pass
    
    def upload_pdf(self):
    
        #if instuctor name and class name exist (index in list)
        #pull ids of instructor and class from firebase
        #push pdf file to newly created matierals collection
        # Get the selected instructor and class names from the comboboxes
        selected_instructor = self.instructor_combobox.get()
        selected_class = self.class_combobox.get()
        db = firestore.client()
         # Check if both instructor and class names are selected
        if selected_instructor and selected_class:
            # Query Firestore to check if the instructor exists
            instructor_ref = db.collection('Instructor').document(selected_instructor)
            instructor_doc = instructor_ref.get()

            if instructor_doc.exists:
                # Query Firestore to check if the class exists under the instructor
                class_ref = instructor_ref.collection('Class').document(selected_class)
                class_doc = class_ref.get()

                if class_doc.exists:
                    # If both instructor and class exist, retrieve their IDs
                    instructor_id = instructor_doc.id
                    class_id = class_doc.id

                    # Ask the user to select a PDF file
                    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])

                     # Check if a file is selected
                    if file_path:
                        # Create a new "Materials" collection under the class name collection
                        materials_ref = class_ref.collection('Materials').add({'file_path': file_path})

                        # # Upload the PDF file to the "Materials" collection
                        # materials_ref.add({
                        #     'file_path': file_path,
                        
                        # })

                        print("PDF file uploaded successfully.")

                else:
                    print(f"Class '{selected_class}' does not exist.")
            else:
                print(f"Instructor '{selected_instructor}' does not exist.")
        else:

            #create new collection
            #push files (documents) to collection
            #get the collection id
            #send collection id to script.py
            #### under class collection reference sent id to materials

            # Create a new collection in Firebase
            new_collection_ref = db.collection('NewCollection').add({})

            # Get the ID of the newly created collection
            new_collection_id = new_collection_ref[1].id

            # Upload the PDF file to the new collection
            new_collection_ref.add({
                'file_path': file_path,
            })

            print("PDF file uploaded to a new collection successfully.")
            

if __name__ == "__main__":
    root = tk.Tk()
    app = TesterApp(root)
    
    root.mainloop()
