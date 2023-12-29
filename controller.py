import subprocess
import tkinter as tk
from threading import Thread

class TesterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tester GUI")

        # Entry for instructor name
        self.instructor_label = tk.Label(root, text="Instructor Name:")
        self.instructor_label.pack()
        self.instructor_entry = tk.Entry(root)
        self.instructor_entry.pack(pady=5)

        # Entry for class name
        self.class_label = tk.Label(root, text="Class Name:")
        self.class_label.pack()
        self.class_entry = tk.Entry(root)
        self.class_entry.pack(pady=5)

        # Button to start running test.py
        self.start_button = tk.Button(root, text="Start Test.py", command=self.start_test)
        self.start_button.pack(pady=10)

        # Button to stop running test.py
        self.stop_button = tk.Button(root, text="Stop Test.py", command=self.stop_test)
        self.stop_button.pack(pady=10)

        # Process object to hold the running test.py process
        self.test_process = None

    def start_test(self):
        instructor_name = self.instructor_entry.get()
        class_name = self.class_entry.get()

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


if __name__ == "__main__":
    root = tk.Tk()
    app = TesterApp(root)
    
    root.mainloop()
