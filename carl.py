import json
import subprocess
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QComboBox, QFileDialog, QTextEdit
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import firebase_admin
from firebase_admin import firestore, credentials, db
import ui
#import pathlib
import subprocess
import textwrap
import argparse
import io
import os
import numpy as np
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
import whisper
import torch
import threading
import pyttsx4
import pyaudio 
from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
from pathlib import Path
import cv2
from PIL import Image
import threading
import requests
from openai import OpenAI
import openai
import subprocess
import base64
import ui
import google.generativeai as genai

class BackendThread(QThread):
    finished = pyqtSignal()
    stop_flag = False
    update_text_signal = pyqtSignal(str)

    def __init__(self, instructor_name, class_name, selected_model, parent=None):
        super().__init__(parent)
        self.instructor_name = instructor_name
        self.class_name = class_name
        self.selected_model = selected_model

    def run(self):
        # Your backend.py logic goes here
        # For simplicity, I'm including only the start_test part
        transcription = ['']
        global phrase 
        phrase = ''
        output = ''

        api_keys = ui.extract_api_keys('.\\assets\\logo_carl.png')
        openai_api_key = "add_api_key"#api_keys['openai_api_key']
        genai_api_key = api_keys['genai_api_key']
        client = OpenAI(api_key=openai_api_key)

        genai.configure(api_key=genai_api_key)

        # Set up the model
        generation_config = {
        "temperature": 0.9,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
        }

        vgeneration_config = {
        "temperature": 0.4,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 4096,
        }

        safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
        ]

        model = genai.GenerativeModel(model_name="gemini-pro",
                                    generation_config=generation_config,
                                    safety_settings=safety_settings)

        convo = model.start_chat(history=[])

        vmodel = genai.GenerativeModel(model_name="gemini-pro-vision",
                                    generation_config=vgeneration_config,
                                    safety_settings=safety_settings)


        cred = credentials.Certificate(ui.extract_api_keys('.\\assets\\logo_carl_backup.png'))

        # default_app = firebase_admin.initialize_app(cred, {
        # 'databaseURL': 'https://carl-9b3f3-default-rtdb.firebaseio.com/' 
        #     }) 
        database_ref = db.reference()

        dbase = firestore.client()

        # function to convert text to speech     
        def speak(text):
            response = client.audio.speech.create(
                                            model="tts-1",
                                            voice="alloy",
                                            input=text,
                                        )
            response.stream_to_file("output.mp3")
            audio_data, sample_rate = sf.read("output.mp3", dtype=np.int16)
            sd.play(audio_data, sample_rate)
            sd.wait()


        def take_picture():
            # # For Pi:
            # command = ['fswebcam', 'current_picture.jpg']
            # subprocess.run(command, check=True)
            
            cap = cv2.VideoCapture(0)  # Use the appropriate camera index if you have multiple cameras
        
            _, frame = cap.read()
            cv2.imwrite("current_picture.jpg", frame)

            # Open the image using Pillow and display it
            image = Image.open("current_picture.jpg")
            image.show()
            
            cap.release()

        # Function to encode the image (GPT-4 Vision preview)
        def encode_image(image_path):
          with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

        # Function to check for wake word in the transcription
        def check_for_wake_word():
            selected_model = self.selected_model
            global phrase
            global output
            previous_phrase = ''
            while True:
                with transcription_lock:
                    if phrase != previous_phrase:
                        if "hey carl," in phrase.lower():
                                previous_phrase = phrase
                                self.update_text_signal.emit("Wake word detected! in " + phrase)
                                current_transcription = database_ref.child('-Nju8pyaCCDrfh8v0C_T').child('line').get()
                                
                                if("blackboard" in phrase.lower() or "whiteboard" in phrase.lower()):
                                    take_picture()
                                # Validate that an image is present
                                    if not (img := Path("current_picture.jpg")).exists():
                                        raise FileNotFoundError(f"Could not find image: {img}")
                                    
                                    elif selected_model == "GPT":
                                        image_path = Path("current_picture.jpg")
                                        base64_image = encode_image(image_path)
                                        headers = {
                                            "Content-Type": "application/json",
                                            "Authorization": f"Bearer {openai_api_key}"
                                        }

                                        prompt =  '''You are a helpful teaching assistant named 'Carl' that helps the students better understand the lecture content. 
                                        Just answer the lastest question (you don't have to answer previous questions as they have already been answered,
                                        and don't repeat the question in your response). 
                                        Here is the lecture transcription: '''+ current_transcription +''' And you have also been given an optional image of the classroom whiteboard'''


                                        payload = {
                                                    "model": "gpt-4-vision-preview",
                                                    "messages": [
                                                        {
                                                        "role": "user",
                                                        "content": [
                                                            {
                                                            "type": "text",
                                                            "text": prompt,
                                                            },
                                                            {
                                                            "type": "image_url",
                                                            "image_url": {
                                                                "url": f"data:image/jpeg;base64,{base64_image}"
                                                            }
                                                            }
                                                        ]
                                                        }
                                                    ],
                                                    "max_tokens": 300
                                                    }


                                        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                                        response_dict = json.loads(response)

                                        output = response_dict['choices'][0]['message']['content']
                                        
                                        speak(output)


                                    else:
                                        image_parts = [
                                        {
                                            "mime_type": "image/jpeg",
                                            "data": Path("current_picture.jpg").read_bytes()
                                        },
                                        ]

                                        prompt_parts = [
                                        '''You are a helpful teaching assistant named 'Carl' that helps the students better understand the lecture content. 
                                        Just answer the lastest question (you don't have to answer previous questions as they have already been answered,
                                        and don't repeat the question in your response). Keep your answers brief and to the point.
                                        Here is the lecture transcription: '''+ current_transcription +''' And here is an optional image of the classroom whiteboard:''',
                                        image_parts[0],
                                        ]

                                        output = vmodel.generate_content(prompt_parts).text
                                        speak(output)
                                    
                                    
                                else:
                                    if selected_model == "GPT":
                                        system_prompt = f"You are a helpful teaching assistant named 'Carl' that helps the students better understand the lecture content. Just answer the latest question (you don't have to answer previous questions as they have already been answered, and don't repeat the question in your response). "
                                        user_prompt = f"Here is the lecture transcription: {current_transcription}" 
                                        completion = client.chat.completions.create(
                                            model="gpt-4-turbo-preview",
                                            messages=[
                                                {"role": "system", "content": system_prompt},
                                                {"role": "user", "content": user_prompt}
                                            ]
                                        )
                                        output = completion.choices[0].message.content

                                    else:
                                        convo.send_message('''You are a helpful teaching assistant named 'Carl' that helps the students better understand the lecture content. 
                                        Just answer the lastest question (you don't have to answer previous questions as they have already been answered,
                                        and don't repeat the question in your response). Keep your answers brief and to the point.
                                        Here is the lecture transcription: '''+ current_transcription)
                                        output = convo.last.text
                                    
                                    speak(output)

                                self.update_text_signal.emit(output)
                                output = ''
                            
                                # Implement any action to be taken after wake word detection
                sleep(1)  # Adjust the sleep time as necessary


        def add_or_get_instructor(instructor_name):
            instructor_ref = dbase.collection('Instructor').document(instructor_name)
            if not instructor_ref.get().exists:
                # If the instructor doesn't exist, add them to the database
                instructor_ref.set({})
            return instructor_ref

        def add_or_get_class(instructor_ref, class_name):
            class_ref = instructor_ref.collection('Class').document(class_name)
            if not class_ref.get().exists:
                # If the class doesn't exist, add it to the database
                class_ref.set({})
            return class_ref

        def add_or_get_material(class_ref):
            date = datetime.utcnow()
            material_ref = class_ref.collection('Material').document(date.strftime('%Y-%m-%d %H:%M:%S'))

        def add_or_get_lecture(class_ref):
            date = datetime.utcnow()
            lecture_ref = class_ref.collection('Lecture').document(date.strftime('%Y-%m-%d %H:%M:%S'))
            if not lecture_ref.get().exists:
                # If the lecture doesn't exist, add it to the database
                lecture_ref.set({
                    'Date': date,
                    'Content': ''  # You can set the content later
                })
            return lecture_ref

        def update_lecture_content(lecture_ref, transcription):
            # Update the content of the lecture with the transcription
            lecture_ref.update({
                'Content': transcription
            })

        #parser = argparse.ArgumentParser()
        # parser.add_argument("--model", default="tiny", help="Model to use",
        #                     choices=["tiny", "base", "small", "medium", "large"])
        # parser.add_argument("--non_english", action='store_true',
        #                     help="Don't use the english model.")
        # parser.add_argument("--energy_threshold", default=100,
        #                     help="Energy level for mic to detect.", type=int)
        # parser.add_argument("--record_timeout", default=3,
        #                     help="How real time the recording is in seconds.", type=float)
        # parser.add_argument("--phrase_timeout", default=4,
        #                     help="How much empty space between recordings before we "
        #                         "consider it a new line in the transcription.", type=float)
        # parser.add_argument("--instructor", default="Robert Avanzato", help="Instructor name")
        # parser.add_argument("--class_name", default="CMPEN 270", help="Class name")
        # if 'linux' in platform:
        #     parser.add_argument("--default_microphone", default='pulse',
        #                         help="Default microphone name for SpeechRecognition. "
        #                             "Run this with 'list' to view available Microphones.", type=str)
        # args = parser.parse_args()

        # Access the instructor and class name from args
        instructor_name = self.instructor_name
        class_name = self.class_name
        
        # The last time a recording was retrieved from the queue.
        phrase_time = None
        # Current raw audio bytes.
        last_sample = bytes()
        # Thread safe Queue for passing data from the threaded recording callback.
        data_queue = Queue()
        # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
        recorder = sr.Recognizer()
        recorder.energy_threshold = 100
        # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
        recorder.dynamic_energy_threshold = False

        #source = sr.Microphone(sample_rate=16000)
        # Important for linux users.
        # Prevents permanent application hang and crash by using the wrong Microphone
        if 'linux' in platform:
            mic_name = "pulse"
            if not mic_name or mic_name == 'list':
                print("Available microphone devices are: ")
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    print(f"Microphone with name \"{name}\" found")
                return
            else:
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if mic_name in name:
                        source = sr.Microphone(sample_rate=16000, device_index=index)
                        break
        else:
            source = sr.Microphone(sample_rate=16000)

        # Load / Download model
        # model = args.model
        # if args.model != "large" and not args.non_english:
        #     model = model + ".en"
        # audio_model = whisper.load_model(model)
        

        record_timeout = 3
        phrase_timeout = 4

        temp_file = NamedTemporaryFile(suffix=".wav", delete=False).name

        with source:
            recorder.adjust_for_ambient_noise(source)

        def record_callback(_, audio:sr.AudioData) -> None:
            """
            Threaded callback function to receive audio data when recordings finish.
            audio: An AudioData containing the recorded bytes.
            """
            # Grab the raw bytes and push it into the thread safe queue.
            data = audio.get_raw_data()
            data_queue.put(data)

        # Create a background thread that will pass us raw audio bytes.
        # We could do this manually but SpeechRecognizer provides a nice helper.
        recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

        # Cue the user that we're ready to go.
        self.update_text_signal.emit("Model loaded.\n")
        print("Model loaded.\n")
        # Initialize the lock for thread-safe access to the transcription
        global transcription_lock
        #global phrase
        transcription_lock = threading.Lock()

        # Start the wake word detection thread
        wake_word_thread = threading.Thread(target=check_for_wake_word)
        wake_word_thread.daemon = True  # Optional: makes the thread exit when the main thread exits
        wake_word_thread.start()
        
        while not self.stop_flag:
            try:
                now = datetime.utcnow()
                # Pull raw recorded audio from the queue.
                if not data_queue.empty():
                    phrase_complete = False
                    # If enough time has passed between recordings, consider the phrase complete.
                    # Clear the current working audio buffer to start over with the new data.
                    if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                        last_sample = bytes()
                        phrase_complete = True
                    # This is the last time we received new audio data from the queue.
                    phrase_time = now

                    # Concatenate our current audio data with the latest audio data.
                    while not data_queue.empty():
                        data = data_queue.get()
                        last_sample += data

                    # # Use AudioData to convert the raw data to wav data.
                    # audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                    # wav_data = io.BytesIO(audio_data.get_wav_data())

                    # # Write wav data to the temporary file as bytes.
                    # with open(temp_file, 'w+b') as f:
                    #     f.write(wav_data.read())

                    # Convert last_sample to a NumPy array
                    audio_array = np.frombuffer(last_sample, dtype=np.int16)

                    # Save the raw audio data to a temporary WAV file.
                    sf.write(temp_file, audio_array, source.SAMPLE_RATE)

                    # Read the transcription.
                    audio_file = open(temp_file, "rb")
                    text = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text"
                    )

                    # If we detected a pause between recordings, add a new item to our transcription.
                    # Otherwise edit the existing one.
                    if phrase_complete:
                        transcription.append(text)
                    else:
                        transcription[-1] = text

                    # Clear the console to reprint the updated transcription.
                    database_ref.child('-Nju8pyaCCDrfh8v0C_T').child('line').set('')
                    os.system('cls' if os.name=='nt' else 'clear')
                    for line in transcription:
                        phrase = line
                        self.update_text_signal.emit('flush-display')
                        self.update_text_signal.emit(phrase)
                        current_transcription = database_ref.child('-Nju8pyaCCDrfh8v0C_T').child('line').get()
                        updated_transcription = current_transcription + "\n" + line
                        
                        database_ref.child('-Nju8pyaCCDrfh8v0C_T').child('line').set(updated_transcription)
                    # Flush stdout.
                    
                    
                    # Infinite loops are bad for processors, must sleep.
                    sleep(0.25)
            except KeyboardInterrupt:
                self.stop_flag = True

        self.update_text_signal.emit("\n\nTranscription:")
        
        for line in transcription:
            self.update_text_signal.emit(line)
        self.update_text_signal.emit(output)

        # push to firebase
        transcription_joined = " ".join(transcription)
        instructor_ref = add_or_get_instructor(instructor_name)
        class_ref = add_or_get_class(instructor_ref, class_name)
        lecture_ref = add_or_get_lecture(class_ref)

        # Update the content of the lecture with the transcription
        update_lecture_content(lecture_ref, transcription_joined)

        self.finished.emit()

class TesterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CARL (Classroom Assistant using Real-time LLMs)")
        self.setGeometry(100, 100, 600, 400)
        self.setWindowIcon(QIcon('.\\assets\\logo_carl.png'))  # Set the path to your icon
        self.setStyleSheet(
            "QMainWindow {background-color: #333; color: #fff;} "
            "QLabel, QComboBox, QPushButton {background-color: #555; color: #fff; border: 1px solid #777;} "
            "QComboBox::drop-down {background-color: #555; border: 1px solid #777;}"
            "QComboBox::down-arrow {image: url(arrow_down.png);}"
        )

        cred = credentials.Certificate(ui.extract_api_keys('.\\assets\\logo_carl_backup.png'))
        #cred = credentials.Certificate("/home/raspberry/CARL-Prototype/carl-9b3f3-firebase-adminsdk-9ta75-9b99c0622a.json")
        firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://carl-9b3f3-default-rtdb.firebaseio.com/' 
            })

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

        self.model_label = QLabel("Select Model:", self)
        self.model_label.move(20, 120)
        self.model_combobox = QComboBox(self)
        self.model_combobox.setGeometry(150, 120, 200, 30)
        self.model_combobox.addItems(["Gemini", "GPT"])

        self.start_button = QPushButton("Start Test.py", self)
        self.start_button.setGeometry(20, 170, 120, 30)
        self.start_button.clicked.connect(self.start_test)

        self.stop_button = QPushButton("Stop Test.py", self)
        self.stop_button.setGeometry(160, 170, 120, 30)
        self.stop_button.clicked.connect(self.stop_test)

        self.upload_button = QPushButton("Upload PDF", self)
        self.upload_button.setGeometry(300, 170, 120, 30)
        self.upload_button.clicked.connect(self.upload_pdf)

        self.text_display = QTextEdit(self)
        self.text_display.setGeometry(20, 220, 500, 150)
        self.text_display.setReadOnly(True)
        self.text_display.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.text_display.setStyleSheet("background-color: #555; color: #fff; border: 1px solid #777;")

        # Process object to hold the running test.py process
        self.test_process = None
        self.backend_thread_finished = False
        self.backend_thread = None

    def get_instructor_names(self):
        # Retrieve the document IDs of all instructors from Firebase
        dbase = firestore.client()
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

    def update_class_combobox(self):
        # Update the class combobox based on the selected instructor
        selected_instructor = self.instructor_combobox.currentText()
        if selected_instructor:
            self.class_names = self.get_class_names(selected_instructor)
            self.class_combobox.clear()
            self.class_combobox.addItems(self.class_names)

    def update_text_display(self, text):
        # Slot that updates the GUI textbox
        if text == 'flush-display':
            self.text_display.clear()
        else:
            self.text_display.append(text)
        

    def start_test(self):
        self.text_display.append("Starting...\n")
        instructor_name = self.instructor_combobox.currentText()
        class_name = self.class_combobox.currentText()
        selected_model = self.model_combobox.currentText()

        if instructor_name and class_name and selected_model:
            if self.backend_thread and self.backend_thread.isRunning():
                self.text_display.append("Already running.")
            else:
                self.backend_thread = BackendThread(instructor_name, class_name, selected_model)
                self.backend_thread.update_text_signal.connect(self.update_text_display)
                #self.backend_thread.finished.connect(self.backend_thread_finished)
                self.backend_thread.start()
        

    def stop_test(self):
        # if self.test_process and self.test_process.poll() is None:
        #     try:
        #         self.test_process.send_signal(subprocess.signal.CTRL_C_EVENT)
        #         #For pi: signal.SIGINT
        #         self.test_process.wait()
        #     except KeyboardInterrupt:
        #         pass
        if self.backend_thread and self.backend_thread.isRunning():
            self.backend_thread.stop_flag = True
            self.backend_thread.quit()
            self.text_display.append("Stopped")

    def upload_pdf(self):
          #if instuctor name and class name exist (index in list)
        #pull ids of instructor and class from firebase
        #push pdf file to newly created matierals collection
        # Get the selected instructor and class names from the comboboxes
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
                        self.text_display.append("PDF file uploaded successfully.")

                else:
                    self.text_display.append(f"Class '{selected_class}' does not exist.")
            else:
                self.text_display.append(f"Instructor '{selected_instructor}' does not exist.")
        else:
            new_collection_ref = db.collection('NewCollection').add({})
            new_collection_id = new_collection_ref[1].id
            file_path, _ = QFileDialog.getOpenFileName(self, "Choose PDF File", "", "PDF files (*.pdf)")

            if file_path:
                new_collection_ref.add({'file_path': file_path})
                self.text_display.append("PDF file uploaded to a new collection successfully.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    tester_app = TesterApp()
    tester_app.show()
    sys.exit(app.exec_())
