import json
import subprocess
import sys
from tkinter import image_types
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QComboBox, QFileDialog, QTextEdit, QCheckBox
from PyQt6.QtGui import QIcon, QPixmap, QImage, QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import firebase_admin
from firebase_admin import firestore, credentials, db
import ui
import subprocess
import io
import os
import numpy as np
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
import threading
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
import subprocess
import base64
import ui
import google.generativeai as genai


class BackendThread(QThread):
    finished = pyqtSignal()
    stop_flag = False
    update_text_signal = pyqtSignal(str)
    update_image_signal = pyqtSignal(str)

    def __init__(self, instructor_name, class_name, selected_model, instructions, pic_gen, parent=None):
        super().__init__(parent)
        self.instructor_name = instructor_name
        self.class_name = class_name
        self.selected_model = selected_model
        self.instructions = instructions
        self.pic_gen = pic_gen

    def run(self):
        # Your backend.py logic goes here

        transcription = ['']
        global phrase
        phrase = ''
        output = ''

        api_keys = ui.extract_api_keys('.\\assets\\logo_carl.png')

        openai_api_key = api_keys['openai_api_key']
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

        model = genai.GenerativeModel(model_name="gemini-1.0-pro-001",
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)

        convo = model.start_chat(history=[])
        image_convo = model.start_chat(history=[])

        vmodel = genai.GenerativeModel(model_name="gemini-pro-vision",
                                       generation_config=vgeneration_config,
                                       safety_settings=safety_settings)

        cred = credentials.Certificate(
            ui.extract_api_keys('.\\assets\\logo_carl_backup.png'))

        # default_app = firebase_admin.initialize_app(cred, {
        # 'databaseURL': 'https://carl-9b3f3-default-rtdb.firebaseio.com/'
        #     })
        database_ref = db.reference()

        dbase = firestore.client()

        # OPENAI ASSISTANTS
        assistant = client.beta.assistants.create(
            name="Classroom Teaching Assistant",
            description="You are a helpful classroom assistant named 'Carl'.",
            model="gpt-4-1106-preview",)

        # function to convert text to speech
        def speak(text):
            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text,
            )
            print("making audio file")
            response.stream_to_file("output.mp3")
            audio_data, sample_rate = sf.read("output.mp3", dtype=np.int16)
            sd.play(audio_data, sample_rate)
            sd.wait()

        def generate_image_with_dalle(keywords):
            # client = OpenAI()
            print("type of keywords", type(keywords))
            response = client.images.generate(
                model="dall-e-3",
                prompt=str(keywords),
                size="1024x1024",
                quality="standard",
                n=1,
            )
            image_url = response.data[0].url
            print(image_url)
            return image_url

        def get_image():
            while True:
                sleep(10)
                image_convo.send_message('''Given this lecture content, generate 5 keywords that represent the concepts explained in the lecture. These keywords will be used to generate an image so make sure that the keywords are based on visual concepts. Do not make stuff up. Only include concepts from the lecture.''' +
                                         database_ref.child('-Nju8pyaCCDrfh8v0C_T').child('line').get())
                print("keywords", image_convo.last.text)
                image_url = generate_image_with_dalle(image_convo.last.text)
                self.update_image_signal.emit(image_url)

        def take_picture():
            if 'linux' in platform:
                command = ['libcamera-still', '-o', 'current_picture.jpg']
                subprocess.run(command, check=True)
            else:
                # Use the appropriate camera index if you have multiple cameras
                cap = cv2.VideoCapture(0)

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
                        if "carl" in phrase.lower() or "karl" in phrase.lower():
                            previous_phrase = phrase
                            self.update_text_signal.emit(
                                "Wake word detected! in " + phrase)
                            current_transcription = database_ref.child(
                                '-Nju8pyaCCDrfh8v0C_T').child('line').get()

                            if ("blackboard" in phrase.lower() or "whiteboard" in phrase.lower()):
                                print("image")
                                take_picture()
                            # Validate that an image is present
                                if not (img := Path("current_picture.jpg")).exists():
                                    raise FileNotFoundError(
                                        f"Could not find image: {img}")

                                elif selected_model == "GPT":
                                    image_path = Path("current_picture.jpg")
                                    base64_image = encode_image(image_path)
                                    headers = {
                                        "Content-Type": "application/json",
                                        "Authorization": f"Bearer {openai_api_key}"
                                    }

                                    prompt = '''You are a helpful teaching assistant named 'Carl'. Your job is to help students thrive in the classroom.
            You will be given a lecture transcription. Use that transcription to respond to students and teachers. If the transcription has a question from a student, give a concise and brief answer to the question. 
            Your answers should encourage organic discussions amongst students, spark their curiosity and promote engagement. It should encourage students to critically think and anlyze the material. You may ask follow-up questions to keep your dialogue conversational. 
             Keep in mind that the student group belongs to K-12. Tailor your responses to their level of knowledge and understanding. If the teacher gives you special instructions, follow them while responding to students.
                                                                Just answer the lastest question (you don't have to answer previous questions as they have already been answered and don't repeat the question in your response).
                                        Here is the lecture transcription: ''' + self.instructions + " " + current_transcription + ''' and an image of the classroom whiteboard'''

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

                                    response = requests.post(
                                        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                                    response_dict = response.json()

                                    output = response_dict['choices'][0]['message']['content']

                                    speak(output)

                                else:  # Gemini Vision
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
                                        Here is the lecture transcription: ''' + current_transcription + ''' And here is an optional image of the classroom whiteboard:''',
                                        image_parts[0],
                                    ]

                                    output = vmodel.generate_content(
                                        prompt_parts).text
                                    speak(output)

                            else:  # GPT Text
                                if selected_model == "GPT":
                                    print("no image")
                                    system_prompt = f'''You are a helpful teaching assistant named 'Carl' for ''' + instructor_name + '''. Your job is to help students thrive in the classroom.
            You will be given a lecture transcription. Use that transcription to respond to students and teachers. If the transcription has a question from a student, give a concise and brief answer to the question. 
            Your answers should encourage organic discussions amongst students, spark their curiosity and promote engagement. It should encourage students to critically think and anlyze the material. You may ask follow-up questions to keep your dialogue conversational. 
             Keep in mind that the student group belongs to K-12. Tailor your responses to their level of knowledge and understanding. If the teacher gives you special instructions, follow them while responding to students.
                                                                Just answer the lastest question (you don't have to answer previous questions as they have already been answered and don't repeat the question in your response).'''
                                    user_prompt = f"Here is the lecture transcription: {self.instructions + current_transcription }"
                                    completion = client.chat.completions.create(
                                        model="gpt-4-turbo-preview",
                                        messages=[
                                            {"role": "system",
                                                "content": system_prompt},
                                            {"role": "user", "content": user_prompt}
                                        ],
                                        stream=True
                                    )
                                    # output = completion.choices[0].message.content
                                    output = ""

                                    for chunk in completion:
                                        latest_word = chunk.choices[0].delta.content
                                        # Check if latest_word is not empty or just whitespace
                                        if latest_word is not None and latest_word.strip():
                                            output += latest_word + " "
                                            # Check if the latest word contains a punctuation mark
                                            if any(p in latest_word for p in ['.', '!', '?']):
                                                # If punctuation mark found, speak the accumulated phrase
                                                speak(output)
                                                self.update_text_signal.emit(
                                                    output)
                                                # Reset accumulated_phrase for the next phrase
                                                output = ""

            #                         thread = client.beta.threads.create(
            #                             messages=[
            #                                 {
            #                                     "role": "user",
            #                                     "content": current_transcription
            #                                 }
            #                             ]
            #                         )

            #                         description = '''You are a helpful teaching assistant named 'Carl' for ''' + instructor_name + '''. Your job is to help students thrive in the classroom.
            # You will be given a lecture transcription. Use that transcription to respond to students and teachers. If the transcription has a question from a student, give a concise and brief answer to the question.
            # Your answers should encourage organic discussions amongst students, spark their curiosity and promote engagement. It should encourage students to critically think and anlyze the material. You may ask follow-up questions to keep your dialogue conversational.
            #  Keep in mind that the student group belongs to K-12. Tailor your responses to their level of knowledge and understanding. If the teacher gives you special instructions, follow them while responding to students.
            #                                                     Just answer the lastest question (you don't have to answer previous questions as they have already been answered and don't repeat the question in your response).'''
            #                         run = client.beta.threads.runs.create(
            #                             thread_id=thread.id,
            #                             assistant_id=assistant.id,
            #                             instructions=description + " " + self.instructions,
            #                         )

            #                         while run.status != 'completed':
            #                             run = client.beta.threads.runs.retrieve(
            #                                 thread_id=thread.id,
            #                                 run_id=run.id
            #                             )
            #                             print(run.status)
            #                             sleep(0.5)

            #                         messages = client.beta.threads.messages.list(
            #                             thread_id=thread.id)

            #                         output = messages.data[0].content[0].text.value

                                    # speak(output)

                                else:  # Gemini Text
                                    convo.send_message('''You are a helpful teaching assistant named 'Carl' that helps the students better understand the lecture content. 
                                        Just answer the lastest question (you don't have to answer previous questions as they have already been answered,
                                        and don't repeat the question in your response). Keep your answers brief and to the point. You might have special instructions 
                                                       from the teacher here:''' + self.instructions + ''' Here is the lecture transcription: ''' + current_transcription)
                                    output = convo.last.text

                                    speak(output)

                            output = ''

                            # Implement any action to be taken after wake word detection
            #             elif "this is the class teacher" in phrase.lower():
            #                 previous_phrase = phrase
            #                 self.update_text_signal.emit(
            #                     "Wake word detected! in " + phrase)
            #                 current_transcription = database_ref.child(
            #                     '-Nju8pyaCCDrfh8v0C_T').child('line').get()

            #                 if selected_model == "GPT":
            #                     system_prompt = f'''You are a helpful teaching assistant named 'Carl' for ''' + instructor_name + '''. Your job is to help students thrive in the classroom.
            # You will be given a lecture transcription. Use that transcription to respond to students and teachers. If the transcription has a question from a student, give a concise and brief answer to the question.
            # Your answers should encourage organic discussions amongst students, spark their curiosity and promote engagement. It should encourage students to critically think and anlyze the material. You may ask follow-up questions to keep your dialogue conversational.
            #  Keep in mind that the student group belongs to K-12. Tailor your responses to their level of knowledge and understanding. If the teacher gives you special instructions, follow them while responding to students.
            #                                                     Just answer the lastest question (you don't have to answer previous questions as they have already been answered and don't repeat the question in your response).'''
            #                     user_prompt = f"Here is the lecture transcription: {current_transcription}"
            #                     completion = client.chat.completions.create(
            #                         model="gpt-4-turbo-preview",
            #                         messages=[
            #                             {"role": "system",
            #                                 "content": system_prompt},
            #                             {"role": "user", "content": user_prompt}
            #                         ]
            #                     )
            #                     output = completion.choices[0].message.content
            #                     speak(output)
            #                     thread = client.beta.threads.create(
            #                         messages=[
            #                             {
            #                                 "role": "user",
            #                                 "content": current_transcription
            #                             }
            #                         ]
            #                     )

            #                     description = '''You are a helpful teaching assistant named 'Carl' for ''' + instructor_name + '''. You help students thrive in the classroom.
            # You will be given a lecture transcription. In the transcription, you will find instructions given by the teacher. The instructions will end with: this is the class teacher.
            # When you receive the instructions, ONLY respond with 'Got it'. Follow those instructions when you respond to students' questions.  '''
            #                     run = client.beta.threads.runs.create(
            #                         thread_id=thread.id,
            #                         assistant_id=assistant.id,
            #                         instructions=description + " " + self.instructions,
            #                     )

            #                     while run.status != 'completed':
            #                         run = client.beta.threads.runs.retrieve(
            #                             thread_id=thread.id,
            #                             run_id=run.id
            #                         )
            #                         print(run.status)
            #                         sleep(0.5)

            #                     messages = client.beta.threads.messages.list(
            #                         thread_id=thread.id)

            #                     output = messages.data[0].content[0].text.value

                sleep(0.75)  # Adjust the sleep time as necessary

        def add_or_get_instructor(instructor_name):
            instructor_ref = dbase.collection(
                'Instructor').document(instructor_name)
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
            material_ref = class_ref.collection('Material').document(
                date.strftime('%Y-%m-%d %H:%M:%S'))

        def add_or_get_lecture(class_ref):
            date = datetime.utcnow()
            lecture_ref = class_ref.collection('Lecture').document(
                date.strftime('%Y-%m-%d %H:%M:%S'))
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

        # parser = argparse.ArgumentParser()
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

        # source = sr.Microphone(sample_rate=16000)
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
                        source = sr.Microphone(
                            sample_rate=16000, device_index=index)
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

        def record_callback(_, audio: sr.AudioData) -> None:
            """
            Threaded callback function to receive audio data when recordings finish.
            audio: An AudioData containing the recorded bytes.
            """
            # Grab the raw bytes and push it into the thread safe queue.
            data = audio.get_raw_data()
            data_queue.put(data)

        # Create a background thread that will pass us raw audio bytes.
        # We could do this manually but SpeechRecognizer provides a nice helper.
        recorder.listen_in_background(
            source, record_callback, phrase_time_limit=record_timeout)

        # Cue the user that we're ready to go.
        self.update_text_signal.emit("Model loaded.\n")
        print("Model loaded.\n")
        # Initialize the lock for thread-safe access to the transcription
        global transcription_lock
        # global phrase
        transcription_lock = threading.Lock()

        # Start the wake word detection thread
        wake_word_thread = threading.Thread(target=check_for_wake_word)
        # Optional: makes the thread exit when the main thread exits
        wake_word_thread.daemon = True
        wake_word_thread.start()

        picture_thread = threading.Thread(target=get_image)
        picture_thread.daemon = True
        if self.pic_gen:
            picture_thread.start()

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
                    database_ref.child(
                        '-Nju8pyaCCDrfh8v0C_T').child('line').set('')
                    os.system('cls' if os.name == 'nt' else 'clear')
                    for line in transcription:
                        phrase = line
                        self.update_text_signal.emit('flush-display')
                        self.update_text_signal.emit(phrase)
                        current_transcription = database_ref.child(
                            '-Nju8pyaCCDrfh8v0C_T').child('line').get()
                        updated_transcription = current_transcription + "\n" + line

                        database_ref.child(
                            '-Nju8pyaCCDrfh8v0C_T').child('line').set(updated_transcription)
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


class CarlApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CARL (Classroom Assistant using Real-time LLMs)")
        self.setGeometry(0, 0, 1500, 800)
        self.setWindowIcon(QIcon('.\\assets\\logo_carl.png')
                           )  # Set the path to your icon
        self.setStyleSheet("""
    QMainWindow {
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2c3e50, stop:1 #34495e);
        color: #ecf0f1;
    }
    QLabel, QComboBox, QPushButton {
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3c3c3c, stop:1 #4c4c4c);
        color: #ecf0f1;
        border: 1px solid #555;
        border-radius: 5px;
        padding: 5px;
    }
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 15px;
        border-left-width: 1px;
        border-left-color: darkgray;
        border-left-style: solid;
        border-top-right-radius: 3px;
        border-bottom-right-radius: 3px;
    }
    QComboBox::down-arrow {
        image: url(arrow_down.png);
    }
    QPushButton:pressed {
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2c3c3c, stop:1 #3c3c3c);
    }
    QPushButton:hover {
        border: 1px solid #1abc9c;
    }
""")

        cred = credentials.Certificate(
            ui.extract_api_keys('.\\assets\\logo_carl_backup.png'))
        # cred = credentials.Certificate("/home/raspberry/CARL-Prototype/carl-9b3f3-firebase-adminsdk-9ta75-9b99c0622a.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://carl-9b3f3-default-rtdb.firebaseio.com/'
        })

        # Set a common font size for labels
        font = QFont("Helvetica", 20)

        # Model selection
        self.model_label = QLabel("Select Model:", self)
        self.model_label.setFont(font)
        self.model_label.setGeometry(20, 20, 230, 50)
        self.model_combobox = QComboBox(self)
        self.model_combobox.setGeometry(270, 20, 280, 50)
        self.model_combobox.setFont(font)
        self.model_combobox.addItems(["Gemini", "GPT"])

        # Instructor name selection
        self.instructor_label = QLabel("Instructor Name:", self)
        self.instructor_label.setFont(font)
        self.instructor_label.setGeometry(20, 90, 230, 50)
        self.instructor_combobox = QComboBox(self)
        self.instructor_combobox.setGeometry(270, 90, 280, 50)
        self.instructor_combobox.setFont(font)
        # Assuming get_instructor_names() is defined elsewhere
        self.instructor_combobox.addItems(self.get_instructor_names())
        self.instructor_combobox.currentIndexChanged.connect(
            self.update_class_combobox)

        # Class name selection
        self.class_label = QLabel("Class Name:", self)
        self.class_label.setFont(font)
        self.class_label.setGeometry(20, 160, 230, 50)
        self.class_combobox = QComboBox(self)
        self.class_combobox.setGeometry(270, 160, 280, 50)
        self.class_combobox.setFont(font)

        # Activity instructions
        self.instructions_label = QLabel(
            "Activity Instructions (Optional):", self)
        self.instructions_label.setFont(font)
        self.instructions_label.setGeometry(20, 230, 400, 50)
        self.instructions = QTextEdit(self)
        self.instructions.setGeometry(20, 300, 550, 200)
        self.instructions.setFont(font)
        self.instructions.setStyleSheet(
            "background-color: #555; color: #fff; border: 1px solid #777;")

        # Generate image toggle
        self.generate_image_label = QLabel("Generate Image:", self)
        self.generate_image_label.setFont(font)
        self.generate_image_label.setGeometry(20, 520, 230, 50)
        self.generate_image_toggle = QCheckBox(self)
        self.generate_image_toggle.setGeometry(270, 520, 80, 50)
        self.generate_image_toggle.setFont(font)
        # Connect the checkbox stateChanged signal
        self.generate_image_toggle.stateChanged.connect(
            self.toggle_image_label_visibility)

        # Start, Stop, Upload buttons
        self.start_button = QPushButton("Start", self)
        self.start_button.setGeometry(20, 590, 180, 50)
        self.start_button.setFont(font)
        self.start_button.clicked.connect(self.start_test)
        self.start_button.setEnabled(False)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.setGeometry(220, 590, 180, 50)
        self.stop_button.setFont(font)
        self.stop_button.clicked.connect(self.stop_test)
        self.stop_button.setEnabled(False)

        self.upload_button = QPushButton("Upload PDF", self)
        self.upload_button.setGeometry(420, 590, 180, 50)
        self.upload_button.setFont(font)
        self.upload_button.clicked.connect(self.upload_pdf)
        self.upload_button.setEnabled(False)

        # Text display
        self.text_display = QTextEdit(self)
        self.text_display.setGeometry(700, 20, 700, 300)
        self.text_display.setReadOnly(True)
        self.text_display.setFont(font)
        self.text_display.setStyleSheet(
            "background-color: #555; color: #fff; border: 1px solid #777;")

        # QLabel to display the image
        self.image_label = QLabel(self)
        self.image_label.setGeometry(900, 20, 300, 300)
        # Initially hide the image label if needed
        self.toggle_image_label_visibility()

        # Process object to hold the running test.py process
        self.test_process = None
        self.backend_thread_finished = False
        self.backend_thread = None

    def toggle_image_label_visibility(self):
        # Check if the checkbox is checked
        checkbox_is_checked = self.generate_image_toggle.isChecked()

        # Set the visibility of the image label based on the checkbox state
        self.image_label.setVisible(checkbox_is_checked)

        # Adjust the geometry of the text display based on the checkbox state
        if checkbox_is_checked:
            # Checkbox is checked, show the image label and position text display lower
            self.text_display.setGeometry(700, 350, 700, 300)
        else:
            # Checkbox is unchecked, hide the image label and move text display up
            self.text_display.setGeometry(
                700, 20, 700, 630)  # Adjust height if needed

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
        instructor_ref = dbase.collection(
            'Instructor').document(instructor_name)
        class_docs = instructor_ref.collection('Class').stream()
        class_names = [doc.id for doc in class_docs]
        return class_names

    def update_class_combobox(self):
        # Update the class combobox based on the selected instructor
        selected_instructor = self.instructor_combobox.currentText()
        selected_instructor = self.instructor_combobox.currentText()
        selected_class = self.class_combobox.currentText()
        selected_model = self.model_combobox.currentText()
        if selected_instructor and selected_model:
            self.class_names = self.get_class_names(selected_instructor)
            self.class_combobox.clear()
            self.class_combobox.addItems(self.class_names)
            self.enable_buttons()

    def enable_buttons(self):
        # Enable buttons if all necessary selections are made
        instructor_name = self.instructor_combobox.currentText()
        class_name = self.class_combobox.currentText()
        selected_model = self.model_combobox.currentText()

        if instructor_name and class_name and selected_model:
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.upload_button.setEnabled(True)

        else:
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.upload_button.setEnabled(False)

    def update_text_display(self, text):
        # Slot that updates the GUI textbox
        if text == 'flush-display':
            self.text_display.clear()
        else:
            self.text_display.append(text)

    def display_image(self, image_url):
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        image_bytes = response.content
        # Open the image using PIL
        pil_image = Image.open(io.BytesIO(image_bytes))
        # Convert PIL image to QImage
        q_image = QImage(pil_image.tobytes(), pil_image.width,
                         pil_image.height, QImage.Format.Format_RGB888)

        # Convert QImage to QPixmap
        pixmap = QPixmap.fromImage(q_image)
        # Set the QPixmap to the QLabel
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def start_test(self):
        self.text_display.append("Starting...\n")
        instructor_name = self.instructor_combobox.currentText()
        class_name = self.class_combobox.currentText()
        selected_model = self.model_combobox.currentText()
        instructions = self.instructions.toPlainText()
        pic_gen = self.generate_image_toggle.isChecked()

        if instructor_name and class_name and selected_model:
            if self.backend_thread and self.backend_thread.isRunning():
                self.text_display.append("Already running.")

            else:

                self.backend_thread = BackendThread(
                    instructor_name, class_name, selected_model, instructions, pic_gen)
                self.backend_thread.update_text_signal.connect(
                    self.update_text_display)
                self.backend_thread.update_image_signal.connect(
                    self.display_image)
                # self.backend_thread.finished.connect(self.backend_thread_finished)
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
        # if instuctor name and class name exist (index in list)
        # pull ids of instructor and class from firebase
        # push pdf file to newly created matierals collection
        # Get the selected instructor and class names from the comboboxes
        selected_instructor = self.instructor_combobox.currentText()
        selected_class = self.class_combobox.currentText()

        db = firestore.client()

        if selected_instructor and selected_class:
            instructor_ref = db.collection(
                'Instructor').document(selected_instructor)
            instructor_doc = instructor_ref.get()

            if instructor_doc.exists:
                class_ref = instructor_ref.collection(
                    'Class').document(selected_class)
                class_doc = class_ref.get()

                if class_doc.exists:
                    instructor_id = instructor_doc.id
                    class_id = class_doc.id

                    file_path, _ = QFileDialog.getOpenFileName(
                        self, "Choose PDF File", "", "PDF files (*.pdf)")

                    if file_path:
                        materials_ref = class_ref.collection(
                            'Materials').add({'file_path': file_path})
                        self.text_display.append(
                            "PDF file uploaded successfully.")

                else:
                    self.text_display.append(
                        f"Class '{selected_class}' does not exist.")
            else:
                self.text_display.append(
                    f"Instructor '{selected_instructor}' does not exist.")
        else:
            new_collection_ref = db.collection('NewCollection').add({})
            new_collection_id = new_collection_ref[1].id
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Choose PDF File", "", "PDF files (*.pdf)")

            if file_path:
                new_collection_ref.add({'file_path': file_path})
                self.text_display.append(
                    "PDF file uploaded to a new collection successfully.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    carl_app = CarlApp()
    carl_app.show()
    sys.exit(app.exec())
