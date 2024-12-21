#! python3.7check_for_wake_word()
import pathlib
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
import firebase_admin
from firebase_admin import credentials, firestore
import threading
import requests
import ui

# Other imports remain the same

# Global variable to hold the transcribed text
transcription = ['']
phrase = ''
output = ''

import google.generativeai as genai
#just_a_variable="AIzaSyCi06GaEKyCKqjctTG-bldhQfcmmFBlXKA"
ui_color_scheme = ui.ui_colors_hex()
#print(just_a_variable)
ui_color_scheme = bytes.fromhex(ui_color_scheme).decode()
genai.configure(api_key=ui_color_scheme)

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



LLM_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
headers = {"Authorization": "Bearer add_api_key"}
engine = pyttsx4.init()

def query(payload):
    response = requests.post(LLM_API_URL, headers=headers, json=payload)
    return response.json()
     
def speak(text):
   engine.say(text)
   engine.startLoop(False)
   engine.iterate()
   engine.endLoop()

def take_picture():
    cap = cv2.VideoCapture(0)  # Use the appropriate camera index if you have multiple cameras

   
    _, frame = cap.read()
    cv2.imwrite("current_picture.jpg", frame)

    # Open the image using Pillow and display it
    image = Image.open("current_picture.jpg")
    image.show()


    cap.release()

# Function to check for wake word in the transcription
def check_for_wake_word():
    global phrase
    global output
    previous_phrase = ''
    while True:
        with transcription_lock:
            if phrase != previous_phrase:
                if "hey carl," in phrase.lower():
                        previous_phrase = phrase
                        print("Wake word detected! in " + phrase)
                        current_transcription = database_ref.child('-Nju8pyaCCDrfh8v0C_T').child('line').get()
                        
                        if("blackboard" in phrase.lower() or "whiteboard" in phrase.lower()):
                            take_picture()
                        # Validate that an image is present
                            if not (img := Path("current_picture.jpg")).exists():
                                raise FileNotFoundError(f"Could not find image: {img}")

                            image_parts = [
                            {
                                "mime_type": "image/jpeg",
                                "data": Path("current_picture.jpg").read_bytes()
                            },
                            ]

                            prompt_parts = [
                            '''You are a helpful teaching assistant named 'Carl' that helps the students better understand the lecture content. 
                            Just answer the lastest question (you don't have to answer previous questions as they have already been answered,
                            and don't repeat the question in your response). 
                            Here is the lecture transcription: '''+ current_transcription +''' And here is an optional image of the classroom whiteboard:''',
                            image_parts[0],
                            ]

                            output = vmodel.generate_content(prompt_parts).text
                            speak(output)
                        else:
                            convo.send_message('''You are a helpful teaching assistant named 'Carl' that helps the students better understand the lecture content. 
                            Just answer the lastest question (you don't have to answer previous questions as they have already been answered,
                            and don't repeat the question in your response). 
                            Here is the lecture transcription: '''+ current_transcription)
                            output = convo.last.text
                            speak(output)

                        print(output)
                        output = ''
                       
                        # Implement any action to be taken after wake word detection
        sleep(1)  # Adjust the sleep time as necessary

from firebase_admin import credentials, db
cred = credentials.Certificate("C:\\Users\\Jared\\Downloads\\CARL-Prototype\\carl-9b3f3-firebase-adminsdk.json")
#cred = credentials.Certificate("/home/raspberry/CARL-Prototype/carl-9b3f3-firebase-adminsdk.json")
# realtime stuff
default_app = firebase_admin.initialize_app(cred, {
'databaseURL': 'https://carl-9b3f3-default-rtdb.firebaseio.com/' 
    }) 
database_ref = db.reference()

# cloud firestore stuff

dbase = firestore.client()

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



def main():
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    parser.add_argument("--instructor", default="Robert Avanzato", help="Instructor name")
    parser.add_argument("--class_name", default="CMPEN 270", help="Class name")
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # Access the instructor and class name from args
    instructor_name = args.instructor
    class_name = args.class_name

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
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
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name
    # transcription = ['']

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
    print("Model loaded.\n")
    # Initialize the lock for thread-safe access to the transcription
    global transcription_lock
    global phrase
    transcription_lock = threading.Lock()

    # Start the wake word detection thread
    wake_word_thread = threading.Thread(target=check_for_wake_word)
    wake_word_thread.daemon = True  # Optional: makes the thread exit when the main thread exits
    wake_word_thread.start()

    # Start the picture-taking thread
    # picture_thread = threading.Thread(target=take_picture)
    # picture_thread.daemon = True
    # picture_thread.start()
    
    while True:
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

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
                text = result['text'].strip()

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
                    print(phrase)
                    current_transcription = database_ref.child('-Nju8pyaCCDrfh8v0C_T').child('line').get()
                    updated_transcription = current_transcription + "\n" + line
                    
                    database_ref.child('-Nju8pyaCCDrfh8v0C_T').child('line').set(updated_transcription)
                # Flush stdout.
                print('', end='', flush=True)
                 
                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
     
    for line in transcription:
        print(line)
    print(output)

    # push to firebase
    transcription_joined = " ".join(transcription)
    instructor_ref = add_or_get_instructor(instructor_name)
    class_ref = add_or_get_class(instructor_ref, class_name)
    lecture_ref = add_or_get_lecture(class_ref)

    # Update the content of the lecture with the transcription
    update_lecture_content(lecture_ref, transcription_joined)

    

if __name__ == "__main__":
    main()
