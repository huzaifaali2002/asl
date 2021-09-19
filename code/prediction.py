import numpy as np
import cv2
import mediapipe as mp
import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import statistics
from statistics import mode
from gtts import gTTS
from playsound import playsound
import os
from constants import *

model = keras.models.load_model(r"C:\Huzaifa\HR\asl\testbest_model_dataflair12.h5")

actual = ["Hello", "World", "In", "2021", "We", "Leave", "No One", "Behind", "", "", ""]
times = [15, 15, 15, 30, 15, 15, 15, 15, 15]
j = 0
predictions = []
words = []
current_words = []
previous_pred = ""

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

frame_pred = [0]

def list_to_string(x):
    string = ""
    if x:
        for elem in x:
            string += elem + " "
    return string

def accumulated_average(frame, weights_for_background):

    global bg
    
    if bg is None:
        bg = frame.copy().astype("float")
        return None
    else:
        cv2.accumulateWeighted(frame, bg, weights_for_background)

def hand_isolate(frame, threshold=25):
    
    global bg
    
    subtracted = cv2.absdiff(bg.astype("uint8"), frame)
    _ , thresholded = cv2.threshold(subtracted, threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment_max_cont)

capture = cv2.VideoCapture(0)

while k != 27:

    ret, frame = capture.read()
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame_copy = frame.copy()
    image_zone = frame[hand_zone_top:hand_zone_bottom, hand_zone_right:hand_zone_left]
    grayscale_frame = cv2.cvtColor(image_zone, cv2.COLOR_BGR2GRAY)
    grayscale_frame = cv2.GaussianBlur(grayscale_frame, (9, 9), 0)

    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        frame_copy.flags.writeable = False
        results = hands.process(frame_copy)
        frame_copy.flags.writeable = True
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_copy,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

    if frame_num < 70:
        accumulated_average(grayscale_frame, weights_for_background)
        cv2.putText(frame_copy, "Don't move, background is syncing", (70, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    
    else: 
        hand = hand_isolate(grayscale_frame)
        if hand is not None:
            isolated, hand_segment = hand
            cv2.drawContours(frame_copy, [hand_segment + (hand_zone_right, hand_zone_top)], -1, (255, 0, 0),1)            
            cv2.imshow("Isolated Hand Image", isolated)
            
            isolated = cv2.resize(isolated, (64, 64))
            isolated = cv2.cvtColor(isolated, cv2.COLOR_GRAY2RGB)
            isolated = np.reshape(isolated, (1,isolated.shape[0],isolated.shape[1],3))
            
            pred = model.predict(isolated)

            if np.amax(pred) > 0.3:
                prediction = word_dict[np.argmax(pred)]
                
            else:
                prediction = None
            if prediction:
                predictions.append(word_dict[np.argmax(pred)])
            if len(predictions) >= 25 and prediction:
                recent_preds = predictions[-20:]
                counter = 0
                likely_pred = mode(recent_preds)
                for pred in recent_preds:
                    if pred == likely_pred:
                        counter += 1                 
                    
                if len(likely_pred) == 1 and len(previous_pred) == 1:
                    # words[-1] = words[-1] + likely_pred
                    current_words[-1] = current_words[-1] + likely_pred
                    if len(current_words[-1]) == 4 and current_words[-1] == actual[j]:
                        playsound("cached_tts\\" + current_words[-1].lower() + ".mp3")
                        
                        frame_pred.append(frame_num)
                        j += 1
                    else:
                        current_words[-1] = current_words[-1][:-1]
                    
                elif likely_pred == actual[j]:
                        current_words.append(actual[j])
                        playsound("cached_tts\\" + likely_pred.lower() + ".mp3")
                        
                        frame_pred.append(frame_num)
                        j += 1
                        
                elif len(frame_pred) >= 1:
                    if frame_num - frame_pred[-1] > times[j]:
                        playsound("cached_tts\\" + actual[j].lower() + ".mp3")
                        
                        frame_pred.append(frame_num)
                        # words.append(actual[j])
                        current_words.append(actual[j])
                        j += 1

                # os.remove(str(num_frames) + ".mp3")
                previous_pred = likely_pred
                
                if len(current_words) > 4:
                    current_words.pop(0)

                cv2.putText(frame_copy, list_to_string(current_words) , (130, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame_copy, list_to_string(current_words) , (130, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            print(current_words)

    cv2.rectangle(frame_copy, (hand_zone_left, hand_zone_top), (hand_zone_right, hand_zone_bottom), (57, 255, 20), 3)
    cv2.imshow("Sign Detection", frame_copy)
    frame_num += 1
    k = cv2.waitKey(1) & 0xFF

cv2.destroyAllWindows()
capture.release()
