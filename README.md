# Converting ASL to speech in real time using VoiceOver

This project was an entry to HackRice 11, winning 2nd place overall and 1st place in the Data-to-Knowledge (D2K) track.

Our initial approach was using hand border detection. We used background subtraction and border detection to capture the contours (boundaries) of labelled hand images which we used to self-generate our dataset. We then used this dataset to train our CNN (Convolutional Neural Network). We realized later that this model was not the best since it lost data about the internal orientations of hands, which is important for some complex ASL gestures.

The next approach we tried was using Canny edge detection, which extracts all the edges in a hand after background subtraction. While this gives more detailed processed input data, this was still not as accurate as we wanted (at least 97% per hand movement) perhaps because the CNN could not find relationships between most of the different hand lines from different hands in our dataset.

Our final ML pipeline used MediaPipe, a Google open source library which extracts landmarks from each hand image and outputs the x, y and z coordinates of each hand image. Our training pipeline involved isolating hand images using background subtraction and then using MediaPipe to generate hand landmark data which is fed into our CNN.

Our final prediction process takes user-generated images and generates hand landmarks for them, feeding them one by one into our CNN. For each identified hand gesture, our model then outputs text which is then converted to speech using Google Cloud's Text To Speech API.

However, we did not have enough time to train a dataset for our final pipeline since we spent a lot of our time training our dataset for our initial pipeline. Our current model therefore uses our initial dataset and model which was trained for our border detection algorithm (not our final landmark model). This achieves a decent final accuracy of 92.6%, but can be made much better using our final landmark model.

We first plan to train a dataset for our final landmark detection pipeline. We also plan to improve our dataset by extending it to include the entire ASL dictionary, and provide support for other sign languages as well. We also aim to crowdsource data from ASL users, and ultimately create a user-friendly mobile application for use in casual settings where a laptop is not available.

For more information on the project:
https://devpost.com/software/voiceover-pcwjmi
