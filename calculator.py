import cv2
import numpy as np
import mediapipe as mp
import requests
import base64
import json
import os
from dotenv import load_dotenv

load_dotenv()
# ... [previous code remains the same] ...

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize a canvas to capture an image (white background)
canvas = np.ones((480, 640, 3), dtype="uint8") * 255


# Set up some drawing parameters
drawing = False
prev_x, prev_y = None, None

while True:
    
    # Wait for the user to capture the image

    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect the hand
    results = hands.process(rgb_frame)
    
    # Get hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the tip of the index finger
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, c = frame.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            
            if drawing:
                if prev_x is not None and prev_y is not None:
                    # Draw the line on the canvas
                    cv2.line(canvas, (prev_x, prev_y), (cx, cy), (0, 0, 0), 5)
                    
            prev_x, prev_y = cx, cy

            # Draw hand landmarks on the frame (optional)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display the frame and the canvas separately
    combined_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow('Finger Drawing', combined_frame)
    
    # Key press logic
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord('c'):  # Press 'c' to clear the canvas
        canvas = np.ones((480, 640, 3), dtype="uint8") * 255
        prev_x, prev_y = None, None
    elif key == ord('d'):  # Press 'd' to start/stop drawing
        drawing = not drawing
    elif key == ord('s'):  # Press 's' to save the canvas
        # Save the image
        image_path = 'math_problem.png'
        cv2.imwrite(image_path, canvas)



# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()

# Convert the captured image to base64
with open('math_problem.png', 'rb') as img_file:
    image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

# Use OpenAI's vision model to extract the math problem from the image
openai_url = "https://api.openai.com/v1/chat/completions"
openai_payload = {
    "model": "gpt-4o",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract the math problem from this image. Only provide the math problem, nothing else."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 300
}
openai_headers = {
    'Authorization': f'Bearer {os.environ.get("OPENAI_API_KEY")}',
    'Content-Type': 'application/json'
}

openai_response = requests.post(openai_url, headers=openai_headers, json=openai_payload)
openai_result = openai_response.json()

# Extract the math problem from the OpenAI response
if 'choices' in openai_result:
    math_problem = openai_result['choices'][0]['message']['content'].strip()
    print("Extracted Math Problem:", math_problem)
else:
    print("Error in extracting the math problem:", openai_result)
    exit()

# Step 3: Send the math problem to OpenAI API and get the solution
solution_payload = {
    "model": "gpt-4",  # or "gpt-3.5-turbo" based on your preference
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Solve the following math problem: {math_problem}"}
    ],
    "temperature": 0  # Make the output more deterministic
}

solution_response = requests.post(openai_url, headers=openai_headers, json=solution_payload)
solution_result = solution_response.json()

# Print the solution
if 'choices' in solution_result:
    solution = solution_result['choices'][0]['message']['content'].strip()
    print("Solution:", solution)
else:
    print("Could not solve the problem:", solution_result)