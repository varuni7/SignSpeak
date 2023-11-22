import streamlit as st
import torch
from PIL import Image
import io
import cv2
import numpy as np
import cv2
from streamlit.components.v1 import components

st.set_page_config(page_title="Sign Speak", page_icon="✌️")

# Initialize session state
if 'is_authenticated' not in st.session_state:
    st.session_state.is_authenticated = False

# Define actual credentials
actual_email = "email"
actual_password = "password"

# Create an empty container
placeholder = st.empty()

# Check if the user is authenticated
if not st.session_state.is_authenticated:
    # Insert a form in the container
    with placeholder.form("login"):
        st.markdown("#### Welcome to Sign Speak!")
        st.markdown("Enter your credentials")
        email = st.text_input("Email",key='email')
        password = st.text_input("Password", type="password",key='password')
        submit = st.form_submit_button("Login")

    if submit and email == actual_email and password == actual_password:
        # If the form is submitted and the email and password are correct,
        # set the authentication status to True
        st.session_state.is_authenticated = True
        # Clear the form/container
        placeholder.empty()
    elif submit:
        # If the form is submitted and the credentials are incorrect, display an error message
        st.error("Invalid Email and Password")

# If the user is authenticated, display the content
if st.session_state.is_authenticated:
    # Define function to display each page
    def display_page(page_number):
        st.title(f"Letter {chr(65+page_number-1)}")
        col1, col2 = st.columns(2)

        # Display an image from file
        image_path = f"{page_number}.jpg"
        col1.image(image_path, caption=f"Letter {chr(65+page_number-1)}", width=200)
        
        with col2:
            
            # Streamlit window to display webcam feed
            FRAME_WINDOW = st.image([])
            stop_button = col2.button("Stop Camera", key="stop_button")
            # OpenCV camera initialization
            camera = cv2.VideoCapture(0)

            while True:
                # Capture frame-by-frame
                ret, frame = camera.read()

                # Check if the frame is captured successfully
                if not ret:
                    break

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Display the frame in Streamlit
                FRAME_WINDOW.image(frame_rgb, channels="RGB")

                if stop_button:
                    break



            # Release the camera resources
            camera.release()
 
        # # Create a VideoCapture object
        # cap = cv2.VideoCapture(0)

        # if not cap.isOpened():
        #     st.error("Error capturing video stream.")
        # else:
        #     # Read frames from the camera and display them
        #     while True:
        #         ret, frame = cap.read()

        #         if not ret:
        #             st.error("Error capturing video stream.")
        #             break

        #         # Resize the frame to fit the column width
        #         resized_frame = cv2.resize(frame, (200, 200))

        #         # Display the frame in the Streamlit app
        #         col2.image(resized_frame, channels="BGR", use_column_width=True)


        

        # # Release the VideoCapture object when done
        # cap.release()

    selected_tab = st.sidebar.radio('Select a tab', ['Home', 'Learn', 'About Us'])

    # Display content based on the selected tab
    if selected_tab == 'Home':
        st.title('Welcome to Sign Speak')
        
        col1, col2 = st.columns(2)
        with col1:
            st.image("ondemand.png", caption='On-Demand lectures', use_column_width=True)

        with col2:
            st.header('On-Demand lectures')
            st.write('Progress at your own pace, change the video speed, replay lessons, and review content as needed. Each course is packed with vocabulary, numbers, tips about learning Indian Sign Language, fingerspelling practice, and special knowledge of Deaf culture.')
        
        col1, col2 = st.columns(2)
        with col1:
            st.header('Access on the go or at home!')
            st.write('Busy lifestyle? No problem! You can immerse yourself in ISL from the comfort of your home or even during your lunch break at work.')
        

        with col2:
            st.image("onthego.jpg", caption='On the Go!', use_column_width=True)

        st.header("ISL Courses")
        st.write("Whether you're fully committed to learning ISL or just want to get your feet wet, we've got the course for you. We’ve even bundled our most popular courses for even greater savings!")
        with st.expander("Click here to explore what we offer!", expanded=False):
            cols1,cols2 = st.columns(2)
            with cols1:
                st.write('**1. Learn ISL Alphabets**')
                st.write('**2. Learn ISL Numbers**')
                st.write('**3. Explore our ISL Glossary!**')
                st.write('Please proceed to the **Learn** tab to continue learning!')

            with cols2:
                st.image('signs.jpg')
            


    elif selected_tab == 'About Us':
        st.title('About Us')
        st.header('Our Mission')
        st.write('In a world increasingly interconnected, communication is the cornerstone of understanding and inclusion. However, for the Deaf and Hard of Hearing communities, effective communication often hinges on knowing sign language. Much like acquiring any foreign language, mastering ISL is a valuable skill that fosters meaningful interactions and bridges the communication gap.The Comprehensive ISL Teaching Website with Real-Time Feedback project aims to address a pressing need in society—a lack of accessible, effective, and interactive ISL learning resources. Our mission is to empower individuals to learn ISL as easily as they would a spoken language, enabling them to communicate with Deaf and dumb confidently. As part of user testing we are in talks with the Smiles Foundation to get iterative feedback on our product')
        st.header('Our Team')
        
    elif selected_tab == 'Learn':
        st.title('Learn')

        # Provide choices between numbers and alphabets
        choice = st.radio('Choose an option Numbers and Alphabets:', ['Numbers', 'Alphabets','Glossary'])

        if choice == 'Numbers':
            st.write('You selected Numbers. Add your code for Numbers here if needed.')
        elif choice == 'Alphabets':
            st.write("You've selected the Alphabets course")
            
            # Keep track of the current page number
            current_page = st.session_state.get('current_page', 1)
            
            # Add dropdown for all pages
            all_pages = list(range(1, 27))
            all_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
            page_selection = st.selectbox('Select a Letter', all_letters, index=current_page - 1)
            display_page(int(ord(page_selection)-65 + 1))
            # Add Next and Previous buttons in the same row
            col1, col2, col3 = st.columns(3)
            if col1.button('Previous', key='prev_button'):
                current_page = max(1, current_page - 1)
            col2.write(f'Page: {current_page}')
            if col3.button('Next', key='next_button'):
                current_page += 1

            # Update the current page in session state
            st.session_state.current_page = current_page
        
        elif choice == 'Glossary':
            # YouTube playlist URL
            playlist_url = "https://www.youtube.com/playlist?list=PLFjydPMg4DapfRTBMokl09Ht-fhMOAYf6"

            # Generate the HTML code for the embedded YouTube playlist
            iframe_code1 = f'<iframe width="700" height="500" src="https://www.youtube.com/embed/neE5Fg4FVtA?list=PLFjydPMg4DapfRTBMokl09Ht-fhMOAYf6" title="Module 1.3 Some polite useful phrases." frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>'
            iframe_code2 = f'<iframe width="700" height="500" src="https://www.youtube.com/embed/n42ohSmbAFI?list=PLFjydPMg4DapfRTBMokl09Ht-fhMOAYf6" title="Module 1.1 Manners and etiquettes" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>'
            iframe_code3 = f'<iframe width="700" height="500" src="https://www.youtube.com/embed/A-glx15JuWE?list=PLFjydPMg4DapfRTBMokl09Ht-fhMOAYf6" title="Module 8.3 Country and states of India." frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>'

            # Display the embedded playlist using the IFrame component
            st.markdown(iframe_code1, unsafe_allow_html=True)
            st.markdown(iframe_code2, unsafe_allow_html=True)
            st.markdown(iframe_code3, unsafe_allow_html=True)
