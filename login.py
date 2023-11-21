import streamlit as st
import torch
from PIL import Image
import io
import cv2
import numpy as np


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
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
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
        st.title(f"Page {page_number}")
        col1, col2 = st.columns(2)
        cap = cv2.VideoCapture(0)
        image_path = f"{page_number}.jpg"
        col1.image(image_path, caption=f"Image {page_number}", width=200)
        with col2:
                while cap.isOpened():
                    ret, frame = cap.read()

                    if not ret:
                        st.error("Error capturing video stream.")
                        break

                    resized_frame = cv2.resize(frame, (640, 480))

    selected_tab = st.sidebar.radio('Select a tab', ['Home', 'About Us', 'Learn'])

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
            st.image("1.jpg", caption='On-Demand lectures', use_column_width=True)

        st.header("ISL Courses")
        st.write("Whether you're fully committed to learning ISL or just want to get your feet wet, we've got the course for you. We’ve even bundled our most popular courses for even greater savings!")
        with st.expander("Click here to explore what we offer!", expanded=False):
            # Add clickable buttons inside the expander
            if st.button('Button 1'):
                st.session_state.current_choice = 'Numbers'
                st.session_state.current_page = 1
            if st.button('Button 2'):
                st.session_state.current_choice = 'Alphabets'
                st.session_state.current_page = 1
            if st.button('Button 3'):
                st.write('Button 3 clicked!')

    elif selected_tab == 'About Us':
        st.title('About Us')
        st.write('Learn more about us here.')

    elif selected_tab == 'Learn':
        st.title('Learn')

        # Provide choices between numbers and alphabets
        choice = st.radio('Choose an option Numbers and Alphabets:', ['Numbers', 'Alphabets','Glossary'])

        if choice == 'Numbers':
            st.write('You selected Numbers. Add your code for Numbers here if needed.')
        elif choice == 'Alphabets':
            st.write('You selected Alphabets. Here is the code:')
            
            # Keep track of the current page number
            current_page = st.session_state.get('current_page', 1)
            
            # Add dropdown for all pages
            all_pages = list(range(1, 27))
            page_selection = st.selectbox('Select a page', all_pages, index=current_page - 1)
            display_page(page_selection)
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
            st.write("test")