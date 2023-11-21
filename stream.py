import streamlit as st

# Define function to display each page
def display_page(page_number):
    st.title(f"Page {page_number}")
    col1,col2 = st.columns(2)

    image_path = f"{page_number}.jpg"
    col1.image(image_path, caption=f"Image {page_number}", width=200)
    col2.image(image_path, caption=f"Image {page_number}", width=200)


selected_tab = st.sidebar.radio('Select a tab', ['Home', 'About Us', 'Learn'])

st.markdown(
    """
    <style>
        .tab:hover {
            background-color: #ddd;
            cursor: pointer;
        }
    </style>
    """,
    unsafe_allow_html=True
)

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
    st.write("Whether you're fully commited to learning ISL or just want to get your feet wet, we've got the course for you. We’ve even bundled our most popular courses for even greater savings!")
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
    choice = st.radio('Choose between Numbers and Alphabets:', ['Numbers', 'Alphabets'])

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

        # Display the selected page
