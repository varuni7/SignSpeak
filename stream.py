import streamlit as st

# Define function to display each page
def display_page(page_number):
    st.title(f"Page {page_number}")
    col1,col2 = st.columns(2)

    image_path = f"{page_number}.jpg"
    col1.image(image_path, caption=f"Image {page_number}", width=200)
    col2.image(image_path, caption=f"Image {page_number}", width=200)

# Create a sidebar for page navigation
selected_tab = st.sidebar.radio('Select a tab', ['Home', 'About Us', 'Learn'])

# Inject CSS for hover effect
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
    st.title('Welcome to the Home Page')
    st.write('This is the Home page content.')
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
        
