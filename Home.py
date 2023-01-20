import streamlit as st  
import admin_portal
    
    
    
# Create an empty container
placeholder = st.empty()

actual_email_admin = "admin@mail.aub.edu"
actual_password_admin = "123"
#name_admin = "Bothaina Amro"

actual_email_annotator = "annotator@mail.aub.edu"
actual_password_annotator = "456"
#name_annotator = "Fouad Zablith"

# Insert a form in the container
with placeholder.form("login"):
    st.markdown("#### Enter your credentials")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    submit = st.form_submit_button("Login")

if submit and email == actual_email_admin and password == actual_password_admin:
    # If the form is submitted and the email and password are correct,
    # clear the form/container and display a success message
    placeholder.empty()
    st.success("Login successful")
    #st.write(f'Welcome *{name_admin}*')
    exec(open('admin_portal.py').read())
elif submit and email == actual_email_annotator and password == actual_password_annotator:
    placeholder.empty()
    st.success("Login successful")
    #st.write(f'Welcome *{name_annotator}*')
    exec(open('ActionRec_web_app.py').read())
elif submit and (email != actual_email_annotator or actual_email_admin) and (password != actual_password_annotator or actual_password_annotator):
        st.error("Login failed")

    
