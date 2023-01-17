import streamlit as st  
import admin_portal
    
    
    
# Create an empty container
placeholder = st.empty()

actual_email = "bma52@mail.aub.edu"
actual_password = "123"
name = "Bothaina Amro"

# Insert a form in the container
with placeholder.form("login"):
    st.markdown("#### Enter your credentials")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    submit = st.form_submit_button("Login")

if submit and email == actual_email and password == actual_password:
    # If the form is submitted and the email and password are correct,
    # clear the form/container and display a success message
    placeholder.empty()
    st.success("Login successful")
elif submit and email != actual_email and password != actual_password:
    st.error("Login failed")
else:
    st.write(f'Welcome *{name}*')
    exec(open('admin_portal.py').read())