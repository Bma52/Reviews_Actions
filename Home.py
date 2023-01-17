import streamlit_authenticator as stauth
import streamlit as st
import yaml
from yaml.loader import SafeLoader



with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
    
    
    
    
authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)



name, authentication_status, username = authenticator.login('Login', 'main')


name, authentication_status, username = authenticator.login('Login', 'main')
if authentication_status:
    authenticator.logout('Logout', 'main')
    if username == 'bma52':
        st.write(f'Welcome *{name}*')
        st.title('Annotator Page')
    elif username == 'fz13' or username == 'wk47':
        st.write(f'Welcome *{name}*')
        st.title('Admin page')
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
    
    
