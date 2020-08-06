import email
import smtplib
import ssl
import pickle
import streamlit as st
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.mime.application import MIMEApplication

class Email():
    def __init__(self,sender_email,receiver_email,subject,message,model_name,model_folder,sender_password):
        self.sender_email = sender_email
        self.receiver_email = receiver_email
        self.subject = subject
        self.message = message
        self.model_path = model_folder
        self.model_name = model_name
        self.sender_password = sender_password

    def send_email(self):
        try:
            message = MIMEMultipart()
            message['From'] = self.sender_email
            message['To'] = self.receiver_email
            message['Subject'] = self.subject

            message.attach(MIMEText(self.message,'plain'))
            st.write('Getting ready to send!')
            with open(self.model_path+self.model_name,'rb') as file:
                try:
                    part = MIMEApplication(file.read(),name=self.model_name)
                    part['Content-Disposition'] = 'attachment; filename="%s"' % self.model_name
                    message.attach(part)
                except Exception as e:
                    st.write(e)
                    #print(type(message))
                    #print(message)
                    #print(e)

            context = ssl.create_default_context()
            text = message.as_string()
            #print('ready to send!')
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                #print('here we go!')
                #print(self.sender_email, self.sender_password)
                server.login(self.sender_email, self.sender_password)
                #print('logged in!')
                try:
                    server.sendmail(self.sender_email, self.receiver_email, text)
                    #print('sent!')
                except Exception as e:
                    st.write(e)
                    #print('couldnt send')
                    #print(e)
        except Exception as f:
            st.write(f)
