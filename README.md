# Easy ML

This is a web app/website which you can use to apply basic machine learning algorithms such as SVMs and regression, to your data. It is not 100% finished yet, so bear with me here. I probably shouldn't be putting my email on the internet, but if you have any questions or suggestions please email me at zarif.azher@gmail.com. Please keep in mind that this is very much a work in progress and not everything works (yet).

# Known Issues/Possible Imrovements
- Can currently only work with cleaned data
- Sending model via email doesn't always work - I have to often direct Gmail to allow emails from 'unknown' sources
- Due to the way the framework is built, the entire app is rerun every time there is new user input. Thus, a new model is trained whenever you make a change. Ex; when you click 'send me my model', a new model is trained and that one is sent to you.

# Notes
- Up and running here: https://stark-waters-72314.herokuapp.com/
- Medium article for the project: https://medium.com/@zarif.azher/building-an-automl-tool-that-anyone-can-use-99405ed32dba
- If you are going to attempt to clone this and create the platform for yourself, you have to create a new file in the main directory called 'email_info.py'. It must specify your email address and password (currently only support Gmail, unless you configure 'send_files.py')...see the example below
```shell
sender_email = 'youremail@gmail.com'
sender_password = 'your_password'

def get_sender_email():
    return sender_email

def get_sender_password():
    return sender_password
```
- You'll also have to create a folder in the main directory called 'user_models'
- Additionally, if you attempt this custom usage, make sure to allow 'less secure' apps in your Gmail: https://hotter.io/docs/email-accounts/secure-app-gmail/
- The web app is made with a great framework built for Python data scientists and machine learning engineers, called Streamlit.
- It asks for your email only to send you your model, in the form of a Python pickle. I'm NOT storing your email anywhere or using it to send you spam mail.
