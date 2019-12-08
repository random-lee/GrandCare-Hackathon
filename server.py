import os
import requests
from flask import Flask, request

from asr import ASR

app = Flask(__name__)

# ==============================================================================
# Parameters
# ==============================================================================

FB_API_URL = 'https://graph.facebook.com/v5.0/me/messages'
VERIFY_TOKEN = '7F151hSt4EBuUSxfc+OxEdNRR20ZCfkVe6i3ywb2ZUY='
PAGE_ACCESS_TOKEN = 'EAAvvvwDxUFABANyZBdYXUjeg6TRCLzgHJsctZCAq9xVRVrtZCgBbsbUPTZB9CnCJSmKBW1nqlyWBO1ZBFmAOME85iFr7pPeHGkm8oHHCBn9EfLXnxZCXVz1JKJnWywdW00aqgtAUa1s5AeeIrOmR3OGsZBt0xXMPvjlKfUQuhCxygZDZD'

asr_model = ASR()

# ==============================================================================
# Messager
# ==============================================================================

def triggers(text):

    text = ' ' + text.lower() + ' '
    
    if ' no ' in text:
        response = ' -- Can you please take your medication'
    
    elif ' not yet ' in text:
        response = ' -- Can you please take your medication'
        
    elif ' later ' in text:
        response = ' -- Can you please take your medication'
        
    elif ' yes ' in text:
        response = ' -- Thank you'
        
    elif ' ok ' in text:
        response = ' -- Thank you'
        
    else:
        response = " -- sorry didn't understand what you said"

    return response

# ------------------------------------------------------------------------------

def send_message(recipient_id, text):
    """Send a response to Facebook"""
    payload = {
        'message': {
            'text': text
        },
        'recipient': {
            'id': recipient_id
        },
        'notification_type': 'regular'
    }

    auth = {
        'access_token': PAGE_ACCESS_TOKEN
    }

    response = requests.post(
        FB_API_URL,
        params=auth,
        json=payload
    )

    return response.json()

# ------------------------------------------------------------------------------

def verify_webhook(req):
    if req.args.get("hub.verify_token") == VERIFY_TOKEN:
        return req.args.get("hub.challenge")
    else:
        return "incorrect"

# ------------------------------------------------------------------------------

def text_respond(sender_id, message):
    """Formulate a response to the user and
    pass it on to a function that sends it."""
    response = "You typed '{}'".format(message)
    response += triggers(message)
    
    send_message(sender_id, response)

# ------------------------------------------------------------------------------

def voice_respond(sender_id, trans):
    """Formulate a response to the user and
    pass it on to a function that sends it."""
    response = "You said '{}'".format(trans.lower())
    response += triggers(trans.lower())
    
    send_message(sender_id, response)

# ------------------------------------------------------------------------------

def get_voice_wav(url, output='./tmp.wav'):
    try:
        print(url)
        os.system('ffmpeg -i "{0}" {1}'.format(url, output))
    except Exception as err:
        print(err)

# ------------------------------------------------------------------------------

@app.route("/webhook", methods=['GET', 'POST'])
def listen():
    """This is the main function flask uses to 
    listen at the `/webhook` endpoint"""
    
    if request.method == 'GET':
        return verify_webhook(request)

    if request.method == 'POST':
        payload = request.get_json()

        event = payload['entry'][0]['messaging']
        print(event) #???
        for x in event:
            if 'message' in x:
                sender_id = x['sender']['id']
                # has message text
                if 'text' in x['message']:
                    text = x['message']['text']
                    print('text message recieved', text)
                    text_respond(sender_id, text)
                    
                elif 'attachments' in x['message']:
                    for attachment in x['message']['attachments']:
                        if attachment['type'] == 'audio':
                            # download url to tmp wav
                            print('downloading & converting')
                            output = './tmp.wav'
                            get_voice_wav(attachment['payload']['url'], output)
                            trans = asr_model.get_transcription(output)
                            voice_respond(sender_id, trans)
                            
                            os.system('rm {}'.format(output))
                            break

        return "ok"

# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    #send_message('106667660823037', 'Have you taken your medication?')
    app.run(port=5000, debug=True)
