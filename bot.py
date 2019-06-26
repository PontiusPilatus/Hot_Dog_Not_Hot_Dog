import telebot
import requests
import os
import pprint
import pandas as pd
import numpy as np
import keras
import pickle
import cv2
import io
from io import BytesIO
from PIL import Image

TOKEN = os.environ.get('TOKEN_BOT')
bot = telebot.TeleBot(TOKEN)
RESCALE_SIZE = 224
bot = telebot.TeleBot(TOKEN)

model = keras.models.load_model('model.h5')
model._make_predict_function()

def request_file(file_id):
    file_info = bot.get_file(file_id)
    file = requests.get('https://api.telegram.org/file/bot{0}/{1}'
            .format(TOKEN, file_info.file_path), stream=True)
    return file

def get_image_array_from_response(response):
    img = Image.open(response.raw).convert('RGB')
    # Rewrite
    response.close()
    #
    img = np.array(img) 
    return img

def prepare_image(img):
    img = cv2.resize(img, (RESCALE_SIZE, RESCALE_SIZE))
    img = np.expand_dims(img, axis = 0)
    return img

def get_predict(img):
    result = model.predict(img)
    return "Hot Dog" if result == 1 else "Not Hot Dog"

@bot.message_handler(content_types=['photo'])
def handle_docs_audio(message):
    response = request_file(message.photo[-1].file_id)
    img = get_image_array_from_response(response)
    img = prepare_image(img)

    result = get_predict(img)

    bot.send_message(message.chat.id, result)

@bot.message_handler(func = lambda message: True)
def handle_any(message):
    bot.reply_to(message, "Отправь фото!")

bot.polling()