import requests
from googletrans import Translator as Tr
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackContext, ConversationHandler
from translate import Translator
import os
import random

"Чтобы найти бота t.me/my_1213323bot"
TOKEN = "7040147021:AAG28PQRvnA5K-PlCiUn4UXGeCXIOGsN8t0"
API_KEY = "d6e9ab901cff0a5239be68dc4801ea5a"
translator = Translator(from_lang='en', to_lang='ru', secret_access_key='weather report')

reply_keyboard = [['/help', '/weather', '/start'],
                  ['/photo', '/weather_list', '/music']]
markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=False)
reply_markup = markup


def get_music_file(folder_path):
    """
    Случайно выбирает музыкальный файл из папки.
    """
    music_files = [f for f in os.listdir(folder_path) if f.endswith((".mp3", ".wav", ".ogg"))]
    if music_files:
        return os.path.join(folder_path, random.choice(music_files))
    else:
        return None


def send_music(update, context):
    """
    Отправляет музыкальный файл пользователю.
    """
    chat_id = update.effective_chat.id
    music_folder = f"music"  # Замените на путь к вашей папке с музыкой
    music_file = get_music_file(music_folder)
    if music_file:
        context.bot.send_audio(chat_id, audio=open(music_file, 'rb'))
    else:
        context.bot.send_message(chat_id, "Музыкальные файлы не найдены")


def save_city(update: Update, context: CallbackContext) -> int:
    city = " ".join(context.args)
    context.user_data['city'] = city  # Сохраняем город в user_data
    update.message.reply_text(f"Город сохранен: {city}")
    return ConversationHandler.END


def translate_text(text):
    translation = Tr().translate(text, dest='ru', src='en')
    return translation.text


def get_random_weather_image():
    """
    Получает случайное фото погодного явления с Unsplash.
    """
    url = "https://api.unsplash.com/photos/random"
    params = {
        "client_id": "3R7wYzg0sni89-5oWXwEAyBVZ9DFNrMCgF7OEDYmvi4",
        "query": "weather",
        "orientation": "landscape"
    }
    response = requests.get(url, params=params)
    data = response.json()
    image_url = data['urls']['regular']
    return image_url


def weather_photo(update, context):
    """
    Отправляет случайное фото погодного явления в чат.
    """
    chat_id = update.effective_chat.id
    image_url = get_random_weather_image()
    context.bot.send_photo(chat_id, photo=image_url)


# Функция для получения прогноза погоды
def get_weather(update: Update, context: CallbackContext):
    city = context.user_data.get('city')
    if not city:
        update.message.reply_text("Город не указан. Укажите город, написав /city <ваш город>")
    else:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            weather_en = data['weather'][0]['description']
            weather = translator.translate(weather_en)
            if (97 <= ord(weather[-1]) <= 122) or (65 <= ord(weather[0]) <= 90):
                weather = translate_text(weather_en)
            weather = weather.lower()
            temp = data['main']['temp']
            humidity = data['main']['humidity']
            wind_speed = data['wind']['speed']

            # Определение иконок для разных типов погоды
            weather_icons = {
                "ясно": "☀️",
                "переменная облачность": "⛅",
                "несколько облаков": "⛅",
                "пасмурные облака": "☁",
                "снег": "🌨️",
                "снегопад": "🌨️",
                "ветер": "💨",
                "туман": "🌫️",
                "облаков": "☁️",
                "облачно": "☁️",
                "дождь": "🌧️"
            }

            weather_icon = "❓"  # По умолчанию, если нет соответствующей иконки

            # Проверяем описание погоды и выбираем соответствующую иконку
            for key, value in weather_icons.items():
                if key.lower() in weather.lower():
                    weather_icon = value
                    break

            message = f"Погода в {city}:\n{weather_icon} {weather}\nТемпература: {temp}°C\nВлажность: {humidity}%\nСкорость ветра: {wind_speed} м/с"
        else:
            message = "Извините, что-то пошло не так. Проверьте название города и попробуйте еще раз."

        update.message.reply_text(message)


def start(update: Update, context: CallbackContext):
    update.message.reply_text("Привет! Я бот прогноза погоды. Введи /help, чтобы узнать что я могу", reply_markup=markup)


def help_command(update: Update, context: CallbackContext):
    update.message.reply_text("Доступные команды:\n"
        "/weather - получить прогноз погоды.\n"
        "/city - указать город.\n"
        "/weather_list - получить прогноз погоды на несколько дней.\n"
        "/photo - отправить случайное фото погодного явления.\n"
        "/music - отправить музыкальный файл.\n"
        "/help - показать список команд и их описание.\n"
        "/start - начать взаимодействие с ботом.\n")


def get_forecast(update: Update, context: CallbackContext):
    if not context.user_data.get('city'):
        update.message.reply_text("Город не указан. Укажите город, написав /city <ваш город>")
    else:
        city = context.user_data.get('city')
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            forecast_data = data['list']  # Получаем данные прогноза на несколько дней
            forecast_message = f"Прогноз погоды на несколько дней в городе {city}:\n"

            weather_icons = {
                "ясно": "☀️",
                "переменная облачность": "⛅",
                "несколько облаков": "⛅",
                "пасмурные облака": "☁",
                "снег": "🌨️",
                "снегопад": "🌨️",
                "ветер": "💨",
                "туман": "🌫️",
                "облаков": "☁️",
                "облачно": "☁️",
                "дождь": "🌧️"
                # добавьте другие типы погоды с соответствующими иконками по желанию
            }

            for forecast in forecast_data:
                date_time = forecast['dt_txt']
                weather_en = forecast['weather'][0]['description']
                weather = translator.translate(weather_en)
                if (97 <= ord(weather[-1]) <= 122) or (65 <= ord(weather[0]) <= 90):
                    weather = translate_text(weather_en)
                weather = weather.lower()
                temp = forecast['main']['temp']

                weather_icon = "❓"  # По умолчанию, если нет соответствующей иконки

                # Проверяем описание погоды и выбираем соответствующую иконку
                for key, value in weather_icons.items():
                    if key.lower() in weather.lower():
                        weather_icon = value
                        break

                forecast_message += f"\nДата/Время: {date_time}\n{weather_icon} {weather}\nТемпература: {temp}°C\n--------"

            update.message.reply_text(forecast_message)
        else:
            update.message.reply_text("Извините, что-то пошло не так. Проверьте название города и попробуйте снова.")


def main():
    updater = Updater(TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("weather", get_weather))
    dispatcher.add_handler(CommandHandler("weather_list", get_forecast))
    dispatcher.add_handler(CommandHandler("city", save_city))
    dispatcher.add_handler(CommandHandler("photo", weather_photo))
    dispatcher.add_handler(CommandHandler("music", send_music))

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
