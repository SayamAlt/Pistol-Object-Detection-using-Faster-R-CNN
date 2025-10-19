import cv2, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from detection import pistol_detector

load_dotenv()

# Class names and colors
class_names = ['bg', 'Pistol']
color_sample = [
    (0,0,0),
    (36,100,85) # Pistol
]

# Telegram Bot Token
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

def detect_pistols(image):
    image_with_boxes = pistol_detector(image, class_names, color_sample)
    # Convert image to Numpy array
    image = np.array(image_with_boxes)
    _, encoded_image = cv2.imencode('.jpg', image_with_boxes)
    return BytesIO(encoded_image.tobytes())

async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "🔫 **Welcome to the Pistol Detection Bot!** 🎯\n\n"
        "📸 **Send me an image**, and I will:\n"
        "✅ Detect and classify pistols\n"
        "✅ Identify potential threats in images\n"
        "✅ Generate an image with bounding boxes 🔍\n\n"
        "**Detected Objects:**\n"
        "🟡 **Pistol**\n"
        "⚠️ **Potential Threat Warning**\n\n"
        "🚀 *Send an image to start the detection!*"
    )
    
async def help_command(update: Update, context: CallbackContext):
    help_text = (
        "🔫 **Pistol Detection Bot Help** 🎯\n\n"
        "This bot detects pistols in images and identifies potential threats. 📸\n"
        "Simply send an image, and I'll analyze it to detect:\n"
        "🟡 **Pistols**\n"
        "⚠️ **Potential Threat Alerts**\n\n"
        "🔍 The bot will also **generate an image with bounding boxes** to highlight detected pistols.\n\n"
        "**Available Commands:**\n"
        "📌 /start - Welcome message\n"
        "📌 /help - Information on how to use the bot\n"
        "📌 /about - Details about the detection model\n"
    )
    await update.message.reply_text(help_text)
    
async def about_command(update: Update, context: CallbackContext):
    about_text = (
        "🔫 **Pistol Object Detection Bot** 🎯\n\n"
        "This bot detects and classifies pistols in images using an advanced deep learning model. 🏆\n\n"
        
        "**🔍 Features:**\n"
        "✅ Detects and classifies **pistols** in images 🔍\n"
        "✅ **Bounding Boxes** around detected weapons for precise localization 📏\n"
        "✅ **Threat Identification**: Helps in security applications 🚨\n"
        "✅ **Deep Learning Model**: Utilizes **Faster R-CNN with ResNet-50** for high-accuracy detection 🧠\n"
        "✅ **Optimized Training Strategy**: Trained with SGD optimizer, StepLR learning rate scheduler, and augmentation 🚀\n"
        "✅ **Hardware Acceleration**: Runs efficiently on GPU for real-time performance ⚡\n\n"

        "**💡 Interesting Facts About Pistols:**\n"
        "🔹 **The first semi-automatic pistol** was invented in 1893 by **Hugo Borchardt**. 🏛️\n"
        "🔹 **The Glock 17** is one of the most widely used pistols by law enforcement worldwide. 👮\n"
        "🔹 **Revolvers vs. Semi-Automatic Pistols**: Revolvers use a rotating cylinder, while semi-autos use a magazine for faster reloading. 🔄\n"
        "🔹 **The world's smallest pistol**, the **SwissMiniGun**, is only 5.5 cm long but fires real bullets! 🔬\n"
        
        "📸 *Send an image to detect pistols now!*"
    )
    await update.message.reply_text(about_text)
    
async def handle_image(update: Update, context: CallbackContext):
    image = update.message.photo[-1:][0] # Process the highest resolution image
    image_file = await image.get_file()
    image_bytes = BytesIO(await image_file.download_as_bytearray())
    image = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)

    # Convert Numpy array to PIL image
    image = Image.fromarray(image)
    
    image_with_boxes = detect_pistols(image)
    
    # Send detected image with bounding boxes
    image_with_boxes.seek(0) # Reset file pointer before sending the image
    await update.message.reply_photo(photo=image_with_boxes, caption="Detected Pistols and Potential Threats")
    
def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("about", about_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.run_polling()
    
if __name__ == "__main__":
    main()