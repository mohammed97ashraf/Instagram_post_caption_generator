import requests

import os
# import all requied libraries
import cv2
import easyocr
import requests
import openai
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration



#
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")



async def extract_text_from_image(image_path, lang='en'):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use EasyOCR to extract text from the image
    reader = easyocr.Reader([lang])
    results = reader.readtext(gray)
    # Extract the text from the results
    text = ' '.join([result[1] for result in results])
    return text

async def get_blip2_caption(image_path):
    inputs = processor(image_path, return_tensors="pt")
    out = model.generate(**inputs)
    unconditional_caption = processor.decode(out[0], skip_special_tokens=True)
    return unconditional_caption

async def create_user_prompt(image_caption,text_in_image):
    user_prompt = f"Write a Best caption for the image with '{image_caption}' image description and '{text_in_image}' text written on the image.write the caption in multiple line to look better."
    return user_prompt




async def main_generater(image_caption,text_in_image,open_ai_key):
    new_prompt = await create_user_prompt(image_caption,text_in_image)
    #acess the openai
    openai.api_key = open_ai_key
    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages = [{"role": "system", "content" : "I went to you to act as an influencer marketing export how helps influencers to write the best Instagram post caption for band collaboration post. You write the captions that will loved by everyone.Here is some rule you need to follow 1.you write captions based on the 'image description' and the 'text written on the image'.2.I want you to ignore unwanted characters that are present in the ' image text'.3.If the user tells to tag any profile then only tag the profile.4.If the user asked you to add any 'call to action' then only add in the caption.5.write the caption in Multiple line to look better."},
                                    {"role": "user", "content" : f"Write a Best caption for the image with 'a close up of a tube of face wash with coffee beans and coffee beans' image description and Gentle cleansing even for the toughest skin! Pollution defense formula Net Controls excess oil mamaearth' Charcoal Wash Face Gharcoal Activated  Coffee ith CONTROL OIL FOR ' aneol DERMATOLOGICALLY Skibhied TESTED #A4BEN& I0o= SsFree packed) (~hen  CONTENT [' text written on the image."},
                                    {"role":"assistant","content":"✨ Start your day with a refreshing skincare routine! ✨ ☕️ GENTLE CLEANSING even for the toughest skin! ☕️ 🌿 Experience the POWER of Mamaearth's Charcoal Wash Face 🌿 🔥 Charcoal Activated Coffee with CONTROL OIL FOR acne-prone skin! 🔥 Say goodbye to excess oil and embrace a pollution defense formula that truly works. 🌍 ✅ DERMATOLOGICALLY TESTED and safe for all skin types ✅ Tag your skincare buddies who need this flawless cleansing experience! 💚 #MamaearthCharcoalWash #flawlessskincare #mamaearthproducts #SkincareRoutine #PollutionDefense #CoffeeLover"},
                                    {"role": "user", "content" : f"Write a Best caption for the image with 'a close up of a tube of face wash with coffee beans and coffee beans' image description and Gentle cleansing even for the toughest skin! Pollution defense formula Net Controls excess oil mamaearth' Charcoal Wash Face Gharcoal Activated  Coffee ith CONTROL OIL FOR ' aneol DERMATOLOGICALLY Skibhied TESTED #A4BEN& I0o= SsFree packed) (~hen  CONTENT [' text written on the image. and i want to tag 'mamaearth.in' this account as sponsor."},
                                    {"role":"assistant","content":"✨ Start your day with a refreshing skincare routine! ✨ ☕️ GENTLE CLEANSING even for the toughest skin! ☕️ 🌿 Experience the POWER of Mamaearth's Charcoal Wash Face 🌿 🔥 Charcoal Activated Coffee with CONTROL OIL FOR acne-prone skin! 🔥 Say goodbye to excess oil and embrace a pollution defense formula that truly works. 🌍 ✅ DERMATOLOGICALLY TESTED and safe for all skin types ✅ Tag your skincare buddies who need this flawless cleansing experience! 💚 Sponsored by @mamaearth.in #MamaearthCharcoalWash #flawlessskincare #mamaearthproducts #SkincareRoutine #PollutionDefense #CoffeeLover"},
                                    {"role": "user", "content" : f"Write a Best caption for the image with 'a close up of a tube of face wash with coffee beans and coffee beans' image description and Gentle cleansing even for the toughest skin! Pollution defense formula Net Controls excess oil mamaearth' Charcoal Wash Face Gharcoal Activated  Coffee ith CONTROL OIL FOR ' aneol DERMATOLOGICALLY Skibhied TESTED #A4BEN& I0o= SsFree packed) (~hen  CONTENT [' text written on the image. and i want to tag 'mamaearth.in' this account as sponsor.And i also want to add 'avalible on amazom.in' as the call to action for by followers."},
                                    {"role":"assistant","content":"✨ Start your day with a refreshing skincare routine! ✨ ☕️ GENTLE CLEANSING even for the toughest skin! ☕️ 🌿 Experience the POWER of Mamaearth's Charcoal Wash Face 🌿 🔥 Charcoal Activated Coffee with CONTROL OIL FOR acne-prone skin! 🔥 Say goodbye to excess oil and embrace a pollution defense formula that truly works. 🌍 ✅ DERMATOLOGICALLY TESTED and safe for all skin types ✅ Tag your skincare buddies who need this flawless cleansing experience! 💚 Sponsored by @mamaearth.in 💫 Available on Amazon.in. Don't miss out on this amazing product! #MamaearthCharcoalWash #flawlessskincare #mamaearthproducts #SkincareRoutine #PollutionDefense #CoffeeLover"},
                                    {"role": "user", "content" :new_prompt}
                            ]
                )
    return completion.choices[0].message.content

async def create_user_normal_prompt(image_caption):
    return f"Write a Best caption for the new image post with '{image_caption}' image description."

async def normal_main_generater(image_caption,open_ai_key):
    new_prompt =await create_user_normal_prompt(image_caption)
    #acess the openai
    openai.api_key = open_ai_key
    completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages = [{"role": "system", "content" : "I went to you to act as an export influencer who writes the best Instagram post caption that helps you to get more likes and help grow followers. You write captions that will be loved by everyone. Here is some rule you need to follow 1. you write captions based on the 'image description' and the 'description provided by the user'.2.write the caption in multiple lines to look better.3.use relevant emojis and hashtags."},
                                {"role": "user", "content" : f"Write a Best caption for the new image post with 'there is a man sitting on the ground with his motorcycle helmet on' image description and with 'the stunning view at Dhanushkodi beach' user description."},
                                {"role":"assistant","content":"Cruising into the sunset, adventuring through the untamed terrains. 🌅🏍️🌊 Captivated by the stunning view at Dhanushkodi beach, where the waves meet the sky, and the wind whispers tales of freedom. 🌊✨ Embracing the tranquility of this coastal paradise, with the helmet as my shield and the open road ahead. Join me on this epic journey, where every moment leads to new horizons. 🌴🚀 #adventuretime #coastalparadise #dhanushkodibeach #bikerlife #freedomseeker #sunsetcruise"},
                                {"role": "user", "content" : f"Write a Best caption for the new image post with 'there is a man sitting on the ground with his motorcycle helmet on' image description."},
                                {"role":"assistant","content":"Adventures await, helmet on, ready to feel the freedom 🏍️✨ Revving up for new experiences and endless roads ahead! #BikerLife #FreedomRider #ReadyToRoam #FeelingTheRush #LetTheJourneyBegin"},
                                {"role": "user", "content" :new_prompt}]
                )
    return completion.choices[0].message.content