import google.generativeai as genai
import os

os.environ["API_KEY"] = 'ENTER_YOUR_API_KEY'
genai.configure(api_key=os.environ["API_KEY"])

model = genai.GenerativeModel('gemini-1.5-flash-latest')
response = model.generate_content("President of USA is")
print(response.text)
