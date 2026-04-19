from fastapi import FastAPI
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from newsapi import NewsApiClient
from openai import OpenAI
from voxcpm import VoxCPM
import soundfile as sf
import os 
import torch
import json
load_dotenv()

app=FastAPI()
db=[]
voice=None
@app.get("/")
def healthCheck():
    return{
        "message":"server is running!"
    }
@app.get("/news")
def getNews():
    
    top_headlines = NewsApiClient(os.getenv("NEWS_API")).get_top_headlines(
                                          category='business',
                                          language='en',
                                          country='us')
    for news in top_headlines["articles"]:
        db.append(news)
    return db 

@app.post("/summary")
def newsSum():
    client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGING"))
    response=client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {   "role":"system",
            "content":"""You are an expert news summarizer.

Task:
Read the given list which contain news articles and generate a concise, accurate, and engaging summary.

Instructions:
1. Keep the summarys between 80-120 words.
2. Highlight the main event, key facts, and important outcomes.
3. Maintain a neutral and factual tone.
4. Remove repetition, filler, and unnecessary details.
5. If numbers, dates, or names are important, include them.
6. Make the summarys easy to understand for general readers.
7. Output only the summarys.
8. provide each and every news summarys
9.Format:
[
  {"id":1, "summary":"..."},
  {"id":2, "summary":"..."}
]
"""},{
            "role": "user",
            "content":json.dumps(db)
        }
    ])
    return json.loads(response.choices[0].message.content)

@app.post("/AI_voice")
def txt_voice():
   model = VoxCPM.from_pretrained("openbmb/VoxCPM2")
   wav = model.generate(
    text = " ".join(item["summary"] for item in newsSum()),
)
   sf.write("output.wav", wav, 16000)