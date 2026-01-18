from requests import get as http_get, post as http_post, Response
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from threading import Thread
import time, os, dotenv, json

dotenv.load_dotenv()
URL_GET = os.getenv("STREAM_GET")
URL_POST = os.getenv("STREAM_POST")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_MODEL = os.getenv("GEMINI_API_MODEL")
PROMPT = """
Sei un agente che simula dei sensori di monitoraggio di segnali vitali.
Il tuo compito è quello di generare dei dati realistici di segnali vitali umani.
Il tuo input è un JSON che rappresenta lo stato attuale dei segnali vitali.
Il tuo output deve essere un JSON che rappresenta lo stato aggiornato dei segnali vitali

---

{vital_data}

"""

async def fetch_data():
    response:Response = await http_get(URL_GET)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")
    
    return response.json()

def post_data(data):
    response:Response = http_post(URL_POST, json=data)
    
    if response.status_code != 200:
        raise Exception(f"Failed to post data: {response.status_code}")
    
def ask_llm(raw) -> dict:
    model = ChatGoogleGenerativeAI(model=GEMINI_API_MODEL, temperature=0.7)
    
    prompt = ChatPromptTemplate.from_template(PROMPT)
    chain = prompt | model | StrOutputParser()
    
    return chain.invoke({"vital_data": json.dumps(raw)})

async def generate_streaming_data():
    next = await fetch_data()

    while True:
        next = json.loads(ask_llm(next))
        post_data(next)
        time.sleep(10)

if __name__ == "__main__":
    for i in range(3):
        thread = Thread(target=generate_streaming_data)
        thread.start()