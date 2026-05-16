from requests import get as http_get, post as http_post, Response
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from threading import Thread
from datetime import datetime
import time, os, dotenv, json, asyncio, logging, random

logging.basicConfig(level=logging.INFO)

dotenv.load_dotenv()

URL_GET = os.getenv("STREAM_GET")
URL_POST = os.getenv("STREAM_POST")
URL_REPORT = os.getenv("STREAM_REPORT")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_MODEL = os.getenv("GEMINI_API_MODEL")

## RUOLO
PROMPT = """Sei un generatore di dati sintetici stocastico per monitoraggio biomedicale. Il tuo compito è far evolvere i segnali vitali ricevuti in input applicando fluttuazioni casuali e bidirezionali (positive o negative) per simulare l'instabilità naturale dei parametri.

## REGOLE DI GENERAZIONE
1. **Variazione Casuale e Bidirezionale**: Per ogni parametro di input, l'algoritmo deve scegliere casualmente se il valore deve **salire o scendere**. Non deve esserci un trend unidirezionale; le statistiche devono poter diminuire tanto quanto aumentare.
2. **Delta di Oscillazione**: Applica uno scostamento casuale (delta) compreso tra l'1% e il 5% del valore originale. Il segno (+ o -) di tale delta deve essere determinato in modo stocastico (lancio di moneta virtuale) per ogni singola chiave del JSON.
3. **Indipendenza dei Parametri**: La variazione di un segnale non deve influenzare gli altri. Se la frequenza cardiaca sale, la pressione può scendere o restare stabile, senza alcuna coerenza clinica.
4. **Limiti di Sicurezza**: Assicurati che i valori non diventino negativi e non superino i limiti fisiologici massimi (es. SpO2 max 100%).

## VINCOLI DI OUTPUT (STRETTI)
- Rispondi ESCLUSIVAMENTE con un oggetto JSON valido.
- Non includere introduzioni, spiegazioni o blocchi di codice markdown.
- Mantieni esattamente le stesse chiavi ricevute nel JSON di input.

## INPUT ATTUALE
{vital_data}

##STORICO PAZIENTE
{report}

## OUTPUT ATTESO
Restituire solo il JSON aggiornato con le variazioni casuali (positive o negative) applicate.
"""
async def fetch_data(url) -> dict:
    response:Response = http_get(url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")
    
    return response.json()

def post_data(data):
    response:Response = http_post(URL_POST, json=data)
    
    if response.status_code != 200:
        raise Exception(f"Failed to post data: {response.status_code}")
    
def ask_llm(raw,report) -> dict:
    model = ChatGoogleGenerativeAI(model=GEMINI_API_MODEL, temperature=0.7)
    
    prompt = ChatPromptTemplate.from_template(PROMPT)
    chain = prompt | model | JsonOutputParser()
    
    try :
        error = ""
        response = chain.invoke({"vital_data": json.dumps(raw), "report": report})
    except Exception as e:
        error = str(e)
        response = raw  
    logging.info(f"--- BATCH GENERATED ---")
    logging.info(f"Out ({error}): \n{response}")
    response['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    return response

async def generate_streaming_data():
    next = await fetch_data(url=URL_GET)

    while True:
        report = await fetch_data(url=URL_REPORT+f"/{next['Patient ID']}")
        next = ask_llm(next, report['history_report'] if 'history_report' in report else "No history available")
        try:
            post_data(next)
        except Exception as e:
            logging.error(f"Failed to post data: {e}")
        time.sleep(10)
    
if __name__ == "__main__":
    logging.log(logging.INFO, "Starting streaming data generation...")
    for i in range(int(os.getenv("N_JOBS"))):
        thread = Thread(target = lambda : asyncio.run(generate_streaming_data()))
        thread.start()
        time.sleep(5)