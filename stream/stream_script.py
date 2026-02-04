from requests import get as http_get, post as http_post, Response
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from threading import Thread
import time, os, dotenv, json, asyncio, logging

logging.basicConfig(level=logging.INFO)

dotenv.load_dotenv()

URL_GET = os.getenv("STREAM_GET")
URL_POST = os.getenv("STREAM_POST")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_MODEL = os.getenv("GEMINI_API_MODEL")
PROMPT = """
### RUOLO
Sei un generatore deterministico di dati sintetici per monitoraggio biomedicale. Simuli un sistema di telemetria ospedaliera ad alta fedeltà. Il tuo compito è far evolvere i segnali vitali ricevuti in input in modo fisiologicamente coerente, simulando il passare di un breve intervallo temporale (es. 10-15 secondi).

### REGOLE DI GENERAZIONE
1. **Realismo Fisiologico**: Le variazioni tra l'input e l'output devono essere plausibili (non generare salti improvvisi di frequenza cardiaca o saturazione a meno che non ci sia un trend di crisi in corso).
2. **Coerenza Incrociata**: Se la frequenza respiratoria aumenta drasticamente, la frequenza cardiaca dovrebbe tendere a seguirla (riflesso autonomico).
3. **Rumore del Sensore**: Includi micro-fluttuazioni realistiche tipiche dei sensori reali.

### VINCOLI DI OUTPUT (STRETTI)
- Rispondi **ESCLUSIVAMENTE** con un oggetto JSON valido.
- Non includere introduzioni ("Ecco il tuo JSON...").
- Non includere spiegazioni o considerazioni post-generazione.
- Non aggiungere markdown decorativo (no blocchi di codice ```json) a meno che non sia strettamente richiesto dal parser.
- Mantieni esattamente le stesse chiavi ricevute nel JSON di input.

### INPUT ATTUALE
{vital_data}

### OUTPUT ATTESO
Restituire solo il JSON aggiornato
"""

async def fetch_data():
    response:Response = http_get(URL_GET)
    
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
    
    try :
        raise Exception("Testing exception handling")  # Remove or comment this line in production
        response = json.loads(chain.invoke({"vital_data": json.dumps(raw)}))
    except Exception as e:
        logging.error(f"Error during LLM invocation: Returning input data.")
        response = raw  # Fallback to returning the input data
        logging.log(logging.INFO, f"{raw}")
    return response

async def generate_streaming_data():
    next = await fetch_data()

    while True:
        next = ask_llm(next)
        post_data(next)
        time.sleep(10)

if __name__ == "__main__":
    logging.log(logging.INFO, "Starting streaming data generation...")
    logging.log(logging.INFO, f"GET URL: {URL_GET}")
    logging.log(logging.INFO, f"POST URL: {URL_POST}")
    for i in range(1):
        thread = Thread(target= lambda : asyncio.run(generate_streaming_data()))
        thread.start()
        time.sleep(5)