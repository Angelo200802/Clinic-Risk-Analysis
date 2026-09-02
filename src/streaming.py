from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from pyspark.sql.types import Row, StructType, StructField,IntegerType, DoubleType, StringType
from pyspark.sql import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import col, window
from redis import Redis
from fastapi import APIRouter
from spark_manager import load_dataset, get_session
import os, logging
from pyspark.sql import DataFrame
from model.ensemble import Ensemble
from pydantic import BaseModel, Field
from dotenv import load_dotenv 
from fastapi import WebSocket
import bus, json
 
load_dotenv()
logging.basicConfig(level=logging.INFO)
GEMINI_API_MODEL = os.getenv("GEMINI_API_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
 
redis_db = Redis(host=os.getenv("REDIS_HOST"), port=int(os.getenv("REDIS_PORT")), db=0)
ensemble = Ensemble()
router_streaming = APIRouter()
 
 
columns = [
    "Heart Rate",
    "Respiratory Rate",
    "Body Temperature",
    "Oxygen Saturation",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Age",
    "Gender",
    "Weight (kg)",
    "Height (m)",
    "Derived_HRV",
    "Derived_Pulse_Pressure",
    "Derived_BMI",
    "Derived_MAP",
    "Prediction"
]
 
class VitalSigns(BaseModel):
    patient_id : int = Field(alias="Patient ID")
    heart_rate : int = Field(alias="Heart Rate")
    respiratory_rate : int = Field(alias="Respiratory Rate")
    timestamp : str = Field(alias="Timestamp")
    body_temperature : float = Field(alias="Body Temperature")
    oxygen_saturation : float = Field(alias="Oxygen Saturation")
    systolic_blood_pressure : float = Field(alias="Systolic Blood Pressure")
    diastolic_blood_pressure : float = Field(alias="Diastolic Blood Pressure")
    age : int = Field(alias="Age")
    gender : str = Field(alias="Gender")
    weight_kg : float = Field(alias="Weight (kg)")
    height_m : float = Field(alias="Height (m)")
    derived_hrv : float = Field(alias="Derived_HRV")
    derived_pulse_pressure : float = Field(alias="Derived_Pulse_Pressure")
    derived_bmi : float = Field(alias="Derived_BMI")
    derived_map : float = Field(alias="Derived_MAP")
    risk_category : str = Field(default= None, alias="Risk Category")
    
df = load_dataset(os.getenv("DATASET_PATH"))
 
schema = StructType([
    StructField("Patient ID", IntegerType(), True),
    StructField("Heart Rate", IntegerType(), True),
    StructField("Respiratory Rate", IntegerType(), True),
    StructField("Timestamp", StringType(), True),
    StructField("Body Temperature", DoubleType(), True),
    StructField("Oxygen Saturation", DoubleType(), True),
    StructField("Systolic Blood Pressure", DoubleType(), True),
    StructField("Diastolic Blood Pressure", DoubleType(), True),
    StructField("Age", IntegerType(), True),
    StructField("Gender", StringType(), True),
    StructField("Weight (kg)", DoubleType(), True),
    StructField("Height (m)", DoubleType(), True),
    StructField("Derived_HRV", DoubleType(), True),
    StructField("Derived_Pulse_Pressure", DoubleType(), True),
    StructField("Derived_BMI", DoubleType(), True),
    StructField("Derived_MAP", DoubleType(), True),
    StructField("Risk Category", StringType(), True)
])
 
def save_to_redis(row : dict):
    patient_id = row['sensor_update']['Patient ID']
    history_key = f"patient_history:{patient_id}"
    logging.info(f"Salvando dati per Patient ID {patient_id} su Redis...")
    payload = json.dumps(row,default=str)
    pipe = redis_db.pipeline()
    pipe.rpush(history_key, payload)
    pipe.ltrim(history_key, 0, 10)
    pipe.execute()
    logging.info(f"Dati salvati per Patient ID {patient_id} su Redis.")
 
def send_update(update_row: Row):
    dict_row = update_row.asDict()
    dict_row['start'] = dict_row['window'].start.isoformat()
    dict_row['end'] = dict_row['window'].end.isoformat()    
    dict_row.pop('window')
    dict_row['Timestamp'] = dict_row.pop('Timestamp').isoformat() if dict_row['Timestamp'] else ""
    update_row = {
                "type" : "update",
                "sensor_update" : {
                    k : v for k, v in dict_row.items() if k in columns
                },
                "trend_update" : {
                    k : v for k, v in dict_row.items() 
                    if k not in columns 
                    and 'prev' not in k 
                    and 'index' not in k
                    and 'pattern' not in k
                    and 'deteriotation' not in k
                    and 'rate_pp' not in k
                    and 'delta' not in k
                } ,
                "index" : {
                    k : v for k, v in dict_row.items() 
                    if 'index' in k 
                    or 'rate_pp' in k
                } ,
                "pattern" :{
                    k : v for k, v in dict_row.items() 
                    if 'pattern' in k or 'deterioration' in k
                }
            }
    update_row['sensor_update']['Patient ID'] = dict_row['Patient ID']
    update_row['sensor_update']['Timestamp'] = dict_row['Timestamp']
    bus.main_loop.call_soon_threadsafe(
        bus.data_queue.put_nowait, 
        update_row
    )
    logging.info(f"Trend calcolato per Patient ID {update_row['sensor_update']['Patient ID']}\n {update_row}")
    return update_row
 
 
TREND_HISTORY_PREFIX = "patient_trend_buffer"
MAX_HISTORY = 3 
 
HR_DELTA_MIN = 2.0
MAP_DELTA_MIN = 2.0
RR_DELTA_MIN = 1.0
SPO2_DELTA_MIN = 0.5
HRV_DELTA_MIN = 0.005
 
 
def _load_trend_buffer(patient_id: int) -> list:
    raw = redis_db.get(f"{TREND_HISTORY_PREFIX}:{patient_id}")
    return json.loads(raw) if raw else []
 
 
def _save_trend_buffer(patient_id: int, buffer: list):
    redis_db.set(f"{TREND_HISTORY_PREFIX}:{patient_id}", json.dumps(buffer))
 
 
def _hist(buffer: list, field: str, n: int):
    """n=1 -> valore al passo precedente, n=2 -> due passi indietro."""
    idx = len(buffer) - 1 - n
    return buffer[idx][field] if idx >= 0 else None
 
 
def _pct(curr, prev):
    if curr is None or prev is None or prev == 0:
        return None
    return (curr - prev) / prev * 100
 
 
def _rising(curr, prev, prev_lag, min_delta):
    return (
        prev is not None and prev_lag is not None and
        (curr - prev) > min_delta and (prev - prev_lag) > min_delta
    )
 
 
def _falling(curr, prev, prev_lag, min_delta):
    return (
        prev is not None and prev_lag is not None and
        (prev - curr) > min_delta and (prev_lag - prev) > min_delta
    )
 
 
def compute_trend_row(row: Row) -> dict:
    """
    Calcola pct/pattern per una singola riga aggregata (Patient ID + finestra),
    usando lo storico del paziente letto da Redis invece che un Window().lag()
    limitato al solo micro-batch corrente.
    """
    patient_id = row["Patient ID"]
    w_start = row["window"].start.isoformat()
 
    buffer = _load_trend_buffer(patient_id)
 
    point = {
        "window_start": w_start,
        "avg_hr": float(row["avg_hr"]),
        "avg_map": float(row["avg_map"]),
        "avg_rr": float(row["avg_rr"]),
        "avg_spo2": float(row["avg_spo2"]),
        "avg_hrv": float(row["avg_hrv"]),
        "avg_pp": float(row["avg_pp"]),
    }
 
    if buffer and buffer[-1]["window_start"] == w_start:
        buffer[-1] = point
    else:
        buffer.append(point)
        buffer = buffer[-MAX_HISTORY:]
 
    prev_hr, prev_hr_lag = _hist(buffer, "avg_hr", 1), _hist(buffer, "avg_hr", 2)
    prev_map, prev_map_lag = _hist(buffer, "avg_map", 1), _hist(buffer, "avg_map", 2)
    prev_rr, prev_rr_lag = _hist(buffer, "avg_rr", 1), _hist(buffer, "avg_rr", 2)
    prev_spo2, prev_spo2_lag = _hist(buffer, "avg_spo2", 1), _hist(buffer, "avg_spo2", 2)
    prev_hrv, prev_hrv_lag = _hist(buffer, "avg_hrv", 1), _hist(buffer, "avg_hrv", 2)
    prev_pp = _hist(buffer, "avg_pp", 1)
 
    avg_hr, avg_map = point["avg_hr"], point["avg_map"]
    avg_rr, avg_spo2 = point["avg_rr"], point["avg_spo2"]
    avg_hrv, avg_pp = point["avg_hrv"], point["avg_pp"]
    avg_temp = row["avg_temp"]
 
    hemo_pattern = int(
        _rising(avg_hr, prev_hr, prev_hr_lag, HR_DELTA_MIN) and
        _falling(avg_map, prev_map, prev_map_lag, MAP_DELTA_MIN)
    )
    resp_pattern = int(
        _rising(avg_rr, prev_rr, prev_rr_lag, RR_DELTA_MIN) and
        _falling(avg_spo2, prev_spo2, prev_spo2_lag, SPO2_DELTA_MIN)
    )
    sepsis_pattern = int(
        avg_temp is not None and avg_temp > 38 and
        _rising(avg_hr, prev_hr, prev_hr_lag, HR_DELTA_MIN) and
        _falling(avg_hrv, prev_hrv, prev_hrv_lag, HRV_DELTA_MIN)
    )
 
    bmi = row["Derived_BMI"]
    bmi_class = (
        "UNDERWEIGHT" if bmi < 18.5 else
        "NORMAL" if bmi < 25 else
        "OVERWEIGHT" if bmi < 30 else
        "OBESE"
    )
 
    sbp, dbp = row["avg_sbp"], row["avg_dbp"]
    shock_index = avg_hr / sbp if sbp else None
    modified_shock_index = avg_hr / avg_map if avg_map else None
    age_index = row["Age"] * shock_index if shock_index is not None else None
    diastolic_shock_index = avg_hr / dbp if dbp else None
    rate_pp = sbp * avg_hr if sbp else None
    pp_index = avg_pp / avg_hr if avg_hr else None
 
    dict_row = row.asDict()
    dict_row.update({
        "bmi_class": bmi_class,
        "hr_pct": _pct(avg_hr, prev_hr),
        "rr_pct": _pct(avg_rr, prev_rr),
        "spo2_pct": _pct(avg_spo2, prev_spo2),
        "pp_pct": _pct(avg_pp, prev_pp),
        "map_pct": _pct(avg_map, prev_map),
        "progressive_hemo_deterioration": hemo_pattern,
        "progressive_resp_failure_pattern": resp_pattern,
        "dynamic_sepsis_pattern": sepsis_pattern,
        "shock_index": shock_index,
        "modified_shock_index": modified_shock_index,
        "age_index": age_index,
        "diastolic_shock_index": diastolic_shock_index,
        "rate_pp": rate_pp,
        "pp_index": pp_index,
    })
 
    _save_trend_buffer(patient_id, buffer)
 
    return dict_row
 
 
def batch_job_stats(df_stats: DataFrame, batch_id):
    df_stats.show()
    count = df_stats.count()
    if count > 0:
        rows = df_stats.collect()
        # Ordina per paziente e poi per inizio finestra: un batch puo'
        # contenere piu' righe per lo stesso paziente (finestre sovrapposte),
        # e vanno processate in ordine cronologico per aggiornare
        # correttamente il buffer di storico su Redis.
        rows_sorted = sorted(rows, key=lambda r: (r["Patient ID"], r["window"].start))
 
        for row in rows_sorted:
            dict_row = compute_trend_row(row)
            updated_row = send_update(Row(**dict_row))
            save_to_redis(updated_row)
 
        logging.info(f"--- ANALISI TREND BATCH {batch_id} (Redis history) ---")
 
def start_streaming():
 
    df_stats_raw = (get_session().readStream 
        .format("redis")
        .option("redis.host", os.getenv("REDIS_HOST","redis"))  
        .option("redis.port", os.getenv("REDIS_PORT", "6379")) 
        .option("stream.keys", "vital_signs")
        .option("stream.read.batch.size", "50") 
        .option("stream.group.name", "spark-statistics")  
        .schema(schema)
        .load())
    
    df_stats_raw = df_stats_raw.withColumn("Timestamp", col("Timestamp").cast("timestamp"))
    df_stats_raw = ensemble.classify(df_stats_raw)
 
    df_windowed = (
        df_stats_raw
        .withWatermark("Timestamp", "1 minute")
        .groupBy(
            window(col("Timestamp"), "1 minute", "30 seconds"),
            col("Patient ID")
        )
        .agg(
            F.max_by("Heart Rate","Timestamp").alias("Heart Rate"),
            F.max_by("Respiratory Rate","Timestamp").alias("Respiratory Rate"),
            F.max_by("Oxygen Saturation","Timestamp").alias("Oxygen Saturation"),
            F.max_by("Systolic Blood Pressure","Timestamp").alias("Systolic Blood Pressure"),
            F.max_by("Diastolic Blood Pressure","Timestamp").alias("Diastolic Blood Pressure"),
            F.max_by("Body Temperature","Timestamp").alias("Body Temperature"),
            F.max_by("Age","Timestamp").alias("Age"),
            F.max_by("Gender", "Timestamp").alias("Gender"),
            F.max_by("Weight (kg)", "Timestamp").alias("Weight (kg)"),
            F.max_by("Height (m)", "Timestamp").alias("Height (m)"),
            F.max_by("Derived_MAP", "Timestamp").alias("Derived_MAP"),
            F.max_by("Derived_HRV", "Timestamp").alias("Derived_HRV"),
            F.max_by("Derived_BMI", "Timestamp").alias("Derived_BMI"),
            F.max_by("Derived_Pulse_Pressure", "Timestamp").alias("Derived_Pulse_Pressure"), 
            F.max("Timestamp").alias("Timestamp"),
            F.max_by("Prediction", "Timestamp").alias("Prediction"),
            F.avg(
                F.when(F.lower(F.col("Prediction")) == "high risk", 1).otherwise(0)
            ).alias("risk_ratio"),
            F.avg("Heart Rate").alias("avg_hr"),
            F.avg("Systolic Blood Pressure").alias("avg_sbp"),
            F.avg("Diastolic Blood Pressure").alias("avg_dbp"),
            F.avg("Respiratory Rate").alias("avg_rr"),
            F.avg("Oxygen Saturation").alias("avg_spo2"),
            F.avg("Body Temperature").alias("avg_temp"),
            F.avg("Derived_MAP").alias("avg_map"),
            F.avg("Derived_Pulse_Pressure").alias("avg_pp"),
            F.avg("Derived_HRV").alias("avg_hrv"),
            F.stddev("Heart Rate").alias("std_hr"),
            F.count("*").alias("n_samples")
        )
        )
 
    query_stats = (df_windowed.writeStream
        .foreachBatch(batch_job_stats)
        .outputMode("update")
        .option("truncate", "false")
        .start()
    )
 
    return [query_stats]


@router_streaming.get("/getseed")
def get_seed():
    seed = df.rdd.takeSample(False, 1)[0].asDict()
    ret = {}
    for key in seed:
        if key not in ['Risk Category','pred_logistic_regression','pred_mlp', 'pred_naive_bayes', 'weighted_score', 'Prediction'] : 
            ret[key] = seed[key]
    return ret

@router_streaming.post("/newraw")
def new_raw(raw: VitalSigns):
    raw_dict = raw.model_dump(by_alias=True, exclude_none=False)
    clean_data = {k: (v if v is not None else "") for k, v in raw_dict.items()}
    redis_db.xadd('vital_signs', clean_data)
    
    return {"status": "ok"}

@router_streaming.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("WebSocket connection accepted.")
    
    while not bus.data_queue.empty():
        bus.data_queue.get_nowait()
    
    try:
        while True:
            data = await bus.data_queue.get()
            logging.info(f"Sending to WebSocket: type={data.get('type')}, pid={data.get('patient_id')}")
            await websocket.send_json(data)
            logging.info(f"Sent OK: type={data.get('type')}")
    except Exception as e:
        logging.info("WebSocket connection closed.")
    finally:
        logging.info("WebSocket finally — svuoto queue")
        while not bus.data_queue.empty():
            bus.data_queue.get_nowait()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def ask_llm(prompt:str) -> str:
    model = ChatGoogleGenerativeAI(
        model=GEMINI_API_MODEL, 
        temperature=0.4,
        api_key=GEMINI_API_KEY
    )
    
    prompt = ChatPromptTemplate.from_template(prompt)
    chain = prompt | model | StrOutputParser()

    return chain.invoke({})

async def run_llm_inference(patient_id, history):
    current = history[0]
    
    gender = current['sensor_update']['Gender']
    age = current['sensor_update']['Age']
    bmi = current['sensor_update']['Derived_BMI']
    bmi_class = current['trend_update']['bmi_class']
    prediction = current['sensor_update']['Prediction']
    hr = current['sensor_update']['Heart Rate']
    hr_pct = current['trend_update']['hr_pct']
    sbp = current['sensor_update']['Systolic Blood Pressure']
    dbp = current['sensor_update']['Diastolic Blood Pressure']
    map_pct = current['trend_update']['map_pct']
    spo2 = current['sensor_update']['Oxygen Saturation']
    shock_idx = current['index']['shock_index']
    
    patient_report = generate_report(history[1:-1], analysis=True)

    prompt = f"""
    ##RUOLO
    Sei un Assistente Avanzato di Supporto alle Decisioni Cliniche (CDSS) specializzato in monitoraggio emodinamico e terapia intensiva.

    ##SCOPO
    Analizzare lo stato attuale del paziente e i trend calcolati per spiegare la classificazione di rischio e guidare il medico nell'identificare precocemente un deterioramento clinico.
    
    ## CONTESTO CLINICO
    Paziente ID: {patient_id} | Genere: {gender} | Età: {age} | BMI: {bmi:.1f} ({bmi_class})

    ## DATI ATTUALI E TREND (Ultima finestra di osservazione)
    - Stato Rischio: {prediction}
    - Frequenza Cardiaca: {hr} bpm (Variazione: {hr_pct}%)
    - Pressione: {sbp}/{dbp} mmHg (MAP: {map_pct}% variazione)
    - Saturazione Ossigeno: {spo2}%
    - Shock Index: {shock_idx:.2f} (Valore normale: 0.5-0.7)

    ## STORICO PAZIENTE
    {patient_report}

    ## ISTRUZIONI DI ANALISI
    1. SPIEGAZIONE: Analizza la coerenza tra i pattern rilevati e i segni vitali. Perché il rischio è {prediction}?
    2. CRITICITÀ: Identifica il parametro che richiede attenzione immediata (es. Shock Index elevato o calo della MAP).
    3. AZIONE: Suggerisci 2-3 step clinici basati su protocolli standard (es. ACLS, SIRS).

    ## REGOLE DI OUTPUT (FORMATTAZIONE)
    - Sii estremamente conciso (max 100 parole).
    - Usa un tono professionale e analitico.
    - Usa il formato markdown per evidenziare i parametri e le parole chiave (es. **Shock Index: 1.2**).
    - Struttura la risposta in tre brevi paragrafi: **Analisi**, **Punto Critico**, **Suggerimenti**.
    """
    
    ai_response = ask_llm(prompt)
    
    logging.info(f"LLM response ready, length: {len(ai_response)}")
    logging.info(f"Queue size before publish: {bus.data_queue.qsize()}")
    logging.info(f"Queue empty: {bus.data_queue.empty()}")
    
    return ai_response
    

@router_streaming.get("/explain/{patient_id}")
async def get_ai_explanation(patient_id: int):
    history_raw = redis_db.lrange(f"patient_history:{patient_id}", 0, -1)
    
    if not history_raw:
        return {"status": "error", "message": "Dati non trovati per questo paziente"}

    history = [json.loads(h) for h in history_raw]
    ai_response = await run_llm_inference(patient_id, history)
    
    return {"status": "ok", "message": ai_response, "patient_id": patient_id}

def generate_report(history: list,analysis:bool = False) -> str:
    report_steps = []
    
    for i, entry in enumerate(history[0:min(5,len(history))]):
        s = entry["sensor_update"]
        t = entry["trend_update"]
        index = entry["index"]
        
        # 1. Sezione Sensori: Valori attuali e derivati calcolati
        sensors_block = [
            f"   [SENSORS] HR: {s['Heart Rate']} bpm, RR: {s['Respiratory Rate']} resp/min, SpO2: {s['Oxygen Saturation']}%",
            f"   [PRESS.] SBP: {s['Systolic Blood Pressure']} mmHg, DBP: {s['Diastolic Blood Pressure']} mmHg",
            f"   [DERIVED] MAP: {s.get('Derived_MAP', 0):.2f}, PP: {s.get('Derived_Pulse_Pressure', 0)}, HRV: {s.get('Derived_HRV', 0)}"
        ]
        if analysis:
            # 2. Sezione Trend: Medie mobili e statistiche di rischio
            trends_block = [
                f"   [AVG_MOVING] Avg_HR: {t['avg_hr']}, Avg_SBP: {t['avg_sbp']}, Avg_DBP: {t['avg_dbp']} Avg_RR: {t['avg_rr']}, Avg_SpO2: {t['avg_spo2']}",
                f"   [STATS] Risk_Ratio: {t['risk_ratio']}, Samples: {t['n_samples']}, Hemo_Deterioration: {t['progressive_hemo_deterioration']}"
            ]
            trends_block.append(f"   [INDEX] Shock Index: {index['shock_index']:.2f}, Modified Shock Index: {index['modified_shock_index']:.2f}, Age Index: {index['age_index']:.2f} Diastolic Shock Index: {index['diastolic_shock_index']:.2f}, PP Index: {index['pp_index']:.2f}")
        
        # Unione delle sezioni per il singolo step
        step_header = f"STEP {i} | TIMESTAMP: {s['Timestamp']} | STATUS: {s['Prediction']}"
        full_step = [step_header] + sensors_block + (trends_block if analysis else [])
        
        report_steps.append("\n".join(full_step))

    # Join finale: separa ogni blocco temporale con una linea per massima chiarezza
    divider = "\n" + "-"*50 + "\n"
    return divider.join(report_steps)

@router_streaming.get("/history_report/{patient_id}")
async def get_patient_history_report(patient_id: int):
    history_raw = redis_db.lrange(f"patient_history:{patient_id}", 0, -1)
    
    if not history_raw:
        return {"status": "error", "message": "Dati non trovati per questo paziente"}

    history = [json.loads(h) for h in history_raw]
    return {"status": "ok", "history_report": generate_report(history), "patient_id": patient_id}