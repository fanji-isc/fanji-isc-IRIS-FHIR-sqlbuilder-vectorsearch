from __future__ import annotations
import os, json
from typing import List, Dict
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import text
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def send_to_llm(messages, **kwargs):
    completion = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        messages=messages,
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
        **kwargs
    )
    return completion


def llm_answer_rag(batch, query, cutoff=True):
    prompt_text = """You are a medical assistant that answers questions
    about patients using database records.
    Use ONLY the given patient data below.
    """ + (("Use three sentences maximum and keep the answer concise.") if cutoff else "") + """
    Question: {question}
    Context: {context}
    Answer:
    """

    prompt = prompt_text.format(question=query, context=batch)
    messages = [{"role": "user", "content": prompt}]
    completion = send_to_llm(messages)
    response = completion.choices[0].message.content
    answer_lines = [line.strip() for line in response.split("\n") if line.strip()]
    return "\n".join(answer_lines)


def vector_patient_search(
    engine,
    model,
    query_text: str,
    *,
    schema: str = "SQL1",
    table: str = "patient_info",
    vector_col: str = "patient_vector",
    display_columns: list[str] | None = None,
    top_k: int = 5,
) -> List[Dict]:
    if not query_text:
        return []

    vec = model.encode(query_text, normalize_embeddings=True).tolist()
    sql = text(f"""
        SELECT TOP {top_k} *,
               VECTOR_DOT_PRODUCT({vector_col}, TO_VECTOR(:vec)) AS score
        FROM {schema}.{table}
        ORDER BY score DESC
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, {"vec": json.dumps(vec)}).fetchall()

    if not rows:
        return []

    df = pd.DataFrame([dict(r._mapping) for r in rows])
    if display_columns is None:
        display_columns = []
    cols = [c for c in display_columns if c in df.columns]
    if "score" not in cols:
        cols.append("score")
    return df[cols].to_dict(orient="records")


SAFE_FIELDS = ["Name", "DOB", "Medication", "Allergies", "FamilyHistory", "City", "State"]

def _rows_to_batch(rows: List[Dict], safe_fields=SAFE_FIELDS) -> str:
    if not rows:
        return "(no matches)"
    lines = []
    for i, r in enumerate(rows, start=1):
        parts = []
        for f in safe_fields:
            if f in r and r[f]:
                val = r[f]
                if f == "DOB" and isinstance(val, str) and len(val) >= 4:
                    val = val[:4]
                parts.append(f"{f}: {val}")
        score = r.get("score", 0)
        try:
            score = round(float(score), 4)
        except Exception:
            pass
        if parts:
            lines.append(f"[Match {i} | score={score}] " + "; ".join(parts))
    return "\n".join(lines)


def chat_from_query_using_rag(
    engine,
    model,
    user_question: str,
    *,
    schema: str = "SQL1",
    table: str = "patient_info",
    vector_col: str = "patient_vector",
    display_columns: list[str] | None = None,
    top_k: int = 5,
    cutoff: bool = True,
) -> dict:
    rows = vector_patient_search(
        engine=engine,
        model=model,
        query_text=user_question,
        schema=schema,
        table=table,
        vector_col=vector_col,
        display_columns=display_columns or [],
        top_k=top_k,
    )

    batch_text = _rows_to_batch(rows)
    answer = llm_answer_rag(batch=batch_text, query=user_question, cutoff=cutoff)
    return {"rows": rows, "answer": answer}
