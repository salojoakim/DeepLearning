import streamlit as st
import google.generativeai as genai
import fitz
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# ============ INIT ===================
load_dotenv()                       # laddar in .env filen 
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))    # Startar upp Gemini API och använder min nyckel sparad i .env filen

embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # Använder mig av en förtränad språkmodel

# ============ PDF -> CHUNKS ===================
# Min funktion "extract text from pdf" som tar in sökvägen till mina pdfer som ligger i min project directory
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

# Här använder jag mig av funktionen som jag deklarerade ovanför och läser in mina dokument, i detta fall mina 6 lagar inom arbetsrätt
pdf_texts = [
    extract_text_from_pdf("pdfs/AnstallningsskyddLag.pdf"),
    extract_text_from_pdf("pdfs/Arbetstidslag.pdf"),
    extract_text_from_pdf("pdfs/MedbestammandeLag.pdf"),
    extract_text_from_pdf("pdfs/Offentliganstallning.pdf"),
    extract_text_from_pdf("pdfs/RattegangArbetstvist.pdf"),
    extract_text_from_pdf("pdfs/SemesterLag.pdf"),
]

# Kombinerar alla mina 6 lag texter till en lång textsträng för att sedan kunna dela upp den i chunks
all_text = "\n\n".join(pdf_texts)

# Nu skapar jag en funktion för att dela upp hela textsträngen i mindre bitar (max 1000 char) och detta görs pga begränsat minne i Gemini
def split_text_into_chunks(text, max_length=1000):
    chunks = []
    current_chunk = ""

    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(current_chunk) + len(paragraph) < max_length:
            current_chunk += paragraph + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Här använder jag mig av funktionen och applicerar den på hela kombinerade textsträngen
chunks = split_text_into_chunks(all_text)

# ============ EMBEDDINGS & INDEX ===================
# Jag gör om alla chunks till vektorer
embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1]) #skapar ett sökbart index över våra vektorer
index.add(embeddings) #lägger till alla vektorer i indexet

# ============ FUNKTIONER ===================
#Här kopplar jag ihop användarens fråga med lagtexten för att skapa ett relevant svar.
# Jag gör detta genom att skapa en embedding av frågan och söka i FAISS-indexet efter liknande chunkar.

def retrieve_relevant_chunks(query, model, index, chunk_lookup, k=3):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    return [chunk_lookup[i] for i in I[0]]

#Skapar en prompt där de mest relevanta lagtext-chunkarna samt frågan skickas till Gemini.
#Modellen använder detta för att generera ett juridiskt svar baserat på kontext.
def generate_gemini_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
Du är en juridisk assistent med fokus på svensk arbetsrätt.
Använd följande utdrag ur lagtexten för att svara på frågan.

LAGTEXT:
{context}

FRÅGA:
{question}

SVAR:"""
    model = genai.GenerativeModel("gemini-1.5-flash") #Initierar Gemini-modellen för att generera svar
    response = model.generate_content(prompt)
    return response.text

# ============ STREAMLIT UI ===================
#Skapar Streamlit sidan för att kunna visualisera/presentera chatbotten och ge användaren en möjlighet att interagera med botten.
st.set_page_config(page_title="Svensk Juridikbot", layout="centered")
st.title("⚖️ Svensk Juridikbot")
st.markdown("Ställ en fråga relaterad till svensk arbetsrätt:")

user_question = st.text_input("Din fråga", placeholder="Vad gäller vid uppsägning på grund av arbetsbrist?")

if user_question:
    with st.spinner("Hämtar relevant lagtext..."):
        relevant_chunks = retrieve_relevant_chunks(user_question, embedding_model, index, chunks)

    with st.spinner("Genererar svar från Gemini..."):
        response = generate_gemini_answer(user_question, relevant_chunks)

    st.subheader("🧾 Svar")
    st.write(response)

    st.markdown("🔍 *Källor (relevanta utdrag):*")
    for i, c in enumerate(relevant_chunks):
        st.markdown(f"**{i+1}.** {c}")

    # ======= Feedbacksystem ==========
    # Evalueringssystem där användaren kan ange 1-5 beroende på hur bra svaret var. Feedback sparas i .txt fil
    st.markdown("### ⭐ Hur nöjd är du med svaret?")
    rating = st.slider("Betygsätt svaret (1 = dåligt, 5 = utmärkt)", 1, 5, 3)

    if st.button("Skicka feedback"):
        with open("feedback_log.txt", "a", encoding="utf-8") as f:
            f.write(f"Fråga: {user_question}\n")
            f.write(f"Svar: {response}\n")
            f.write(f"Betyg: {rating}\n")
            f.write("-" * 40 + "\n")
        st.success("Tack för din feedback!")
