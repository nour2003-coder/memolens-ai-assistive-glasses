
import json
from pathlib import Path
from datetime import date

import streamlit as st


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

PROFILES_PATH = BASE_DIR / "data" / "family_profiles.json"
CONFIG_PATH = BASE_DIR / "data" / "caregiver_config.json"
STATE_PATH = BASE_DIR / "data" / "patient_state.json"

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. FILE HELPERS
# ==========================================

def load_json(path, default):
    if not path.exists():
        return default

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


# Default routines for the first launch
default_config = {
    "patient_name": "Our loved one",
    "welcome_message": "You are safe. Your family is here for you.",
    "routines": [
        {"id": "water", "name": "Drink a glass of water", "time": "09:00"},
        {"id": "breakfast", "name": "Have breakfast", "time": "09:30"},
        {"id": "lunch", "name": "Have lunch", "time": "12:30"},
        {"id": "walk", "name": "Take a short walk", "time": "15:00"},
        {"id": "dinner", "name": "Have dinner", "time": "19:00"}
    ]
}

config = load_json(CONFIG_PATH, default_config)
profiles = load_json(PROFILES_PATH, {})


# ==========================================
# 3. DAILY PROGRESS
# ==========================================

today = str(date.today())

state = load_json(STATE_PATH, {})

if state.get("date") != today:
    state = {
        "date": today,
        "completed": [],
        "help_requested": False
    }
    save_json(STATE_PATH, state)


# ==========================================
# 4. PAGE DESIGN
# ==========================================

st.set_page_config(
    page_title="MemoLens - Patient",
    page_icon="💜",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background-color: #F8F6FC;
}
h1, h2, h3 {
    color: #4E3B7D;
}
.patient-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    margin-bottom: 15px;
    border: 1px solid #E8E1F3;
}
.big-text {
    font-size: 24px;
    font-weight: 600;
    color: #4E3B7D;
}
.small-text {
    font-size: 17px;
    color: #625A70;
}
div.stButton > button {
    background-color: #715EA9;
    color: white;
    border-radius: 12px;
    min-height: 55px;
    font-size: 18px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 5. WELCOME
# ==========================================

st.title("💜 MemoLens")

st.markdown(
    f"<div class='big-text'>Bonjour, "
    f"{config.get('patient_name', 'Bienvenue')} !</div>",
    unsafe_allow_html=True
)

st.write(config.get("welcome_message", default_config["welcome_message"]))

st.caption(date.today().strftime("%A %d %B %Y"))


# ==========================================
# 6. DAILY ROUTINE
# ==========================================

st.divider()
st.header("🌷 Ma journée")

routines = config.get("routines", [])
completed = state.get("completed", [])

if not routines:
    st.info("Aucune activité n'a été programmée.")

for routine in routines:

    routine_id = routine["id"]
    is_completed = routine_id in completed

    st.markdown(
        f"<div class='patient-card'>"
        f"<div class='big-text'>{routine['name']}</div>"
        f"<div class='small-text'>🕒 {routine['time']}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    checked = st.checkbox(
        "J'ai terminé cette activité",
        value=is_completed,
        key=f"routine_{routine_id}"
    )

    if checked and not is_completed:
        completed.append(routine_id)
        state["completed"] = completed
        save_json(STATE_PATH, state)
        st.rerun()

    elif not checked and is_completed:
        completed.remove(routine_id)
        state["completed"] = completed
        save_json(STATE_PATH, state)
        st.rerun()


# ==========================================
# 7. FAMILY MEMBERS
# ==========================================

st.divider()
st.header("❤️ Ma famille")

if not profiles:

    st.info("Votre famille sera affichée ici après son inscription.")

else:

    for name, details in profiles.items():

        relationship = details.get(
            "relationship",
            "Membre de la famille"
        )

        st.markdown(
            f"<div class='patient-card'>"
            f"<div class='big-text'>👤 {name}</div>"
            f"<div class='small-text'>{relationship}</div>"
            f"</div>",
            unsafe_allow_html=True
        )


# ==========================================
# 8. ASK FOR HELP
# ==========================================

st.divider()
st.header("🤝 Besoin d'aide ?")

st.write(
    "Si vous avez besoin de parler à votre famille, "
    "vous pouvez demander de l'aide."
)

if st.button("💜 Je veux de l'aide"):

    state["help_requested"] = True
    state["help_requested_at"] = str(
        __import__("datetime").datetime.now()
    )

    save_json(STATE_PATH, state)

    st.success(
        "Votre demande a été enregistrée. "
        "Votre proche aidant pourra la consulter."
    )


# ==========================================
# 9. DAILY PROGRESS
# ==========================================

st.divider()
st.header("🌟 Bravo !")

total = len(routines)
done = len(completed)

if total:
    st.progress(min(done / total, 1.0))
    st.write(f"{done} activité(s) terminée(s) sur {total}.")

if done == total and total > 0:
    st.success("Bravo ! Vous avez terminé toutes vos activités.")
