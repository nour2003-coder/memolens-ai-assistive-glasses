
import json
from pathlib import Path
from uuid import uuid4

import streamlit as st


# ==========================================
# 1. FILES
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PROFILES_PATH = DATA_DIR / "family_profiles.json"
CONFIG_PATH = DATA_DIR / "caregiver_config.json"
STATE_PATH = DATA_DIR / "patient_state.json"

DATA_DIR.mkdir(exist_ok=True)


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


# ==========================================
# 2. LOAD DATA
# ==========================================

profiles = load_json(PROFILES_PATH, {})

default_config = {
    "patient_name": "Our loved one",
    "welcome_message": "You are safe. Your family is here for you.",
    "routines": [
        {
            "id": "water",
            "name": "Drink a glass of water",
            "time": "09:00"
        },
        {
            "id": "breakfast",
            "name": "Have breakfast",
            "time": "09:30"
        },
        {
            "id": "lunch",
            "name": "Have lunch",
            "time": "12:30"
        },
        {
            "id": "walk",
            "name": "Take a short walk",
            "time": "15:00"
        },
        {
            "id": "dinner",
            "name": "Have dinner",
            "time": "19:00"
        }
    ]
}

config = load_json(CONFIG_PATH, default_config)

state = load_json(STATE_PATH, {})


# ==========================================
# 3. PAGE DESIGN
# ==========================================

st.set_page_config(
    page_title="MemoLens - Caregiver",
    page_icon="💜",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #F8F6FC;
}
h1, h2, h3 {
    color: #4E3B7D;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. HEADER
# ==========================================

st.title("💜 MemoLens")
st.subheader("Caregiver Dashboard")

st.write(
    "Manage your loved one's familiar faces, "
    "personalized messages and daily routines."
)


# ==========================================
# 5. PATIENT SETTINGS
# ==========================================

st.header("👤 Patient settings")

with st.form("patient_settings"):

    patient_name = st.text_input(
        "Patient's name",
        value=config.get("patient_name", "")
    )

    welcome_message = st.text_area(
        "Welcome message",
        value=config.get("welcome_message", "")
    )

    save_patient = st.form_submit_button("Save patient settings")

    if save_patient:

        config["patient_name"] = patient_name
        config["welcome_message"] = welcome_message

        save_json(CONFIG_PATH, config)

        st.success("Patient settings saved.")


# ==========================================
# 6. FAMILY MEMBERS
# ==========================================

st.divider()
st.header("❤️ Family members")

if not profiles:

    st.info(
        "No family members registered. "
        "Use the Family Registration application first."
    )

else:

    for name, details in list(profiles.items()):

        with st.expander(f"👤 {name}"):

            relationship = st.text_input(
                "Relationship",
                value=details.get("relationship", ""),
                key=f"relationship_{name}"
            )

            if st.button("Update relationship", key=f"save_{name}"):

                profiles[name]["relationship"] = relationship

                save_json(PROFILES_PATH, profiles)

                st.success("Relationship updated.")

            if st.button("Remove family member", key=f"remove_{name}"):

                # Remove from face recognition database too
                database_path = DATA_DIR / "face_database.npz"

                if database_path.exists():

                    import numpy as np

                    with np.load(database_path) as data:
                        names = data["names"]
                        embeddings = data["embeddings"]

                    keep = names != name

                    np.savez_compressed(
                        database_path,
                        names=names[keep],
                        embeddings=embeddings[keep]
                    )

                del profiles[name]
                save_json(PROFILES_PATH, profiles)

                st.rerun()


# ==========================================
# 7. DAILY ROUTINES
# ==========================================

st.divider()
st.header("🌷 Daily routines")

routines = config.get("routines", [])

for index, routine in enumerate(list(routines)):

    with st.expander(
        f"{routine['time']} — {routine['name']}"
    ):

        new_name = st.text_input(
            "Activity",
            value=routine["name"],
            key=f"routine_name_{routine['id']}"
        )

        new_time = st.time_input(
            "Time",
            value=__import__("datetime").time.fromisoformat(
                routine["time"]
            ),
            key=f"routine_time_{routine['id']}"
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button("Save", key=f"update_{routine['id']}"):

                routines[index]["name"] = new_name
                routines[index]["time"] = new_time.strftime("%H:%M")

                config["routines"] = routines
                save_json(CONFIG_PATH, config)

                st.success("Routine updated.")
                st.rerun()

        with col2:

            if st.button("Delete", key=f"delete_{routine['id']}"):

                config["routines"] = [
                    item for item in routines
                    if item["id"] != routine["id"]
                ]

                save_json(CONFIG_PATH, config)

                st.rerun()


# Add a new routine
st.subheader("Add a new activity")

with st.form("new_routine"):

    activity_name = st.text_input(
        "Activity name",
        placeholder="Example: Drink water"
    )

    activity_time = st.time_input("Reminder time")

    add_routine = st.form_submit_button("Add activity")

    if add_routine:

        if not activity_name.strip():

            st.error("Please enter an activity name.")

        else:

            routines.append({
                "id": str(uuid4()),
                "name": activity_name.strip(),
                "time": activity_time.strftime("%H:%M")
            })

            config["routines"] = routines

            save_json(CONFIG_PATH, config)

            st.success("New activity added.")
            st.rerun()


# ==========================================
# 8. PATIENT STATUS
# ==========================================

st.divider()
st.header("📊 Patient activity")

if state:

    st.write("Activity date:", state.get("date", "Unknown"))

    completed = state.get("completed", [])

    st.write(
        f"Completed activities: {len(completed)}"
    )

    if state.get("help_requested"):

        st.error("⚠️ The patient has requested assistance.")

        st.write(
            "Request time:",
            state.get("help_requested_at", "Not available")
        )

    else:

        st.success("No help request recorded.")

else:

    st.info("No patient activity has been recorded yet.")


st.caption(
    "MemoLens prototype — local demonstration, "
    "not a clinical monitoring or emergency service."
)
