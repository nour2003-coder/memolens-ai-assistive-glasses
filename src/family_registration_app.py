
import json
import cv2
import numpy as np
import streamlit as st

from pathlib import Path
from PIL import Image
from insightface.app import FaceAnalysis


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "face_database.npz"
PROFILES_PATH = DATA_DIR / "family_profiles.json"

DATA_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. LOAD INSIGHTFACE
# ==========================================

@st.cache_resource
def load_face_model():

    model = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    model.prepare(ctx_id=-1, det_size=(640, 640))

    return model


# ==========================================
# 3. LOAD EXISTING DATABASE
# ==========================================

def load_database():

    if DATABASE_PATH.exists():

        with np.load(DATABASE_PATH) as data:

            names = data["names"]
            embeddings = data["embeddings"].astype(np.float32)

        return names, embeddings

    return np.array([], dtype=str), np.empty((0, 512), dtype=np.float32)


def load_profiles():

    if PROFILES_PATH.exists():

        with open(PROFILES_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    return {}


# ==========================================
# 4. EXTRACT EMBEDDINGS FROM PHOTOS
# ==========================================

def extract_embeddings(model, uploaded_files):

    valid_embeddings = []
    results = []

    for uploaded_file in uploaded_files:

        try:
            image = Image.open(uploaded_file).convert("RGB")
            image_array = np.array(image)

            frame = cv2.cvtColor(
                image_array,
                cv2.COLOR_RGB2BGR
            )

            faces = model.get(frame)

            if len(faces) == 0:
                results.append(
                    (uploaded_file.name, "No face detected")
                )
                continue

            if len(faces) > 1:
                results.append(
                    (uploaded_file.name, "Multiple faces detected")
                )
                continue

            embedding = faces[0].embedding.astype(np.float32)

            norm = np.linalg.norm(embedding)

            if norm == 0:
                results.append(
                    (uploaded_file.name, "Invalid embedding")
                )
                continue

            embedding /= norm

            valid_embeddings.append(embedding)

            results.append(
                (uploaded_file.name, "Valid")
            )

        except Exception as error:

            results.append(
                (uploaded_file.name, f"Error: {error}")
            )

    return valid_embeddings, results


# ==========================================
# 5. CREATE REPRESENTATIVE FACE PROFILE
# ==========================================

def create_face_profile(embeddings):

    mean_embedding = np.mean(
        embeddings,
        axis=0
    )

    norm = np.linalg.norm(mean_embedding)

    if norm == 0:
        raise ValueError("Could not create a valid face profile.")

    return mean_embedding / norm


# ==========================================
# 6. SAVE FAMILY MEMBER
# ==========================================

def save_family_member(name, relationship, new_embedding):

    names, embeddings = load_database()
    profiles = load_profiles()

    # Replace an existing identity or add a new one
    existing_indices = np.where(names == name)[0]

    if len(existing_indices) > 0:

        index = int(existing_indices[0])
        embeddings[index] = new_embedding

    else:

        names = np.append(names, name)
        embeddings = np.vstack([
            embeddings,
            new_embedding
        ])

    np.savez_compressed(
        DATABASE_PATH,
        names=names,
        embeddings=embeddings
    )

    profiles[name] = {
        "relationship": relationship
    }

    with open(PROFILES_PATH, "w", encoding="utf-8") as file:

        json.dump(
            profiles,
            file,
            ensure_ascii=False,
            indent=4
        )


# ==========================================
# 7. STREAMLIT INTERFACE
# ==========================================

st.set_page_config(
    page_title="MemoLens - Family Setup",
    page_icon="🧠",
    layout="centered"
)

st.title("MemoLens")
st.subheader("Family Registration")

st.write(
    "Register familiar people so MemoLens can recognize "
    "them and provide personalized assistance."
)

st.info(
    "For better recognition, provide 5–10 clear photographs "
    "of each family member."
)

name = st.text_input(
    "Family member's name",
    placeholder="Example: Ahmed"
)

relationship = st.text_input(
    "Relationship with the patient",
    placeholder="Example: Son, daughter, spouse..."
)

uploaded_files = st.file_uploader(
    "Upload photographs",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:

    st.write(f"**{len(uploaded_files)} photographs selected**")

    for uploaded_file in uploaded_files:

        st.image(
            uploaded_file,
            caption=uploaded_file.name,
            width=150
        )


# ==========================================
# 8. PROCESS AND REGISTER
# ==========================================

if st.button("Register family member", type="primary"):

    if not name.strip() or not relationship.strip():

        st.error("Please enter the name and relationship.")

    elif len(uploaded_files) < 3:

        st.error(
            "Please upload at least 3 photographs. "
            "We recommend 5–10."
        )

    else:

        with st.spinner("Analyzing photographs..."):

            model = load_face_model()

            valid_embeddings, results = extract_embeddings(
                model,
                uploaded_files
            )

        st.subheader("Photo analysis")

        for filename, status in results:

            if status == "Valid":
                st.success(f"{filename}: {status}")
            else:
                st.warning(f"{filename}: {status}")

        if len(valid_embeddings) < 3:

            st.error(
                "Registration failed. At least 3 valid "
                "single-face photographs are required."
            )

        else:

            profile = create_face_profile(valid_embeddings)

            save_family_member(
                name.strip(),
                relationship.strip(),
                profile
            )

            st.success(
                f"{name} has been successfully registered "
                f"using {len(valid_embeddings)} valid photographs!"
            )

            st.write(
                "The representative face embedding "
                "has been saved to the MemoLens database."
            )


# ==========================================
# 9. DISPLAY REGISTERED FAMILY MEMBERS
# ==========================================

st.divider()

st.subheader("Registered family members")

names, embeddings = load_database()
profiles = load_profiles()

if len(names) == 0:

    st.write("No family members registered yet.")

else:

    for registered_name in names:

        registered_name = str(registered_name)

        relationship = profiles.get(
            registered_name, {}
        ).get("relationship", "Not specified")

        st.write(
            f"👤 **{registered_name}** — {relationship}"
        )
