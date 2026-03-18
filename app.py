import streamlit as st
import pickle
import faiss
import numpy as np
from deepface import DeepFace
import cv2
import os
import tempfile
import time

from config import (
    MATCH_THRESHOLD,
    DATA_FILE,
    INDEX_FILE,
    IMAGES_DIR,
    APP_TITLE,
    PAGE_ICON,
    THUMBNAIL_COLUMNS,
)

# --- 1. CONFIGURATION ---
st.set_page_config(page_title=APP_TITLE, page_icon=PAGE_ICON, layout="wide")
st.title(f"{PAGE_ICON} {APP_TITLE}")
st.caption("Dynamic identity clustering powered by ArcFace + FAISS-HNSW")

# MATCH_THRESHOLD is imported from config.py

# --- 2. DATA MANAGEMENT FUNCTIONS ---

def load_data():
    """Loads the database and fixes paths for the local machine."""
    if not os.path.exists(DATA_FILE) or not os.path.exists(INDEX_FILE):
        return {}, [], None

    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    index = faiss.read_index(INDEX_FILE)
    
    # Path Fixer Logic
    old_paths = data['paths']
    new_paths = []
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    
    # Create images folder if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Fix paths in the list
    for p in old_paths:
        filename = os.path.basename(p)
        parent_folder = os.path.basename(os.path.dirname(p))
        local_path = os.path.join(base_dir, parent_folder, filename)
        new_paths.append(local_path)
        
    # Fix paths in the clusters dict
    new_clusters = {}
    for c_id, p_list in data['clusters'].items():
        fixed_list = []
        for p in p_list:
            fname = os.path.basename(p)
            parent = os.path.basename(os.path.dirname(p))
            fixed_list.append(os.path.join(base_dir, parent, fname))
        new_clusters[c_id] = fixed_list
            
    return new_clusters, new_paths, index

def save_database(clusters, paths, index):
    """Saves the updated clusters, paths, and vector index to disk."""
    with open(DATA_FILE, 'wb') as f:
        pickle.dump({'clusters': clusters, 'paths': paths}, f)
    faiss.write_index(index, INDEX_FILE)

def add_new_face(img_array, embedding, person_name, cluster_id, clusters, paths, index):
    """Adds a new face to the disk, memory, and VectorDB."""
    # 1. Save Image to Disk
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    person_dir = os.path.join(base_dir, person_name.replace(" ", "_"))
    
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
        
    filename = f"{person_name.replace(' ', '_')}_{int(time.time())}.jpg"
    save_path = os.path.join(person_dir, filename)
    
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_bgr)
    
    # 2. Update In-Memory Data
    paths.append(save_path)
    
    if cluster_id in clusters:
        clusters[cluster_id].append(save_path)
    else:
        clusters[cluster_id] = [save_path]
        
    # 3. Update HNSW Index
    vector = np.array([embedding]).astype('float32')
    faiss.normalize_L2(vector)
    index.add(vector)
    
    # 4. Save
    save_database(clusters, paths, index)
    return save_path

# --- 3. LOAD STATE ---
try:
    clusters, image_paths, index = load_data()
    if 'clusters' not in st.session_state:
        st.session_state['clusters'] = clusters
        st.session_state['paths'] = image_paths
        st.session_state['index'] = index
except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

# --- 4. TABS INTERFACE ---
tab1, tab2 = st.tabs(["🔍 Database Explorer", "➕ Add / Search Face"])

# === TAB 1: EXPLORER (With Search) ===
with tab1:
    st.subheader("View Discovered Identities")
    
    curr_clusters = st.session_state['clusters']
    
    # 1. Build the list of all Names and IDs first
    cluster_names = {}
    valid_ids = []
    
    for c_id, imgs in curr_clusters.items():
        if c_id == -1: continue # Skip noise
        if len(imgs) > 0:
            path = imgs[0]
            name = os.path.basename(os.path.dirname(path)).replace("_", " ")
            cluster_names[c_id] = name
            valid_ids.append(c_id)
            
    valid_ids.sort()

    if not valid_ids:
        st.info("Database is empty.")
    else:
        # 2. Search Box Logic
        col_search, col_space = st.columns([1, 2])
        with col_search:
            search_query = st.text_input("🔍 Search by Name or ID", placeholder="Type 'Ratan' or '12'...")

        # 3. Filter the ID list based on Search
        display_ids = []
        if search_query:
            search_lower = search_query.lower()
            for c_id in valid_ids:
                name = cluster_names[c_id].lower()
                # Check if name matches OR if ID matches
                if search_lower in name or search_query == str(c_id):
                    display_ids.append(c_id)
        else:
            display_ids = valid_ids # Show everyone if search is empty

        # 4. Display Results
        if not display_ids:
            st.warning("No matches found.")
        else:
            if search_query:
                st.caption(f"Found {len(display_ids)} matches.")
                
            sel_id = st.selectbox(
                "Select Person", 
                display_ids, 
                format_func=lambda x: f"{cluster_names[x]} (ID: {x})"
            )
            
            if sel_id is not None:
                imgs = curr_clusters[sel_id]
                st.write(f"📂 **{cluster_names[sel_id]}** - {len(imgs)} images")
                
                cols = st.columns(THUMBNAIL_COLUMNS)
                for i, p in enumerate(imgs):
                    if os.path.exists(p):
                        with cols[i % THUMBNAIL_COLUMNS]:
                            st.image(p, use_container_width=True)

# === TAB 2: SMART ADD / SEARCH ===
with tab2:
    st.subheader("Process New Image")
    
    uploaded_file = st.file_uploader("Upload a face", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        col_preview, col_details = st.columns([1, 2])
        with col_preview:
            st.image(img_rgb, caption="Uploaded Image", width=250)
            
        with col_details:
            if st.button("🔍 Analyze Face"):
                with st.spinner("Analyzing..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
                            tf.write(uploaded_file.getvalue())
                            temp_path = tf.name
                            
                        emb_objs = DeepFace.represent(temp_path, model_name='ArcFace', enforce_detection=True)
                        embedding = emb_objs[0]['embedding']
                        os.remove(temp_path)
                        
                        query = np.array([embedding]).astype('float32')
                        faiss.normalize_L2(query)
                        D, I = st.session_state['index'].search(query, k=1)
                        
                        st.session_state['current_embedding'] = embedding
                        st.session_state['current_img'] = img_rgb
                        st.session_state['search_score'] = D[0][0]
                        st.session_state['search_idx'] = I[0][0]
                        
                    except Exception as e:
                        st.error("No face detected! Try a clearer photo.")

        if 'current_embedding' in st.session_state:
            score = st.session_state['search_score']
            idx = st.session_state['search_idx']
            paths_list = st.session_state['paths']
            
            st.divider()
            
            # Identify closest match
            existing_name = "Unknown"
            matched_cluster_id = None
            
            if idx < len(paths_list):
                match_path = paths_list[idx]
                existing_name = os.path.basename(os.path.dirname(match_path)).replace("_", " ")
                for c_id, p_list in st.session_state['clusters'].items():
                    if match_path in p_list:
                        matched_cluster_id = c_id
                        break
            
            # OPTION 1
            st.subheader("Option 1: Add to Existing Person")
            if score > MATCH_THRESHOLD:
                st.success(f"Closest Match: **{existing_name}** (Similarity: {score:.2f})")
            else:
                st.warning(f"Closest Match: **{existing_name}** (Low Similarity: {score:.2f})")
            
            col_opt1_a, col_opt1_b = st.columns([1, 4])
            with col_opt1_a:
                if idx < len(paths_list) and os.path.exists(match_path):
                     st.image(match_path, caption="Database Match", width=100)
            with col_opt1_b:
                st.write(f"Do you want to add this to **{existing_name}**?")
                if st.button(f"➕ Add to '{existing_name}'"):
                    add_new_face(
                        st.session_state['current_img'],
                        st.session_state['current_embedding'],
                        existing_name,
                        matched_cluster_id,
                        st.session_state['clusters'],
                        st.session_state['paths'],
                        st.session_state['index']
                    )
                    st.success(f"Added to {existing_name}!")
                    del st.session_state['current_embedding']
                    time.sleep(1.5)
                    st.rerun()

            st.divider()

            # OPTION 2
            st.subheader("Option 2: Create New Person")
            st.info("If this is a new person, enter their name below.")
            
            col_opt2_a, col_opt2_b = st.columns([3, 1])
            with col_opt2_a:
                new_name_input = st.text_input("Enter Name:", key="new_name_input")
            with col_opt2_b:
                st.write("##")
                if st.button("✨ Create New Cluster"):
                    if new_name_input.strip():
                        if not st.session_state['clusters']:
                            new_id = 0
                        else:
                            new_id = max(st.session_state['clusters'].keys()) + 1
                        
                        add_new_face(
                            st.session_state['current_img'],
                            st.session_state['current_embedding'],
                            new_name_input,
                            new_id,
                            st.session_state['clusters'],
                            st.session_state['paths'],
                            st.session_state['index']
                        )
                        st.success(f"Created {new_name_input}!")
                        del st.session_state['current_embedding']
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.error("Enter a name.")