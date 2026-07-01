from deepface import DeepFace
from .models import BlacklistPerson
import os
import numpy as np
import threading

# Global lock for DeepFace to prevent thread-safety issues in Keras/TensorFlow
DEEPFACE_LOCK = threading.Lock()

# In-memory embedding cache
# Format: { img_path: (embedding, last_modified_time) }
EMBEDDING_CACHE = {}

def get_cached_embedding(img_path):
    global EMBEDDING_CACHE
    try:
        mtime = os.path.getmtime(img_path)
    except OSError:
        return None
        
    if img_path in EMBEDDING_CACHE:
        cached_embedding, cached_mtime = EMBEDDING_CACHE[img_path]
        if cached_mtime == mtime:
            return cached_embedding
            
    try:
        with DEEPFACE_LOCK:
            # Double-checked locking: check cache again after acquiring lock
            if img_path in EMBEDDING_CACHE:
                cached_embedding, cached_mtime = EMBEDDING_CACHE[img_path]
                if cached_mtime == mtime:
                    return cached_embedding
                    
            print(f"Computing embedding for {img_path}...")
            objs = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet512",
                detector_backend="opencv",
                enforce_detection=False
            )
            if objs:
                embedding = objs[0]["embedding"]
                EMBEDDING_CACHE[img_path] = (embedding, mtime)
                return embedding
    except Exception as e:
        print(f"Error computing embedding for {img_path}: {e}")
        
    return None

def recognize_face(frame_path):
    from .models import Employee
    log_file = "face_debug.log"
    with open(log_file, "a") as f:
        f.write(f"\n--- recognize_face called with {frame_path} ---\n")
    try:
        # Get query embedding for the crop (already cropped, so use skip)
        try:
            with DEEPFACE_LOCK:
                query_objs = DeepFace.represent(
                    img_path=frame_path,
                    model_name="Facenet512",
                    detector_backend="opencv",
                    enforce_detection=False
                )
            if not query_objs:
                with open(log_file, "a") as f:
                    f.write("No face detected in live crop.\n")
                return "Unknown"
            query_embedding = query_objs[0]["embedding"]
        except Exception as e:
            with open(log_file, "a") as f:
                f.write(f"Error representing live face crop: {e}\n")
            print(f"Error representing live face crop: {e}")
            return "Unknown"
            
        employees = Employee.objects.all()
        with open(log_file, "a") as f:
            f.write(f"Found {employees.count()} employees in database.\n")
            
        best_match = "Unknown"
        min_distance = 1.0
        threshold = 0.50 # Cosine distance threshold for Facenet512 (more lenient matching)
        
        for employee in employees:
            image_fields = [employee.image_front, employee.image_right, employee.image_left, employee.image]
            image_fields.extend([getattr(employee, f'image_{i}') for i in range(4, 11)])
            
            for img_field in image_fields:
                if not img_field or not img_field.name:
                    continue
                    
                img_path = str(img_field.path)
                if not os.path.exists(img_path):
                    with open(log_file, "a") as f:
                        f.write(f"Image path {img_path} does not exist for {employee.name}.\n")
                    continue
                    
                emp_embedding = get_cached_embedding(img_path)
                if emp_embedding is None:
                    continue
                    
                # Cosine distance
                a = np.array(query_embedding)
                b = np.array(emp_embedding)
                distance = 1.0 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                
                with open(log_file, "a") as f:
                    f.write(f"Compared {employee.name} image ({os.path.basename(img_path)}) vs frame crop: Distance={distance:.4f}\n")
                
                if distance < threshold:
                    if distance < min_distance:
                        min_distance = distance
                        best_match = employee.name
                        
        if best_match != "Unknown":
            with open(log_file, "a") as f:
                f.write(f"MATCHED {best_match}! Distance {min_distance:.4f} < {threshold}\n")
            print(f"Matched {best_match} with distance {min_distance:.4f}")
        else:
            with open(log_file, "a") as f:
                f.write("No match found, returning Unknown.\n")
                
        return best_match
    except Exception as e:
        import traceback
        with open(log_file, "a") as f:
            f.write(f"Outer Exception in recognize_face: {e}\n{traceback.format_exc()}\n")
        print("ERROR =", e)
        return "Unknown"
    

def detect_emotion(frame_path):
    try:
        with DEEPFACE_LOCK:
            result = DeepFace.analyze(
                img_path=frame_path,
                actions=['emotion'],
                enforce_detection=False
            )

        if isinstance(result, list):
            return result[0]['dominant_emotion']
        else:
            return result['dominant_emotion']
    except Exception as e:
        print("Emotion Error =", e)
        return "Unknown"
    


def check_blacklist(face_path):
    from .models import BlacklistPerson
    try:
        # Get query embedding for the crop (already cropped, so use skip)
        try:
            with DEEPFACE_LOCK:
                query_objs = DeepFace.represent(
                    img_path=face_path,
                    model_name="Facenet512",
                    detector_backend="opencv",
                    enforce_detection=False
                )
            if not query_objs:
                return None
            query_embedding = query_objs[0]["embedding"]
        except Exception as e:
            print(f"Error representing live face crop for blacklist: {e}")
            return None
            
        blacklist = BlacklistPerson.objects.all()
        best_match = None
        min_distance = 1.0
        threshold = 0.50
        
        for person in blacklist:
            if not person.image or not person.image.name:
                continue
                
            img_path = str(person.image.path)
            if not os.path.exists(img_path):
                print("Blacklist file missing:", img_path)
                continue
                
            blacklist_embedding = get_cached_embedding(img_path)
            if blacklist_embedding is None:
                continue
                
            # Cosine distance
            a = np.array(query_embedding)
            b = np.array(blacklist_embedding)
            distance = 1.0 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            
            print(f"Compared Blacklist {person.name} ({os.path.basename(img_path)}) - Distance: {distance:.4f}")
            
            if distance < threshold:
                if distance < min_distance:
                    min_distance = distance
                    best_match = person.name
                    
        return best_match
    except Exception as e:
        print("BLACKLIST ERROR =", e)
        return None