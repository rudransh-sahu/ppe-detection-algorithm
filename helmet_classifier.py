# helmet_classifier.py
import threading
import queue
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

class HelmetClassifier:
    """
    Zero‑shot helmet type classifier using SmolVLM2-256M.
    Runs in a background thread – never blocks the main pipeline.
    """
    def __init__(self, model_id="HuggingFaceTB/SmolVLM2-256M-Instruct", device="cpu"):
        print("[HelmetClassifier] Loading model (background mode)...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id)
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        self.task_queue = queue.Queue(maxsize=3)   # small backlog
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print("[HelmetClassifier] Ready – will process cropped head images.")

    def _worker(self):
        while not self._stop.is_set():
            try:
                img_path, frame_id, request_id = self.task_queue.get(timeout=0.5)
                if img_path is None:
                    continue
                image = Image.open(img_path).convert("RGB")
                # Zero‑shot prompt
                prompt = (
                    "Is the person wearing a safety helmet (hard hat), a crash helmet (bike helmet), or no helmet? "
                    "Answer with exactly one word: safety, crash, or none."
                )
                messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]}
                ]
                text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=10, temperature=0.0)
                response = self.processor.decode(outputs[0], skip_special_tokens=True).strip().lower()

                # Simple keyword extraction
                if "safety" in response:
                    answer = "safety"
                elif "crash" in response:
                    answer = "crash"
                else:
                    answer = "none"

                result = {"frame": frame_id, "request_id": request_id, "helmet_type": answer}
                print(f"[HelmetClassifier] {request_id} -> {answer}")
                with open("helmet_classifications.jsonl", "a") as f:
                    f.write(json.dumps(result) + "\n")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[HelmetClassifier] Worker error: {e}")

    def classify_async(self, image_path, frame_id, request_id):
        """Add a classification task to the queue. Returns immediately."""
        self.task_queue.put((image_path, frame_id, request_id))

    def shutdown(self):
        self._stop.set()
        self._thread.join(timeout=2)
        print("[HelmetClassifier] Shutdown complete.")