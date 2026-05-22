import threading
import queue
import json
import torch
import time
import traceback
import re
import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

class AsyncVlmAnalyzer:
    def __init__(self, model_id="HuggingFaceTB/SmolVLM2-256M-Instruct", device="cpu", max_queue_size=1):
        print(f"[VLM] Loading model {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id)
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        print("[VLM] Model loaded successfully.")

        # Only one pending task at a time
        self.task_queue = queue.Queue(maxsize=max_queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print(f"[VLM] Background worker started (max queue size = {max_queue_size}).")

    @staticmethod
    def repair_json(text):
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return None
        json_str = match.group(0)
        json_str = json_str.replace("'", '"')
        json_str = re.sub(r':\s*true\b', ': true', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r':\s*false\b', ': false', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r':\s*bool\b', ': false', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r',\s*}', '}', json_str)
        return json_str

    def _worker(self):
        print("[VLM Worker] Thread alive.")
        while not self._stop.is_set():
            try:
                img_path, worker_id, prompt = self.task_queue.get(timeout=0.5)
                if img_path is None:
                    continue
                print(f"[VLM Worker] Processing worker {worker_id}, image {img_path}")
                image = Image.open(img_path).convert("RGB")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    # Fast generation – only 30 tokens
                    outputs = self.model.generate(**inputs, max_new_tokens=30)
                full_response = self.processor.decode(outputs[0], skip_special_tokens=True)

                # Extract assistant's reply
                assistant_part = full_response
                if "Assistant:" in full_response:
                    assistant_part = full_response.split("Assistant:")[-1].strip()

                # Parse JSON
                result = None
                try:
                    start = assistant_part.find('{')
                    end = assistant_part.rfind('}') + 1
                    if start != -1 and end != 0:
                        json_str = assistant_part[start:end]
                        result = json.loads(json_str)
                except Exception:
                    repaired = self.repair_json(assistant_part)
                    if repaired:
                        try:
                            result = json.loads(repaired)
                        except Exception:
                            result = {"error": "parse_failed", "raw": assistant_part}
                    else:
                        result = {"error": "no_json", "raw": assistant_part}

                print(f"\n[VLM RESULT] Worker {worker_id}: {result}\n")

                # Save result to JSONL file
                log_entry = {
                    "timestamp": time.time(),
                    "worker_id": worker_id,
                    "image_path": img_path,
                    "result": result
                }
                try:
                    with open("vlm_results.json", "a") as f:
                        f.write(json.dumps(log_entry) + "\n")
                    print("[VLM Worker] Appended result to vlm_results.json")
                except Exception as e:
                    print(f"[VLM Worker] Failed to write file: {e}")

                # Delete snapshot after processing
                try:
                    os.remove(img_path)
                    print(f"[VLM Worker] Deleted snapshot: {img_path}")
                except Exception as e:
                    print(f"[VLM Worker] Failed to delete snapshot: {e}")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VLM Worker] ERROR: {type(e).__name__}: {e}")
                traceback.print_exc()
        print("[VLM Worker] Exiting.")

    def analyze_async(self, image_path, worker_id, prompt=None):
        if self.task_queue.full():
            print(f"[VLM] Queue full, dropping worker {worker_id}")
            # Delete orphaned snapshot immediately
            try:
                os.remove(image_path)
            except:
                pass
            return

        if prompt is None:
            prompt = (
                "You are a strict JSON output generator. Analyze the worker in this image. "
                "Return ONLY a valid JSON object with NO additional text. "
                "Use double quotes. Keys: helmet_type, vest_color, reflective_stripes, overall_compliance. "
                'Example: {"helmet_type": "safety", "vest_color": "green", "reflective_stripes": true, "overall_compliance": true}'
            )
        print(f"[VLM] Queueing worker {worker_id}")
        self.task_queue.put((image_path, worker_id, prompt))

    def shutdown(self):
        print("[VLM] Shutting down...")
        self._stop.set()
        self._thread.join(timeout=2)
        print("[VLM] Shutdown complete.")