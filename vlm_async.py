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

# Optional: use json5 for better parsing (install: pip install json5)
try:
    import json5
    USE_JSON5 = True
except ImportError:
    USE_JSON5 = False
    print("[VLM] json5 not installed, using fallback repair method.")

class AsyncVlmAnalyzer:
    def __init__(self, model_id="HuggingFaceTB/SmolVLM2-256M-Instruct", device="cpu", max_queue_size=1):
        print(f"[VLM] Loading model {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id)
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        print("[VLM] Model loaded successfully.")

        self.task_queue = queue.Queue(maxsize=max_queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print(f"[VLM] Background worker started (max queue size = {max_queue_size}).")

    @staticmethod
    def repair_json(text):
        """Safer repair: use json5 if available, else manual regex."""
        if USE_JSON5:
            try:
                return json5.loads(text)
            except Exception:
                pass
        # Manual fallback: extract content between first { and last }
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return None
        json_str = match.group(0)
        # Replace single quotes only when they are likely key/value delimiters
        # This avoids breaking apostrophes inside strings
        json_str = re.sub(r"(?<=[{, ])'|'(?=[:}])", '"', json_str)
        # Replace bare boolean literals
        json_str = re.sub(r':\s*true\b', ': true', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r':\s*false\b', ': false', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r':\s*bool\b', ': false', json_str, flags=re.IGNORECASE)
        # Remove trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        try:
            return json.loads(json_str)
        except:
            return None

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
                    # CRITICAL: increased max_new_tokens to 80
                    outputs = self.model.generate(**inputs, max_new_tokens=80)
                full_response = self.processor.decode(outputs[0], skip_special_tokens=True)

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
                        result = repaired
                    else:
                        # Truncate raw response to avoid log bloat
                        raw_short = (assistant_part[:200] + '...') if len(assistant_part) > 200 else assistant_part
                        result = {"error": "parse_failed", "raw": raw_short}

                print(f"\n[VLM RESULT] Worker {worker_id}: {result}\n")

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
                # Attempt to delete snapshot even on error
                try:
                    if 'img_path' in locals() and os.path.exists(img_path):
                        os.remove(img_path)
                        print(f"[VLM Worker] Deleted snapshot after error: {img_path}")
                except:
                    pass
        print("[VLM Worker] Exiting.")

    def analyze_async(self, image_path, worker_id, prompt=None):
        # Fix TOCTOU race: use put_nowait
        if prompt is None:
            prompt = (
                "You are a strict JSON output generator. Analyze the worker in this image. "
                "Return ONLY a valid JSON object with NO additional text. "
                "Use double quotes. Keys: helmet_type, vest_color, reflective_stripes, overall_compliance. "
                'Example: {"helmet_type": "safety", "vest_color": "green", "reflective_stripes": true, "overall_compliance": true}'
            )
        try:
            self.task_queue.put_nowait((image_path, worker_id, prompt))
            print(f"[VLM] Queued worker {worker_id}")
        except queue.Full:
            print(f"[VLM] Queue full, dropping worker {worker_id}")
            try:
                os.remove(image_path)
                print(f"[VLM] Deleted orphaned snapshot: {image_path}")
            except:
                pass

    def shutdown(self):
        print("[VLM] Shutting down...")
        self._stop.set()
        # Increase timeout to allow current inference to finish (30s typical)
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            print("[VLM] Worker did not finish within 30s, forcing exit.")
        else:
            print("[VLM] Shutdown complete.")