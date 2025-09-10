from __future__ import annotations
import os
import time
import threading
from pathlib import Path
from bitmap2svg.ingest import load
from bitmap2svg.pipeline import vectorise

class Watcher:
    def __init__(self, input_dir: str, output_dir: str, sleep_time: float = 1.0):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.sleep_time = sleep_time
        self.processed_files = set()

    def run(self):
        while True:
            self.process_new_files()
            time.sleep(self.sleep_time)

    def process_new_files(self):
        for img_path in self.input_dir.glob("*.*"):
            if img_path not in self.processed_files:
                self.process_file(img_path)
                self.processed_files.add(img_path)

    def process_file(self, img_path: Path):
        try:
            img = load(img_path)
            res = vectorise(img, cfg=None)  # Replace cfg=None with actual config if needed
            output_path = self.output_dir / f"{img_path.stem}.svg"
            output_path.write_text(res.svg_min, encoding="utf-8")
            print(f"Processed {img_path} -> {output_path}")
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

def main(input_dir: str, output_dir: str):
    watcher = Watcher(input_dir, output_dir)
    watcher_thread = threading.Thread(target=watcher.run, daemon=True)
    watcher_thread.start()
    watcher_thread.join()

if __name__ == "__main__":
    input_directory = "in/"  # Specify your input directory
    output_directory = "out/"  # Specify your output directory
    main(input_directory, output_directory)