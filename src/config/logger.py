import hashlib
import json
import os
from datetime import datetime

from src.config import CONFIG


def _create_config_hash(config_data):
    config_json = json.dumps(config_data, sort_keys=True, default=str)
    return hashlib.sha256(config_json.encode("utf-8")).hexdigest()


class Logger:

    def __init__(self):
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)

        self.config = dict(CONFIG)
        self.config_hash = _create_config_hash(self.config)

        self.log_file = self._find_log_file()

        if self.log_file is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_file = (
                f"{self.log_dir}/{timestamp}_{self.config_hash[:8]}.log"
            )
            self._write_config()

    def _find_log_file(self):
        for filename in os.listdir(self.log_dir):
            if not filename.endswith(".log"):
                continue

            path = os.path.join(self.log_dir, filename)

            with open(path, "r", encoding="utf-8") as file:
                first_line = file.readline().strip()

            if first_line == f"CONFIG_HASH={self.config_hash}":
                return path

        return None

    def _write_config(self):
        with open(self.log_file, "a", encoding="utf-8") as file:
            file.write(f"CONFIG_HASH={self.config_hash}\n")
            file.write("CONFIG:\n")
            for key, value in self.config.items():
                file.write(f"{key}={value}\n")
            file.write("\n")

    def log(self, message):
        print(message)
        with open(self.log_file, "a", encoding="utf-8") as file:
            file.write(message + "\n")

    def episode(self, episode, score, avg_score, best_score,
                reward, loss, epsilon, steps):
        message = (
            f"Episode {episode} | "
            f"Score {score} | "
            f"Avg {avg_score:.2f} | "
            f"Best {best_score} | "
            f"Reward {reward:.2f} | "
            f"Loss {loss:.4f} | "
            f"Epsilon {epsilon:.4f} | "
            f"Steps {steps}"
        )
        self.log(message)