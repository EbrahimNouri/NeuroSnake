import sys

from src.training.trainer import Trainer


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py train")
        print("  python main.py play")
        return

    mode = sys.argv[1].lower()

    if mode == "train":
        trainer = Trainer(load_checkpoint=True)
        trainer.train()

    elif mode == "play":
        trainer = Trainer(load_checkpoint=False)
        trainer.play()

    else:
        print(f"Unknown mode: {mode}")
        print("Use: train or play")


if __name__ == "__main__":
    main()

