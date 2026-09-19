import sys

from src.config import to_string
from src.training.trainer import Trainer


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    to_string()

    if mode == "train":
        trainer = Trainer(load_checkpoint=True, load_pretrained=True)
        trainer.train()

    elif mode == "play":
        trainer = Trainer(load_checkpoint=False, load_pretrained=False)
        trainer.play()

    elif mode == "pretrain":
        from src.training.pretrain import pretrain
        pretrain(episodes=3000)

    else:
        print(f"Unknown mode: {mode}. Use 'train', 'play', or 'pretrain'.")


if __name__ == "__main__":
    main()