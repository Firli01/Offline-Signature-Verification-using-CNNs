from typing import Tuple
import matplotlib.pyplot as plt
import os


def generate_graphs(save_name: Tuple = None,
                    train_loss_acc: Tuple = None,
                    val_loss_acc: Tuple = None,
                    visualize_loop: bool = False,
                    loop_list: list = None,
                    dataset_names: list = None,
                    test_accuracies: list = None,
                    writer_dependent: bool = False,
                    ):
    if not visualize_loop:
        model_name, data_name = save_name
        val_loss, val_acc = val_loss_acc
        epochs = range(len(val_loss))

        # plt.figure(figsize=(12, 8))
        # plt.plot(epochs, val_loss, label="Val Loss")
        # plt.title(
        #     f"Validation Loss for {model_name} on {data_name} - Writer {'dependent' if writer_dependent else 'independent'}",
        #     fontsize=20)
        # plt.ylabel("Loss", fontsize=18)
        # plt.xlabel("Epoch", fontsize=18)
        # plt.xticks(ticks=range(1, len(epochs) + 1, 1), fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.legend(loc="lower right", fontsize=14)
        # plt.tight_layout()
        # plt.savefig(os.path.join("plots", f"val_loss_{model_name}_{data_name}.png"))
        # plt.close()

        plt.figure(figsize=(12, 8))
        plt.plot(epochs, val_acc, label="Validation Accuracy")
        plt.title(
            f"Validation Accuracy for {model_name} on {data_name} - Writer {'dependent' if writer_dependent else 'independent'}",
            fontsize=20)
        plt.ylabel("Accuracy (%)", fontsize=18)
        plt.xlabel("Epoch", fontsize=18)
        plt.xticks(ticks=range(1, len(epochs) + 1, 1), fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(loc="lower right", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join("plots", f"val_acc_{model_name}_{data_name}.png"))
        plt.close()

    else:
        epochs = range(1, len(loop_list[0][0][1]) + 1)
        colors = ["b", "g", "r", "c", "m"]
        for i, m in enumerate(loop_list):
            model_name = m[-1]

            # plt.figure(figsize=(12, 8))
            # for idx, ds in enumerate(m[:-1]):
            #     plt.plot(epochs, ds[3], label=f"{ds[0]}", color=colors[idx])
            # plt.title(
            #     f"Val Loss for {model_name} on all Datasets - Writer {'dependent' if writer_dependent else 'independent'}",
            #     fontsize=20)
            # plt.ylabel("Loss", fontsize=18)
            # plt.xlabel("Epoch", fontsize=18)
            # plt.xticks(ticks=range(1, len(epochs) + 1, 1), fontsize=16)
            # plt.yticks(fontsize=16)
            # plt.legend(loc="lower right", fontsize=14)
            # plt.tight_layout()
            # plt.savefig(os.path.join("plots", f"val_loss_{model_name}.png"))
            # plt.close()

            plt.figure(figsize=(12, 8))
            for idx, ds in enumerate(m[:-1]):
                plt.plot(epochs, ds[4], label=f"{ds[0]}", color=colors[idx])
            plt.title(
                f"Val Accuracy for {model_name} on all Datasets - Writer {'dependent' if writer_dependent else 'independent'}",
                fontsize=20)
            plt.ylabel("Accuracy (%)", fontsize=18)
            plt.xlabel("Epoch", fontsize=18)
            plt.xticks(ticks=range(1, len(epochs) + 1, 1), fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend(loc="lower right", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join("plots", f"val_acc_{model_name}.png"))
            plt.close()

            plt.figure(figsize=(12, 8))
            bars = plt.bar(dataset_names, test_accuracies[i], color="skyblue")
            for j, label in enumerate(plt.gca().get_xticklabels()):
                plt.gca().get_xticklabels()[j].set_rotation(45)
                plt.gca().get_xticklabels()[j].set_ha("right")
                plt.gca().get_xticklabels()[j].set_fontsize(16)
            plt.title(
                f"Test Accuracy for {model_name} on all Datasets - Writer {'dependent' if writer_dependent else 'independent'}",
                fontsize=20)
            plt.xlabel("Datasets", fontsize=18)
            plt.ylabel("Test Accuracy (%)", fontsize=18)
            plt.yticks(fontsize=16)
            plt.ylim(0, 100)  # Set the y-axis limit from 0 to 100
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{yval:.2f}%", ha='center', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join("plots", f"test_{model_name}.png"))
            plt.close()

