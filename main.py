import json
import os
import torch
from datetime import datetime
import time

from select_dataset import get_dataset, get_all_datasets
from select_model import get_model, get_all_models
from dataloader import get_dataloader
from train import train_model
from test import test_model
from visualize_results import generate_graphs

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA: ", torch.cuda.is_available())
print("CUDA Devices: ", torch.cuda.device_count())
print("CUDA Device: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


# noinspection PyShadowingNames
def main(model, learning_rate: float, batch_size: int, num_epochs=25, train_size: float = 0.8):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)

    if not train_all:
        train_loader, val_loader, test_loader = get_dataloader(dataset=dataset,
                                                               train_percentage=train_size,
                                                               batch_size=batch_size,
                                                               save_model_and_dataloader=save_model_and_dataloader,
                                                               save_name=save_name
                                                               )
        # TRAIN
        train_model(model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    num_epochs=num_epochs,
                    device=device,
                    save_model_and_dataloader=save_model_and_dataloader,
                    save_name=save_name,
                    plot_name=plot_name
                    )

        # EVALUATE
        test_model(model=model,
                   test_loader=test_loader,
                   device=device)
    else:
        train_loader, val_loader, test_loader = get_dataloader(dataset=dataset,
                                                               train_percentage=train_size,
                                                               batch_size=batch_size
                                                               )
        # TRAIN
        train_loss_history, train_acc_history, val_loss_history, val_acc_history = train_model(model=model,
                                                                                               train_loader=train_loader,
                                                                                               val_loader=val_loader,
                                                                                               optimizer=optimizer,
                                                                                               criterion=criterion,
                                                                                               num_epochs=num_epochs,
                                                                                               device=device
                                                                                               )
        # EVALUATE
        test_acc = test_model(model=model,
                              test_loader=test_loader,
                              device=device)

        return train_loss_history, train_acc_history, val_loss_history, val_acc_history, test_acc


if __name__ == '__main__':
    start_time = time.time()
    # PARAMETER
    with open("config.json", 'r') as fh:
        config = json.load(fh)

    save_model_and_dataloader = False
    train_all = True

    if not train_all:
        data_path, dataset = get_dataset(writer_dependent=config["writer_dependent"])
        num_classes = dataset.classes
        model_name, model = get_model(num_classes=num_classes,
                                      pretrained_weights=config["pretrained_weights"]
                                      )

        save_name = f"{model_name}_{data_path}_{'dependent' if config['writer_dependent'] else 'independent'}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
        plot_name = (model_name, data_path)

        if config["eval_only"]:
            test_model(model=model.load_state_dict(torch.load(os.path.join("models", "model_XXXXINSERTNAMEXXXX.pth"))),
                       test_loader=torch.load(os.path.join("test_loader", "test_loader_XXXXINSERTNAMEXXXX.pth")),
                       device=device
                       )
        else:
            main(model=model,
                 learning_rate=config["learning_rate"],
                 num_epochs=config["num_epochs"],
                 batch_size=config["batch_size"],
                 train_size=config["train_size"],
                 )
    else:
        results = []
        test_accuracies = []
        for i in range(1, 7):
            res_per_model = []
            test_acc_per_model = []
            dat = get_all_datasets(writer_dependent=config["writer_dependent"])
            datasets = [x[0] for x in dat]
            dataset_names = [x[1] for x in dat]
            for j, dataset in enumerate(datasets):
                num_classes = dataset.classes
                model_name, model = get_all_models(num_classes=num_classes,
                                                   pretrained_weights=config["pretrained_weights"],
                                                   model_counter=i
                                                   )
                train_loss, train_acc, val_loss, val_acc, test_acc = main(model=model,
                                                                          learning_rate=config["learning_rate"],
                                                                          num_epochs=config["num_epochs"],
                                                                          batch_size=config["batch_size"],
                                                                          train_size=config["train_size"]
                                                                          )

                test_acc_per_model.append(test_acc)
                res_per_model.append([dataset_names[j], train_loss, train_acc, val_loss, val_acc])
            test_accuracies.append(test_acc_per_model)
            res_per_model.append(model_name)
            results.append(res_per_model)
        generate_graphs(visualize_loop=train_all,
                        loop_list=results,
                        dataset_names=dataset_names,
                        test_accuracies=test_accuracies,
                        writer_dependent=config["writer_dependent"]
                        )
    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Elapsed time: {int(hours)} hr {int(minutes)} min {seconds:.2f} sec")
