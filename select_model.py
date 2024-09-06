from torchvision import models
import torch.nn as nn

from CNN import CNN


def get_model(num_classes: int, pretrained_weights: bool):
    model_list = ["1: Simple CNN",
                  "2: AlexNet",
                  "3: EfficientNet V2 M",
                  "4: GoogLeNet",
                  "5: ResNet",
                  "6: Vision Transformer L 32"]

    print(f"\nAvailable Models:")
    for m in model_list:
        print(m)

    while True:
        try:
            selection = int(input("Type a Number to select a model: "))
            if selection == 1:
                model = CNN(num_classes=num_classes)
            elif selection == 2:
                model = models.alexnet(pretrained=pretrained_weights)
            elif selection == 3:
                model = models.efficientnet_v2_m(pretrained=pretrained_weights)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            elif selection == 4:
                if pretrained_weights:
                    model = models.googlenet(pretrained=pretrained_weights)
                else:
                    model = models.googlenet(pretrained=pretrained_weights, aux_logits=False)
            elif selection == 5:
                model = models.resnet152(pretrained=pretrained_weights)
            elif selection == 6:
                model = models.vit_l_32(pretrained=pretrained_weights)
            break
        except ValueError:
            print("Invalid Input")

    if selection != 1:
        print(f"\nPretrained weights: {pretrained_weights}")

    model_name = model_list[int(selection) - 1][3:]

    print(f"\nUsing Model: {model_name}")

    return model_name, model


def get_all_models(num_classes: int, pretrained_weights: bool, model_counter: int):
    model_name, model = None, None
    if model_counter == 1:
        model = CNN(num_classes=num_classes)
        model_name = "Simple CNN"
    elif model_counter == 2:
        model = models.alexnet(pretrained=pretrained_weights)
        model_name = "AlexNet"
    elif model_counter == 3:
        model = models.efficientnet_v2_m(pretrained=pretrained_weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        model_name = "EfficientNet V2 M"
    elif model_counter == 4:
        if pretrained_weights:
            model = models.googlenet(pretrained=pretrained_weights)
        else:
            model = models.googlenet(pretrained=pretrained_weights, aux_logits=False)
        model_name = "GoogLeNet"
    elif model_counter == 5:
        model = models.resnet152(pretrained=pretrained_weights)
        model_name = "ResNet"
    elif model_counter == 6:
        model = models.vit_l_32(pretrained=pretrained_weights)
        model_name = "Vision Transformer L 32"

    return model_name, model
