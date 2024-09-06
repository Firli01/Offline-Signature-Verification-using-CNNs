from typing import Tuple

from datasets import CEDAR, SignatureDataset300, BHSigBengali, BHSigHindi, CombinedDataset, Dataset

dataset_list = ["1: CEDAR",
                "2: SignatureDataset300",
                "3: BHSigBengali",
                "4: BHSigHindi",
                "5: CombinedDataset"]


def get_dataset(writer_dependent: bool = False) -> Tuple[str, Dataset]:

    print(f"\nAvailable Datasets:")
    for d in dataset_list:
        print(d)

    while True:
        try:
            selection = int(input("Type a Number to select a Dataset: "))
            if selection == 1:
                dataset = CEDAR(writer_dependent=writer_dependent)
            elif selection == 2:
                dataset = SignatureDataset300(writer_dependent=writer_dependent)
            elif selection == 3:
                dataset = BHSigBengali(writer_dependent=writer_dependent)
            elif selection == 4:
                dataset = BHSigHindi(writer_dependent=writer_dependent)
            elif selection == 5:
                dataset = CombinedDataset(writer_dependent=writer_dependent)
            break
        except ValueError:
            print("Invalid Input")

    data_path = dataset_list[selection - 1][3:]

    print(f"\nUsing Model: {data_path}")

    return data_path, dataset


def get_all_datasets(writer_dependent: bool = False):
    return ((CEDAR(writer_dependent=writer_dependent), dataset_list[0][3:]),
            (SignatureDataset300(writer_dependent=writer_dependent), dataset_list[1][3:]),
            (BHSigBengali(writer_dependent=writer_dependent), dataset_list[2][3:]),
            (BHSigHindi(writer_dependent=writer_dependent), dataset_list[3][3:]),
            (CombinedDataset(writer_dependent=writer_dependent), dataset_list[4][3:])
            )
