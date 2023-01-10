import pandas as pd

from evaluation.data_transformation import create_lists_from_df, get_results_df_from_dataset

### Definitions for the dataset and models
datasets = ["mura", "mura"]
directions = ["normal2abnormal", "abnormal2normal"]
epochs = ["_18", "_18", "_16"]

new_names = ["ABC-GAN", "GANterfactual", "CycleGAN"]
averages = True

for idx, (dataset, direction) in enumerate(zip(datasets, directions)):
    # Load results metrics from csv
    results_df = get_results_df_from_dataset(dataset, direction)

    # Transform to desired format
    # First, filter the roght epochs
    counter = 0
    keep_results = pd.DataFrame(columns=results_df.columns)
    for i in range(0, len(results_df), 2):
        # get the rows
        if epochs[counter] in results_df.iloc[i]["Name"]:
            r = results_df.iloc[i]
        else:
            r = results_df.iloc[i + 1]
        keep_results = keep_results.append(r, ignore_index=True)
        counter += 1

    # Chnage Names and combine kid and kid std.
    keep_results['Name'] = keep_results['Name'].replace(keep_results['Name'].unique(), new_names)
    keep_results['KID'] = keep_results['KID'].astype(str) + " +/- " + keep_results['KID_STD'].astype(str)
    keep_results = keep_results.drop('KID_STD', axis=1)
    print(keep_results.to_latex(index=False, index_names=False))
