import pandas as pd

from evaluation.scripts.data_transformation import get_results_df_from_dataset

### Definitions for the dataset and models
dataset = "mura"
directions = ["averages"] # "normal2abnormal", "abnormal2normal",
epochs = None  # epochs = ["_180", "_180", "_180"]
new_names = ["CycleGAN", "GANterfactual", "ABC-GAN (Ours)"]
averages = True

for idx, direction in enumerate(directions):
    # Load results metrics from csv
    results_df = get_results_df_from_dataset(dataset, direction)

    # Transform to desired format
    # First, filter the right epochs
    counter = 0
    keep_results = pd.DataFrame(columns=results_df.columns)
    if epochs is not None:
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
    else:
        keep_results = results_df


    try:
        keep_results['KID'] = keep_results['KID'].astype(str) + " +/- " + keep_results['KID_STD'].astype(str)
        keep_results = keep_results.drop('KID_STD', axis=1)
    except KeyError:
        pass  # average doesn't have KID_STD

    keep_results = keep_results.reindex(columns=['Name', 'KID', 'TCV', 'SSIM', 'PSNR'])
    keep_results[['TCV', 'SSIM', 'PSNR']] = keep_results[['TCV', 'SSIM', 'PSNR']].round(2)
    keep_results = keep_results.sort_values(['KID', 'TCV'], ascending=[False, True])

    print(direction)
    print(keep_results.to_latex(index=False, index_names=False))
