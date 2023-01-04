import matplotlib.pyplot as plt

from evaluation.data_transformation import create_lists_from_df, get_results_df_from_dataset

datasets = ["rsna"]
directions = ["averages"]
marker_list = ["o"]
averages = True

# Create a figure and 2D subplot
fig, ax = plt.subplots()
for idx, (dataset, direction, marker) in enumerate(zip(datasets, directions, marker_list)):
    lists_dict = create_lists_from_df(get_results_df_from_dataset(dataset, direction))
    names = lists_dict["Name"]
    TCV = lists_dict["TCV"]
    SSIM = lists_dict["SSIM"]
    KID = lists_dict["KID"]

    # Plot the data as a 2D scatterplot
    scatter = ax.scatter(TCV, KID, c=SSIM, vmin=0.5, vmax=1, marker=marker, label=f"{dataset}_{direction}")

    # Add the names of the models to the points in the plot
    for i, name in enumerate(names):
        # Calculate the offset for the xytext parameter
        offset = (max(TCV) - min(TCV)) * 0.01
        ax.text(TCV[i] + 0.01, KID[i], name)

    # Add a colorbar to show the SSIM values
    if idx == 0:
        colorbar = fig.colorbar(scatter)
        colorbar.set_label('SSIM')

# Add labels for the axes
ax.set_xlabel('TCV')
ax.set_ylabel('KID')

# Add a title for the plot
ax.set_title(f'2D Scatterplot of ABC-GANs')

# Show the plot
plt.legend()
plt.show()
