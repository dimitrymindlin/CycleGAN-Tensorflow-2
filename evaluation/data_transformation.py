import pandas as pd

# This is data pasted from excel
column_names = ['Name', 'TCV_B2A', 'TCV_A2B', 'TCV_Avg', 'KID_B2A', 'KID_A2B', 'KID_Avg', 'SSIM_B2A',
                'SSIM_A2B', 'SSIM_Avg', 'PSNR_B2A', 'PSNR_A2B', 'PSNR_Avg']

simplified_col_names = ['Name', 'TCV', 'KID', 'KID_STD', 'SSIM', 'PSNR']
simplified_avg_col_names = ['Name', 'TCV', 'KID', 'SSIM', 'PSNR']

mura = """
2022-11-28--10.48_18	0,963	0,975	0,97	0,18	0,36	0,27	0,915	0,886	0,90	29,331	27,466	28,40
2022-11-30--10.29_18	0,983	0,997	0,99	0,18	0,3	0,24	0,911	0,868	0,89	28,77	27,134	27,95
2022-11-30--10.45_18	0,902	0,951	0,93	0,2	0,32	0,26	0,917	0,888	0,90	28,987	27,175	28,08
2022-12-03--16.29_18	0,976	1	0,99	0,18	0,32	0,25	0,912	0,862	0,89	28,387	26,067	27,23
2022-12-06--07.00_18	0,99	0,997	0,99	0,1	0,38	0,24	0,883	0,871	0,88	26,038	27,659	26,85
2022-12-06--07.03_18	0,946	0,995	0,97	0,2	0,48	0,34	0,835	0,857	0,85	21,55	24,969	23,26
2022-12-07--05.20_16	0,973	1	0,99	0,26	0,5	0,38	0,829	0,853	0,84	21,476	24,888	23,18
2022-12-08--01.00_18	0,973	1	0,99	0,22	0,4	0,31	0,835	0,883	0,86	22,192	27,078	24,64
2022-11-22--13.19	0,166	0,739	0,45	0,22	0,38	0,30	0,909	0,902	0,91	27,947	27,96	27,95
2022-11-06--18.19	0,976	1,00	0,99	0,44	0,33	0,38	0,85	0,84	0,84	28,98	29,22	29,10
"""

mura_name_mapping = {
    '2022-11-28--10.48_18': '0-0',
    '2022-11-30--10.29_18': '0-1',
    '2022-11-30--10.45_18': '1-0',
    '2022-12-03--16.29_18': '1-1',
    '2022-12-06--07.00_18': '0-1-d',
    '2022-12-06--07.03_18': '0-0-d',
    '2022-12-07--05.20_16': '1-0-d',
    '2022-12-08--01.00_18': '1-1-d',
    '2022-11-22--13.19': 'norm',
    '2022-11-06--18.19': 'gtf'
}

rsna = """
2022-11-08--17.40_18	0,617	0,789	0,70	0,52	0,54	0,53	0,773	0,778	0,78	23,93	21,21	22,57
2022-11-22--13.21_18	0,578	0,721	0,65	0,44	0,6	0,52	0,801	0,79	0,80	25,51	22,01	23,76
2022-11-23--09.53_18	0,55	0,731	0,64	0,5	0,6	0,55	0,797	0,778	0,79	25,13	21,42	23,27
2022-11-25--09.08_18	0,792	0,959	0,88	0,88	0,5	0,69	0,774	0,713	0,74	23,44	19,88	21,66
2022-11-26--09.52_18	0,772	0,95	0,86	0,9	0,6	0,75	0,769	0,704	0,74	23,10	19,44	21,27
2022-11-26--09.54_18	0,798	0,943	0,87	0,8	0,5	0,65	0,772	0,705	0,74	23,41	19,68	21,54
2022-12-29--10.46_16	0,773	0,948	0,86	1	0,55	0,78	0,768	0,703	0,74	23,02	19,38	21,20
2022-11-18--11.14_18	0,586	0,318	0,45	0,825	1,21	1,02	0,617	0,71	0,66	22,20	23,92	23,06
2022-11-02--16.45_18	0,911	0,979	0,95	2,215	1,033	1,62	0,471	0,649	0,56	18,06	19,83	18,95
"""

rna_name_mapping = {
    '2022-11-08--17.40_18': '0_1',
    '2022-11-22--13.21_18': '0-0',
    '2022-11-23--09.53_18': '1-0',
    '2022-11-25--09.08_18': '1-0-d',
    '2022-11-26--09.52_18': '1-1-d',
    '2022-11-26--09.54_18': '0-1-d',
    '2022-12-29--10.46_16': '1-0-d-c',
    '2022-11-18--11.14_18': 'norm',
    '2022-11-02--16.45_18': 'gtf'
}


def transform_comma_to_dot(x):
    return float(x.replace(',', '.'))


def create_lists_from_df(df: pd.DataFrame):
    """Creates a list for each column of a DataFrame.

    The lists are named after the columns and contain the values in the columns.

    Args:
        df: The DataFrame.
    """
    # Get the column names
    column_names = list(df.columns)

    # Iterate over the columns
    for col in column_names:
        # Create a list with the values in the column
        col_list = list(df[col])
        # Create a variable with the same name as the column and assign the list to it
        locals()[col] = col_list
    return locals()


def df_to_tuples(df: pd.DataFrame) -> list:
    # Get the column names
    column_names = list(df.columns)
    tuples = []

    # Iterate over the rows of the DataFrame
    for _, row in df.iterrows():
        # Create a tuple with the values in the row
        t = tuple(row[col] for col in column_names)
        # Add the tuple to the list
        tuples.append(t)

    return tuples


root_file = "/Users/dimitrymindlin/UniProjects/CycleGAN-Tensorflow-2/checkpoints/gans/"


def get_results_df_from_dataset(dataset, direction):
    # Try reading CSV:
    df = pd.read_csv(root_file + f'{dataset}/experiment_results_{dataset}.csv')

    """if dataset == "mura":
        data = mura
        name_mapping = mura_name_mapping
    else:
        data = rsna
        name_mapping = rna_name_mapping

    df = pd.DataFrame([x.split('\t') for x in data.strip().split('\n')], columns=column_names)
    # Replace the comma with a dot and turn the string into a float
    df.iloc[:, 1:] = df.iloc[:, 1:].applymap(lambda x: x.replace(',', '.')).astype(float)"""

    """# Replace the values in the Name column using the mapping
    df['Name'] = df['Name'].map(name_mapping)"""

    # Create separate DataFrames for the A2B and B2A data
    df_a2b = df[['Name', 'TCV_A2B', 'KID_A2B', 'KID_STD_A2B', 'SSIM_A2B', 'PSNR_A2B']]
    df_b2a = df[['Name', 'TCV_B2A', 'KID_B2A', 'KID_STD_B2A', 'SSIM_B2A', 'PSNR_B2A']]

    # Create DF for the avgs
    df_avgs = df[['Name', 'TCV_Avg', 'KID_Avg', 'SSIM_Avg', 'PSNR_Avg']]

    # Rename the columns to remove the A2B and B2A suffixes
    df_a2b.columns = simplified_col_names
    df_b2a.columns = simplified_col_names
    df_avgs.columns = simplified_avg_col_names

    # Convert the TCV, SSIM, PSNR, and KID columns to float

    df_a2b = df_a2b.copy()
    df_b2a = df_b2a.copy()
    for df in [df_a2b, df_b2a, df_avgs]:
        df[list(df.columns)[1:]] = df[list(df.columns)[1:]].apply(pd.to_numeric, errors='coerce', downcast='float')

    if direction == "normal2abnormal":
        return df_a2b
    elif direction == "abnormal2normal":
        return df_b2a
    else:
        return df_avgs
