import pandas as pd

# Read the CSV file
df = pd.read_csv("/home/aziz/Desktop/fatigue_detection/datasets/data_selected1.csv")

# Read the Excel file
excel_df = pd.read_excel(
    "/home/aziz/Desktop/fatigue_detection/datasets/archive/fatigueset/preliminary_questionnaire.xlsx"
)
