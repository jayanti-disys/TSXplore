import pandas as pd
import glob 

class LadaLoader:
    def __init__(self, input_data_dir):

        self.input_data_dir = input_data_dir 
        all_files = glob.glob(input_data_dir + os.sep + "*.csv")
        self.all_files = [os.path.basename(a).replace(".csv","") for a in all_files]

    def get_column_names (self, name):
        file_name = self.input_data_dir + os.sep + name + ".csv"
        df = pd.read_csv(file_name)
        columns = list(df.columns)
        return columns 

