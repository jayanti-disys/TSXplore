import pandas as pd
import matplotlib.pyplot as plt
import argparse 

if __name__ == "__main__":

    parser = argparser.ArgumentParser()
    parser.add_argument('-i','--input-file',help='Input file')

    args = parser.parse_args()

    df = pd.read_csv(args.input_file,parse_dates=['Date'])

    
    

