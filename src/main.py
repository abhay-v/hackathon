import kagglehub
import tensorflow as tf
import csv
import numpy as np
import pandas as pd

def main():
    path = kagglehub.dataset_download("s3programmer/flood-risk-in-india")
    path += "/flood_risk_dataset_india.csv"

    flood_variables = pd.read_csv(path, header=0)
    #print(dict(flood_variables))

    expected = flood_variables.pop(flood_variables.columns[-1])

    model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Dense(flood_variables.columns),
#                   tf.keras.layers.Dense(128, activation='relu'),
#                   tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(1, activation='relu'),
                ]
            )

    flood_variables = tf.data.Dataset.from_tensor_slices(dict(flood_variables[:2500]))
    print(flood_variables)


    predictions = model(flood_variables).numpy
    #print(predictions)





if __name__ == "__main__":
    main()

