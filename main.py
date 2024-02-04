import pandas as pd
from datetime import datetime

# sort data into input features: for every race

#   driverID
#   circuitID
#   teamID
#   year?
#   quali position
#   quali 1 time
#   quali 2 time
#   quali 3 time
#   past five results
#   race weather
#   racetrack type
#   general classification ahead of the race
#   ----
#   fp1 time/position
#   fp2 time/position
#   fp3 time/position
#   upgrades brought to the race?

features_df = pd.DataFrame(columns=["raceId", "driverId", "circuitId",
                                    "constructorId", "year", "grid",
                                    "finish",
                                    "q1time", "q2time", "q3time",
                                    "resultN1", "resultN2", "resultN3",
                                    "weather", "circuitType",
                                    "generalClassification"])


def run_main():
    global features_df
    drivers_df = pd.read_csv("kaggle F1/drivers.csv")
    results_df = pd.read_csv("kaggle F1/results.csv")
    races_df = pd.read_csv("kaggle F1/races.csv")
    qualifying_df = pd.read_csv("kaggle F1/qualifying.csv")

    # populate the dataset with race results in those years
    for year in range(2011, 2024):

        # here we grab an ID of a race from a particular year
        for index_races, row_races in races_df[races_df.year == year].iterrows():

            # make sure there are results from that race
            if len(results_df[results_df.raceId == row_races["raceId"]]) != 0:

                # for this race, we need to find all drivers finishing results
                for index_res, row_res in results_df[results_df.raceId == row_races["raceId"]].iterrows():

                    # we also need quali times
                    try:
                        q1_time = qualifying_df[(qualifying_df.raceId == row_races["raceId"]) & (
                                    qualifying_df.driverId == row_res["driverId"])]["q1"].iloc[0]
                        q2_time = qualifying_df[(qualifying_df.raceId == row_races["raceId"]) & (
                                    qualifying_df.driverId == row_res["driverId"])]["q2"].iloc[0]
                        q3_time = qualifying_df[(qualifying_df.raceId == row_races["raceId"]) & (
                                    qualifying_df.driverId == row_res["driverId"])]["q3"].iloc[0]
                    except IndexError:
                        q1_time = "\\N"
                        q2_time = "\\N"
                        q3_time = "\\N"

                    try:
                        q1_time = datetime.strptime(q1_time, "%M:%S.%f")
                        q1_time = q1_time.minute * 60 + q1_time.second + q1_time.microsecond / 1e6
                    except ValueError:
                        q1_time = "\\N"
                    except TypeError:
                        q1_time = "\\N"

                    try:
                        q2_time = datetime.strptime(q2_time, "%M:%S.%f")
                        q2_time = q2_time.minute * 60 + q2_time.second + q2_time.microsecond / 1e6
                    except ValueError:
                        q2_time = "\\N"
                    except TypeError:
                        q2_time = "\\N"

                    try:
                        q3_time = datetime.strptime(q3_time, "%M:%S.%f")
                        q3_time = q3_time.minute * 60 + q3_time.second + q3_time.microsecond / 1e6
                    except ValueError:
                        q3_time = "\\N"
                    except TypeError:
                        q3_time = "\\N"

                    if len(features_df[features_df.driverId == row_res["driverId"]]) > 2:
                        result_n1 = features_df[features_df.driverId == row_res["driverId"]]["finish"].iloc[-1]
                        result_n2 = features_df[features_df.driverId == row_res["driverId"]]["finish"].iloc[-2]
                        result_n3 = features_df[features_df.driverId == row_res["driverId"]]["finish"].iloc[-3]

                    elif len(features_df[features_df.driverId == row_res["driverId"]]) > 1:
                        result_n1 = features_df[features_df.driverId == row_res["driverId"]]["finish"].iloc[-1]
                        result_n2 = features_df[features_df.driverId == row_res["driverId"]]["finish"].iloc[-2]
                        result_n3 = "25"

                    elif len(features_df[features_df.driverId == row_res["driverId"]]) > 0:
                        result_n1 = features_df[features_df.driverId == row_res["driverId"]]["finish"].iloc[-1]
                        result_n2 = "25"
                        result_n3 = "25"

                    else:
                        result_n1 = "25"
                        result_n2 = "25"
                        result_n3 = "25"

                    if row_res["position"] == "\\N":
                        finish = 25
                    else:
                        finish = row_res["position"]

                    entry = {
                        "raceId": row_races["raceId"],
                        "driverId": row_res["driverId"],
                        "circuitId": row_races["circuitId"],
                        "constructorId": row_res["constructorId"],
                        "year": year,
                        "grid": row_res["grid"],
                        "finish": finish,
                        "q1time": q1_time,
                        "q2time": q2_time,
                        "q3time": q3_time,
                        "resultN1": result_n1,
                        "resultN2": result_n2,
                        "resultN3": result_n3,
                        "weather": 0,
                        "circuitType": 0,
                        "generalClassification": 0
                    }

                    entry_df = pd.DataFrame([entry])
                    features_df = pd.concat([features_df, entry_df], ignore_index=True)

    # represent quali times as difference to the best time
    for race in set(features_df.raceId):

        q1_min = features_df[(features_df["raceId"] == race) & (features_df["q1time"] != "\\N")]["q1time"].min()
        features_df.loc[(features_df["raceId"] == race) & (features_df["q1time"] != "\\N"),"q1time"] -= q1_min

        q2_min = features_df[(features_df["raceId"] == race) & (features_df["q2time"] != "\\N")]["q2time"].min()
        features_df.loc[(features_df["raceId"] == race) & (features_df["q2time"] != "\\N"),"q2time"] -= q2_min

        q3_min = features_df[(features_df["raceId"] == race) & (features_df["q3time"] != "\\N")]["q3time"].min()
        features_df.loc[(features_df["raceId"] == race) & (features_df["q3time"] != "\\N"),"q3time"] -= q3_min

    features_df.loc[(features_df["q1time"] == "\\N"), "q1time"] = 15
    features_df.loc[(features_df["q2time"] == "\\N"), "q2time"] = 15
    features_df.loc[(features_df["q3time"] == "\\N"), "q3time"] = 15

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_main()
    print(features_df.head())
