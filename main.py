import pandas as pd

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

features_df = pd.DataFrame(columns=["raceId","driverId","circuitId",
                                    "constructorId","year","grid",
                                    "q1time","q2time","q3time",
                                    "resultN1","resultN2","resultN3",
                                    "weather","circuitType",
                                    "generalClassification"])

def run_main():
    # global drivers_df
    # global results_df
    # global races_df
    drivers_df = pd.read_csv("kaggle F1/drivers.csv")
    results_df = pd.read_csv("kaggle F1/results.csv")
    races_df = pd.read_csv("kaggle F1/races.csv")
    qualifying_df = pd.read_csv("kaggle F1/qualifying.csv")

    # here we grab an ID of a race
    for index_races, row_races in races_df[races_df.year == 2023].iterrows():
        print("NEW RACE\n\n")
        # for this race, we need to find all results
        for index_res, row_res in results_df[results_df.raceId == row_races["raceId"]].iterrows():
            print("started {:d}, finished {:s} ".format(row_res["grid"], row_res["position"]))




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_main()
