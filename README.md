# F1 Model

**For personal educational purposes only**

---

## Problem Statement

With what sort of success and certainty levels can the F1 race results be predicted?
The prediction in the first instance will be made knowing the qualifying results. At the time of writing this there is no empiric or heuristic evidence that this should be better or easier. Gut feel and intuition say it ought to be easier?

## Approach To Understanding the Problem

This problem, while it will involve learning a bunch of new skills, will follow like any other project a systemic route. PACE framework learned in Google's data analytics program will be adopted. For more reference feel free to read a [good summary of PACE framework](https://medium.com/p/12206e1ea536).


### Plan

1. Data
    - The data should at first be fully inclusive from the beginning of F1 existence.
    - The data may later be trimmed or otherwise selectively discarded following EDA.
    - Data shall be acquired from an external source such as [Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
    - Data shall be analysed and missing fields shall be filled in, and missing features should be added. E.g., it is known that weather information is missing. 
    - Data shall be stored on a relational database for ease of manipulation.
2. Stakeholder
    - I am mainly doing this for myself. I will be the accuser, the defendant, the jury and the judge in this court room :)
    - Not discarding the chance for a potential future employer or colleague to view this repository. Their holding stake in this project is the visibility and understanding of the process, as opposed to the final result.
3. Timeline
    - **No strict timelines**. Given some early investigatory work, I estimate **between 20 and 40h** of total work. Time depends on the extent of model building and results of EDA. My knowledge/ignorance permitting, project should aim to be as comprehensive as it takes to be of a great research standard.
4. Milestones
    - **Milestone for Planning Stage**: clear understanding of the problem at hand. Visibility into available data, and best choice of the dataset. Knowledge of data pitfalls and good understanding of what is needed to complete the dataset.
    - **Milestone for Analyse Stage**: Setup database (will require understanding the best solution to this, whether it is SQLite or PostgreSQL or any other)
    - **Milestone for Analyse Stage**: Clean and complete dataset
    - **Milestone for Analyse Stage**: EDA report highlighting key avenues of solving the problem
    - **Milestone for Construct Stage**: Good analysis of a few models along with results analysis from those ML models. This project is for me, need not be efficient in execution. 
    - **Milestone for Construct Stage**: Model comparison and final results report, optionally inclusive of a visualisation dashboard in Tableau
    - **Milestone for Execute Stage**: Make this repository public and link it to my CV

---

## Data

The used dataset will be taken from [Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020). It consists of 14 interlinked CSV files, which screams to be put onto a relational database. But steady...

First - load each file and examine its contents. Check what columns they have, how are they interlinked, what data is missing, how complete is the dataset from 1950 to 2023. Check against other sources. Then complete and clean the dataset. Finally run some basic data description to further understand the data a little more.

### ER diagram of F1 Data

![ER diagram of F1](f1_dataset/F1_ER.png)  
