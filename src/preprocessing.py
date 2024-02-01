import pandas as pd

# Set display options to show all columns in DataFrames
pd.set_option("display.max_columns", None)

# Read data from CSV files into DataFrames
clients = pd.read_csv("../data/D_clients.csv")
closeLoan = pd.read_csv("../data/D_close_loan.csv")
job = pd.read_csv("../data/D_job.csv")
lastCred = pd.read_csv("../data/D_last_credit.csv")
loan = pd.read_csv("../data/D_loan.csv")
pens = pd.read_csv("../data/D_pens.csv")
salary = pd.read_csv("../data/D_salary.csv")
target = pd.read_csv("../data/D_target.csv")
work = pd.read_csv("../data/D_work.csv")

# Create a dictionary to store DataFrames with corresponding names
tables = {
    "clients": clients,  # Client data
    "closeLoan": closeLoan,
    "job": job,  # Job data
    "lastCred": lastCred,  # Last credit data
    "loan": loan,  # Loan data
    "pens": pens,  # Pension data
    "salary": salary,  # Salary data
    "target": target,  # Target data
    "work": work  # Work data
}

# Print data types of 'ID' columns in DataFrames that have the column
for t in tables:
    if "ID" in tables[t].columns:
        print(t, tables[t].ID.dtypes)

# Create a copy of 'clients' DataFrame, removing duplicate rows based on 'ID'
df = clients.copy().drop_duplicates(keep="last")

# Rename and merge DataFrames based on the 'ID_CLIENT' column
for t in tables:
    if "ID_CLIENT" in tables[t].columns:
        print(t, tables[t].ID_CLIENT.dtypes)
        tables[t] = tables[t].rename(columns={"ID_CLIENT": "ID"})

        df = pd.merge(
            left=df,
            right=tables[t],
            on="ID",
            how="left",
            suffixes=["", f"_{t}"]
        )

# Merge DataFrames based on the 'ID_LOAN' column
df = pd.merge(
    left=df,
    right=closeLoan,
    on="ID_LOAN"
)

# Explore and clean duplicated rows in the DataFrame
df.duplicated().sum()

# Drop duplicate rows, reset index, and display summary statistics
df = df.drop_duplicates(keep="last")\
    .reset_index(drop=True)

# Group by 'ID' and aggregate statistics
stats = df.groupby(by=["ID"]).agg(
    LOAN_NUM_TOTAL=pd.NamedAgg(column="ID_LOAN", aggfunc="count"),
    LOAN_NUM_CLOSED=pd.NamedAgg(column="CLOSED_FL", aggfunc="sum")
)

# Merge aggregated statistics with the original DataFrame
df = pd.merge(
    left=df.drop_duplicates(subset=["ID"]),
    right=stats.reset_index(drop=False),
    on=["ID"],
    how="inner"
)

# Print summary statement
print("The DataFrame 'df' now contains cleaned and merged data.")

"""
Полученные данные в т.ч. содержат приведенные ниже признаки:
- AGREEMENT_RK — уникальный идентификатор объекта в выборке;
- TARGET — целевая переменная: отклик на маркетинговую кампанию (1 — отклик был зарегистрирован, 0 — отклика не было);
- AGE — возраст клиента;
- SOCSTATUS_WORK_FL — социальный статус клиента относительно работы (1 — работает, 0 — не работает);
- SOCSTATUS_PENS_FL — социальный статус клиента относительно пенсии (1 — пенсионер, 0 — не пенсионер);
- GENDER — пол клиента (1 — мужчина, 0 — женщина);
- CHILD_TOTAL — количество детей клиента;
- DEPENDANTS — количество иждивенцев клиента;
- PERSONAL_INCOME — личный доход клиента (в рублях);
- LOAN_NUM_TOTAL — количество ссуд клиента;
- LOAN_NUM_CLOSED — количество погашенных ссуд клиента.
"""
