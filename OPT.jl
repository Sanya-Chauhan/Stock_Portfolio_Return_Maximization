using CSV, DataFrames, Statistics, Random, Plots, Distances, Clustering, Gurobi, JuMP,  Statistics, Dates
using ScikitLearn, DecisionTree, MLDataUtils, CategoricalArrays

# Train Optimal Classifier Tree on Optimal Decisions Using Only Historical Pricing

function create_lag_variables(df::DataFrame, freq::String="daily")
    lag_vars = [:Open, :High, :Low, :Close, :Volume]
    lag_periods = freq == "daily" ? 14 : 3

    for var in lag_vars
        for i in 1:14
            df[!, Symbol(string(var, "_lag_", i))] = circshift(df[!, var], i)
        end
    end

    return df
end
                
# Assuming 'stock' is your DataFrame with stock data
df = deepcopy(aapl_df_subset)

# Converting 'Date' to DateTime and extracting day, month, year
#df[!, :Date] = Date.(df[!, :Date], "yyyy-mm-dd")
                
# Correctly extracting year, month, and day
df[!, :Year] = year.(df[!, :Date])
df[!, :Month] = month.(df[!, :Date])
df[!, :Day] = day.(df[!, :Date])

# Optionally, you can drop the original 'Date' column
select!(df, Not(:Date))


# Reordering columns
column_order = [:Day, :Month, :Year]
append!(column_order, [Symbol(col) for col in setdiff(names(df), ["Day", "Month", "Year"])])
df = df[!, column_order]

# Creating lag variables
df = create_lag_variables(df, "daily")

# Dropping rows with missing values
dropmissing!(df)

# Dropping certain columns
select!(df, Not([:High, :Low, :Volume, :Close, Symbol("Adj Close")]))

df


if length(x_val) != length(y_val) != length(aapl_df_subset)
    error("Arrays must be of the same size.")
end

# CREATE TARGETS BASED ON DECISIONS TO PREDICT 
targets = String[]

# Iterate over the arrays
for i in 1:length(x_val)
    # Combine the binary numbers into a string
    binary_str = string(Int(x_val[i])) * string(Int(y_val[i]))


    # Append to the new array
    push!(targets, binary_str)
end

# binary_strings now contains the combined binary strings
println(targets)

df[!, :opt_decisions] = targets;
df[!, :opt_decisions] = parse.(Int, df[!, :opt_decisions])

df_subset = select(df, Not(:opt_decisions))


seed = 15095

temp = IAI.OptimalTreeClassifier(random_seed = seed, localsearch = false, max_categoric_levels_before_warning= 100)
grid_cart = IAI.GridSearch(temp, minbucket = 1:10, criterion =:gini, max_depth = 2:6)


IAI.fit_cv!(grid_cart, df_subset, targets, validation_criterion = :gini, n_folds=5)
