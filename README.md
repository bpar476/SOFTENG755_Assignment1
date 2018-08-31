# Ben Partridge SOFTENG755 Assignment 1

## Dependencies

In order to run the code you will need the following dependencies installed.

- scikit-learn
- numpy
- scipy
- pandas
- python (built with version 3.5.2)

## Running the code

Execute each script from its source directory. For example, if you want to run the world cup script, make sure you are in the directory `src/world_cup`. Then run the following command: `python -W ignore world_cup.py`. The -W flag is to suppress warnings, I've had a lot of warnings show up when running the scripts but they don't really matter. Suppressing the warnings makes the output a lot easier to interpret.

If you want to add your own validation data, supply the `--test-data` argument, followed by the RELATIVE path of the test data, in csv form. The test data must have the same columns as the given data (however it doesn't need to have the target column). To make it as easy as possible, just put the test data in the same directory as the script.

`python -W ignore world_cup.py --test-data <RELATIVE_PATH_TO_VALIDATION_CSV>`

## Note

If you're running the world cup script and the Team1 or Team2 columns contain a different set of teams to that from the dataset originally given to us, you're going to need to run that script with the `-d` flag. This is because the trained model uses a OneHotEncoder of the dataset originally given to us which affects the dimensionality. The new data will be encoded with a OneHotEncoder also but if the set of values are different, the processed features will have different dimensionality to what is expected.

Any issues running the code contact me via email at: bpar476@aucklanuni.ac.nz
