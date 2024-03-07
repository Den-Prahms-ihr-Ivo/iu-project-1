from pathlib import Path

import src.iu_project_1 as IU

if __name__ == "__main__":
    # Performs all necessary calulations of the given assignment.

    p = Path.cwd()
    train_path = p / "data" / "train.csv"
    ideal_path = p / "data" / "ideal.csv"
    test_path = p / "data" / "test.csv"

    ff: IU.FunctionFinderBaseClass = IU.FunctionFinder(
        train_set=train_path, ideal_set=ideal_path, test_set=test_path
    )

    ff.plot_functions(Path.cwd() / "tests" / "plots")
    ff.save_test_data_to_sql()
