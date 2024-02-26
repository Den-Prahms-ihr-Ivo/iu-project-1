from pathlib import Path
from src.iu_project_1.function_finder import FunctionFinder

if __name__ == "__main__":
    p = Path.cwd()
    train_path = p / "data" / "train.csv"
    ideal_path = p / "data" / "ideal.csv"
    test_path = p / "data" / "test.csv"

    ff = FunctionFinder(train_set=train_path, ideal_set=ideal_path, test_set=test_path)

    nina = ff.get_best_function()
    ff.compare_functions(Path.cwd() / "tests" / "plots" / "popo.png")
    ff.plot_alpha(Path.cwd() / "tests" / "plots")
