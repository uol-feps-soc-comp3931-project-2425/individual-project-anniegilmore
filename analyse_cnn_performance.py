import json
from pathlib import Path

PATH_TO_PREDICTIONS = Path("results/predictions.json")
PATH_TO_ANALYSIS = Path("results/analysis.json")


def get_averages(accuracies: dict) -> dict:
    """ Calculates overall accuracy, sensitivity and specificity for model as a whole """
    avg_accuracy: int = 0
    avg_sensitivity: int = 0
    avg_specificity: int = 0
    for level in accuracies:
        avg_accuracy += accuracies[level]["accuracy"]
        avg_sensitivity += accuracies[level]["sensitivity"]
        avg_specificity += accuracies[level]["specificity"]
    averages: dict = {
        "average accuracy": avg_accuracy / 4,
        "average sensitivity": avg_sensitivity / 4,
        "average specificity": avg_specificity / 4,
    }

    return averages

def get_accuracies(prediction_data: dict) -> dict:
    """ Calculates and collates accuracy, sensitivity and specificty for each class """
    accuracies: dict = {0: {}, 1: {}, 2: {}, 3: {}}
    for level in [0, 1, 2, 3]:
        metrics: dict = {'true_positive': 0, 'true_negative': 0, 'false_positive': 0, 'false_negative': 0}
        for image in prediction_data["predicted_grades"]:
            prediction = image["predicted_grade"]
            truth = image["truth"]
            if prediction == level and truth == level:
                metrics['true_positive'] += 1
            elif prediction == level and truth != level:
                metrics['false_positive'] += 1
            elif prediction != level and truth != level:
                metrics['true_negative'] += 1
            elif prediction != level and truth == level:
                metrics['false_negative'] += 1

        accuracies[level] = {
            "accuracy": (
                (metrics['true_positive'] + metrics['true_negative'])
                / len(prediction_data["predicted_grades"])
            )
            * 100,
            "sensitivity": (metrics['true_positive'] / (metrics['true_positive'] + metrics['false_positive'])) * 100,
            "specificity": (metrics['true_negative'] / (metrics['true_negative'] + metrics['false_negative'])) * 100,
        }
    return accuracies


def analyse_results() -> None:
    """ Reads in model predictions for test set and evaluates model performance """
    with open(PATH_TO_PREDICTIONS, "r") as file:
        prediction_data = json.load(file)
    accuracies: dict = get_accuracies(prediction_data)
    averages: dict = get_averages(accuracies)
    analysis_results: dict = {"overall": averages, "level specific": accuracies}
    with open(PATH_TO_ANALYSIS, "w", newline="") as f:
        json.dump(analysis_results, f, indent=4)
    return


if __name__ == "__main__":
    analyse_results()
