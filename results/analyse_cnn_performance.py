import json

PATH_TO_RESULTS = "results/results.json"
PATH_TO_ANALYSIS = "results/analysis.json"


def get_accuracies(prediction_data: dict) -> dict:
    accuracies = {0: {}, 1: {}, 2: {}, 3: {}}
    for level in [0, 1, 2, 3]:
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for image in prediction_data["predicted_grades"]:
            prediction = image["predicted_grade"]
            truth = image["truth"]
            if prediction == level and truth == level:
                true_positive += 1
            elif prediction == level and truth != level:
                false_positive += 1
            elif prediction != level and truth != level:
                true_negative += 1
            elif prediction != level and truth == level:
                false_negative += 1

        accuracies[level] = {
            "accuracy": (
                (true_positive + true_negative)
                / (true_negative + true_positive + false_positive + false_negative)
            )
            * 100,
            "sensitivity": (true_positive / (true_positive + false_positive)) * 100,
            "specificity": (true_negative / (true_negative + false_negative)) * 100,
        }
    return accuracies


def get_averages(accuracies: dict) -> dict:
    avg_accuracy = 0
    avg_sensitivity = 0
    avg_specificity = 0
    for level in accuracies:
        avg_accuracy += accuracies[level]["accuracy"]
        avg_sensitivity += accuracies[level]["sensitivity"]
        avg_specificity += accuracies[level]["specificity"]
    averages = {
        "average accuracy": avg_accuracy / 4,
        "average sensitivity": avg_sensitivity / 4,
        "average specificity": avg_specificity / 4,
    }

    return averages


def analyse_results() -> None:
    with open(PATH_TO_RESULTS, "r") as file:
        prediction_data = json.load(file)
    accuracies = get_accuracies(prediction_data)
    averages = get_averages(accuracies)
    analysis_results = {"overall": averages, "level specific": accuracies}
    with open(PATH_TO_ANALYSIS, "w", newline="") as f:
        json.dump(analysis_results, f, indent=4)


if __name__ == "__main__":
    analyse_results()
