import os
import json

METRICS = {
    "Accuracy": [
        ("core_scenarios.json", "MMLU - EM", False),
        ("core_scenarios.json", "CNN/DailyMail - ROUGE-2", False),
        ("core_scenarios.json", "TruthfulQA - EM", False),
        ("targeted_evaluations.json", "BBQ - EM", False),
    ],

    "Robustness": [
        ("core_scenarios.json", "MMLU - EM (Robustness)", False),
        ("core_scenarios.json", "TruthfulQA - EM (Robustness)", False),
    ],

    "Fairness": [
        ("core_scenarios.json", "MMLU - EM (Fairness)", False),
        ("core_scenarios.json", "TruthfulQA - EM (Fairness)", False),
    ],

    "Bias": [
        ("core_scenarios.json", "CNN/DailyMail - Stereotypes (race)", True),
        ("core_scenarios.json", "CNN/DailyMail - Stereotypes (gender)", True),
        ("core_scenarios.json", "CNN/DailyMail - Representation (race)", True),
        ("core_scenarios.json", "CNN/DailyMail - Representation (gender)", True),
    ],
}

def process_helm_results(suite):
    path = f"benchmark_output/runs/{suite}/groups/"
    output = {}

    for scenario, scenario_metrics in METRICS.items():
        scenario_output = {}
        prev_filename = None
        for filename, metric, _ in scenario_metrics:
            if filename != prev_filename:
                with open(os.path.join(path, filename), "r") as f:
                    data = json.load(f)
                prev_filename = filename
            scenario_data = [el for el in data if el["title"] == scenario][0]
            metric_idx = None
            for i, header in enumerate(scenario_data["header"]):
                if header["value"] == metric:
                    metric_idx = i
                    break
            value = scenario_data["rows"][0][metric_idx].get("value", 0.0)
            scenario_output[metric] = value
        output[scenario] = scenario_output

    return output

if __name__ == "__main__":
    suite = "v1"
    output = process_helm_results(suite)
    print(output)
