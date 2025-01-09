import json


def count(file_path: str):
    bins = {f"{i*0.05:.2f}-{(i+1)*0.05:.2f}": [0, 0.0] for i in range(20)}
    total_scores = 0
    with open(file_path, "r") as f:
        for item in json.load(f):
            bin_key = f"{int(item['similarity'] // 0.05) * 0.05:.2f}-{int(item['similarity'] // 0.05 + 1) * 0.05:.2f}"
            bins[bin_key][0] += 1
            total_scores += 1
    for key in bins:
        bins[key][1] = bins[key][0] / total_scores
    return bins


if __name__ == "__main__":
    # res = count("result/test_result_with_knowledge_history_1K.json")
    res = count("result/test_result_without_knowledge_history_1K.json")
    for key, value in res.items():
        print(f"{key}\t{value[0]}\t{value[1]}")
