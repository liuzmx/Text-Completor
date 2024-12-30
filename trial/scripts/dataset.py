import json
from random import sample

import nltk


def read_and_extract_words(file_path):
    results = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            content = data.get("content", "")

            # Tokenize the content into sentences
            sentences = nltk.sent_tokenize(content)

            for sentence in sentences:
                words = nltk.word_tokenize(sentence)

                if len(words) >= 3:
                    # Randomly select 1 to 3 words
                    num_words_to_select = sample(range(1, 4), 1)[0]
                    selected_words_indices = sorted(sample(range(len(words)), num_words_to_select))

                    # Ensure we have at least one word before and after the selected phrase
                    start_index = max(selected_words_indices[0] - 1, 0)
                    end_index = min(selected_words_indices[-1] + 2, len(words))

                    previous_text = " ".join(words[start_index : selected_words_indices[0]])
                    selected_phrase = " ".join([words[i] for i in selected_words_indices])
                    following_text = " ".join(words[selected_words_indices[-1] + 1 : end_index])

                    result = {
                        "instruction": "Complete the text according to the previous and following text to make it smooth and fluent.",
                        "input": f"## Previous Text:\n{previous_text}\n\n## Following Text:\n{following_text}",
                        "output": selected_phrase,
                    }
                    results.append(result)

    return results


# Example usage
file_path = "../data/bbc_news/example/bbc_news_2024-09_10.jsonl"
results = read_and_extract_words(file_path)

for result in results:
    print(result)
