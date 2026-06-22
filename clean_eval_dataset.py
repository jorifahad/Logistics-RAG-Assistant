import json


INPUT_FILE = "evaluation_dataset.json"
OUTPUT_FILE = "cleaned_evaluation_dataset.json"


# =========================================================
# QUESTIONS TO REMOVE
# =========================================================

BAD_QUESTION_PATTERNS = [

    # weak/vague
    "what does the document say",
    "who is the subject of the text",
    "what is the license",
    "on which traditional territories",
    "which program at conestoga",
    "what company produced the video",

    # weak metadata
    "how many chapters",
    "what year did",
    "where did the co-author",
]


# =========================================================
# ANSWERS TO REMOVE
# =========================================================

BAD_ANSWER_PATTERNS = [
    "not mentioned",
]


# =========================================================
# MAIN
# =========================================================

def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = []

    seen_questions = set()

    removed_count = 0

    for item in data:

        question = item["question"].strip()
        answer = item["ground_truth"].strip()

        question_lower = question.lower()
        answer_lower = answer.lower()

        remove_item = False

        # =================================================
        # REMOVE WEAK QUESTIONS
        # =================================================

        for pattern in BAD_QUESTION_PATTERNS:

            if pattern in question_lower:
                remove_item = True
                break

        # =================================================
        # REMOVE BAD ANSWERS
        # =================================================

        for pattern in BAD_ANSWER_PATTERNS:

            if pattern in answer_lower:
                remove_item = True
                break

        # =================================================
        # REMOVE DUPLICATES
        # =================================================

        if question_lower in seen_questions:
            remove_item = True

        seen_questions.add(question_lower)

        # =================================================
        # REMOVE SHORT QUESTIONS
        # =================================================

        if len(question.split()) < 5:
            remove_item = True

        # =================================================
        # SAVE CLEAN QUESTIONS
        # =================================================

        if not remove_item:

            cleaned_data.append(item)

        else:

            removed_count += 1

            print("\nREMOVED:")
            print("QUESTION:", question)
            print("ANSWER:", answer)

    # =====================================================
    # SAVE CLEAN DATASET
    # =====================================================

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        json.dump(
            cleaned_data,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 80)
    print("DATASET CLEANING FINISHED")
    print("=" * 80)

    print(f"Original Questions : {len(data)}")
    print(f"Removed Questions  : {removed_count}")
    print(f"Final Questions    : {len(cleaned_data)}")

    print(f"\nSaved cleaned dataset to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()