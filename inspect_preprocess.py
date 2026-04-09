import pandas as pd

from config import TRAIN_PATH, TEST_PATH
from preprocess import clean_prompt


def _preview(s: str, n: int = 250) -> str:
    if s is None:
        return ""
    s = str(s)
    return s[:n] + ("..." if len(s) > n else "")


def main():
    print("Loading raw CSVs...")
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)

    print("\nTrain columns:", list(train_raw.columns))
    print("Test columns:", list(test_raw.columns))

    # Basic schema validation
    required_train = {"Id", "Prompt", "Answer", "Target"}
    missing_train = required_train - set(train_raw.columns)
    if missing_train:
        raise ValueError(f"Train is missing columns: {sorted(missing_train)}")

    required_test = {"Id", "Prompt", "Answer"}
    missing_test = required_test - set(test_raw.columns)
    if missing_test:
        raise ValueError(f"Test is missing columns: {sorted(missing_test)}")

    # Prompt cleaning behavior check
    print("\nComputing clean_prompt() effect on train prompts...")
    train_prompts = train_raw["Prompt"].fillna("")
    cleaned = train_prompts.apply(clean_prompt)
    empty_rate = (cleaned.str.len() == 0).mean()
    print(f"clean_prompt empty-string rate: {empty_rate:.4f}")

    print("\nSample Prompt before/after clean_prompt():")
    for i in range(min(3, len(train_raw))):
        pid = train_raw.iloc[i]["Id"]
        prompt_before = train_raw.iloc[i]["Prompt"]
        prompt_after = cleaned.iloc[i]
        print(f"\n--- Sample #{i} (Id={pid}) ---")
        print("Before:", _preview(prompt_before))
        print("After: ", _preview(prompt_after))


if __name__ == "__main__":
    main()

