import os
import json
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import numpy as np
import evaluate

# --- Configuration ---
PREPARED_DATA_DIR = "prepared_driver_ner"
MODEL_OUTPUT_DIR = "trained_models"
MODEL_NAME = "xlm-roberta-base"

# Ensure output directories exist
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(MODEL_OUTPUT_DIR, "intent_classifier"), exist_ok=True)
os.makedirs(os.path.join(MODEL_OUTPUT_DIR, "ner_model"), exist_ok=True)
os.makedirs(os.path.join(MODEL_OUTPUT_DIR, "emotion_classifier"), exist_ok=True)

# --- Load Mappings ---
try:
    with open(os.path.join(PREPARED_DATA_DIR, "intent_to_id.json"), 'r', encoding='utf-8') as f:
        intent_to_id = json.load(f)
    with open(os.path.join(PREPARED_DATA_DIR, "id_to_intent.json"), 'r', encoding='utf-8') as f:
        id_to_intent = {int(k): v for k, v in json.load(f).items()}

    with open(os.path.join(PREPARED_DATA_DIR, "entity_to_id.json"), 'r', encoding='utf-8') as f:
        entity_to_id = json.load(f)
    with open(os.path.join(PREPARED_DATA_DIR, "id_to_entity.json"), 'r', encoding='utf-8') as f:
        id_to_entity = {int(k): v for k, v in json.load(f).items()}

    with open(os.path.join(PREPARED_DATA_DIR, "emotion_to_id.json"), 'r', encoding='utf-8') as f:
        emotion_to_id = json.load(f)
    with open(os.path.join(PREPARED_DATA_DIR, "id_to_emotion.json"), 'r', encoding='utf-8') as f:
        id_to_emotion = {int(k): v for k, v in json.load(f).items()}

except FileNotFoundError:
    print(f"Error: Mapping files not found in {PREPARED_DATA_DIR}.")
    print("Please ensure you have run data_preparation.py successfully.")
    exit()

# --- Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- Load Datasets ---
LANG_SUFFIX = "enUS_hiIN_goemotions_hinglish"

try:
    train_dataset = Dataset.from_json(os.path.join(PREPARED_DATA_DIR, f"train_processed_{LANG_SUFFIX}.jsonl"))
    dev_dataset = Dataset.from_json(os.path.join(PREPARED_DATA_DIR, f"dev_processed_{LANG_SUFFIX}.jsonl"))
    test_dataset = Dataset.from_json(os.path.join(PREPARED_DATA_DIR, f"test_processed_{LANG_SUFFIX}.jsonl"))
    print("Datasets loaded successfully.")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Dev dataset size: {len(dev_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

except FileNotFoundError:
    print(f"Error: Processed data files not found in {PREPARED_DATA_DIR}.")
    print("Please ensure you have run data_preparation.py successfully and the filenames match.")
    exit()

# --- Calculate Steps Per Epoch for Backward Compatibility ---
STEPS_PER_EPOCH = len(train_dataset) // 16
print(f"INFO: Calculated {STEPS_PER_EPOCH} steps per epoch for evaluation and saving.")

# --- Tokenization and Alignment Functions ---
def tokenize_and_align_labels_ner(examples):
    tokenized_inputs = tokenizer(
        examples["utterance"],
        padding="max_length",
        truncation=True,
        is_split_into_words=False, # This is correct if "utterance" is a string sentence
        max_length=128
    )

    labels = []
    for i, label_ids_for_example in enumerate(examples["entity_bio_tag_ids"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_id_aligned = []
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens (like CLS, SEP, PAD) get a label of -100
                label_id_aligned.append(-100)
            elif word_idx != previous_word_idx:
                # This is the first token of a new word.
                # Ensure word_idx is within the bounds of label_ids_for_example
                if word_idx < len(label_ids_for_example):
                    # Assign the original word's label to its first subtoken
                    label_id_aligned.append(label_ids_for_example[word_idx])
                else:
                    # Fallback for out-of-bounds word_idx (e.g., if a word was truncated entirely or data issue)
                    label_id_aligned.append(-100)
            else:
                # For subsequent tokens of the same word, set label to -100.
                # The model learns from the first subtoken's label.
                label_id_aligned.append(-100)
            previous_word_idx = word_idx
        labels.append(label_id_aligned)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def tokenize_for_sequence_classification(examples):
    return tokenizer(
        examples["utterance"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# --- Metrics Calculation Functions ---
def compute_metrics_intent(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy_metric = evaluate.load("accuracy")
    return accuracy_metric.compute(predictions=predictions, references=labels)

def compute_metrics_emotion(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy_metric = evaluate.load("accuracy")
    return accuracy_metric.compute(predictions=predictions, references=labels)

def compute_metrics_ner(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (labels == -100)
    true_predictions = [
        [id_to_entity[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_entity[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    seqeval_metric = evaluate.load("seqeval")
    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    
    flattened_results = {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
    for key, value in results.items():
        if isinstance(value, dict) and "f1" in value:
            flattened_results[f"{key}_f1"] = value["f1"]
    return flattened_results


# --- Initial Data Inspection for NER Labels ---
print("\n--- Initial NER Data Inspection (Raw Dataset) ---")
o_tag_id = entity_to_id.get("O", None) # Get the ID for 'O'

# Check training dataset
ner_labels_in_train = [item for sublist in train_dataset["entity_bio_tag_ids"] for item in sublist]
non_o_labels_train_count = sum(1 for l_id in ner_labels_in_train if l_id != o_tag_id and l_id != -100)
print(f"Number of non-'O' NER labels in raw train data: {non_o_labels_train_count}")
if non_o_labels_train_count > 0:
    example_non_o_labels = [id_to_entity[l_id] for l_id in ner_labels_in_train if l_id != o_tag_id and l_id != -100][:10]
    print(f"Example non-'O' labels from train: {example_non_o_labels}")
else:
    print("WARNING: No non-'O' labels found in raw train data. NER model will struggle.")

# Check dev dataset
ner_labels_in_dev = [item for sublist in dev_dataset["entity_bio_tag_ids"] for item in sublist]
non_o_labels_dev_count = sum(1 for l_id in ner_labels_in_dev if l_id != o_tag_id and l_id != -100)
print(f"Number of non-'O' NER labels in raw dev data: {non_o_labels_dev_count}")
if non_o_labels_dev_count > 0:
    example_non_o_labels = [id_to_entity[l_id] for l_id in ner_labels_in_dev if l_id != o_tag_id and l_id != -100][:10]
    print(f"Example non-'O' labels from dev: {example_non_o_labels}")
else:
    print("WARNING: No non-'O' labels found in raw dev data. NER evaluation will be 0.")
print("--- End Initial NER Data Inspection ---\n")


# --- Training Intent Classification Model ---
# print("\n--- Training Intent Classification Model ---")
# intent_train_dataset = train_dataset.map(tokenize_for_sequence_classification, batched=True).rename_columns({"intent_label_id": "labels"})
# intent_dev_dataset = dev_dataset.map(tokenize_for_sequence_classification, batched=True).rename_columns({"intent_label_id": "labels"})
# intent_train_dataset = intent_train_dataset.remove_columns(["utterance", "tokens", "entity_bio_tags", "entity_bio_tag_ids", "emotion_label", "emotion_label_id", "language"])
# intent_dev_dataset = intent_dev_dataset.remove_columns(["utterance", "tokens", "intent_label", "intent_label_id", "entity_bio_tags", "entity_bio_tag_ids", "emotion_label", "language"])

# intent_training_args = TrainingArguments(
#     output_dir=os.path.join(MODEL_OUTPUT_DIR, "intent_classifier"),
#     do_eval=True,
#     eval_strategy="steps", # Changed from evaluation_strategy
#     eval_steps=STEPS_PER_EPOCH,
#     save_steps=STEPS_PER_EPOCH,
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     logging_dir=os.path.join(MODEL_OUTPUT_DIR, "intent_classifier", "logs"),
#     load_best_model_at_end=True,
#     metric_for_best_model="accuracy",
#     greater_is_better=True,
#     report_to="none"
# )

# intent_model = AutoModelForSequenceClassification.from_pretrained(
#     MODEL_NAME,
#     num_labels=len(id_to_intent),
#     id2label=id_to_intent,
#     label2id=intent_to_id
# )

# intent_trainer = Trainer(
#     model=intent_model,
#     args=intent_training_args,
#     train_dataset=intent_train_dataset,
#     eval_dataset=intent_dev_dataset,
#     compute_metrics=compute_metrics_intent,
# )

# intent_trainer.train()
# intent_trainer.save_model(os.path.join(MODEL_OUTPUT_DIR, "intent_classifier", "final_model"))
# tokenizer.save_pretrained(os.path.join(MODEL_OUTPUT_DIR, "intent_classifier", "final_model"))
# print("Intent Classification model training complete and saved.")
print("\n--- Skipping Intent Classification Model Training ---") # Added this for clarity


# --- Training NER Model ---
print("\n--- Training NER Model ---")
ner_train_dataset = train_dataset.map(tokenize_and_align_labels_ner, batched=True)
ner_dev_dataset = dev_dataset.map(tokenize_and_align_labels_ner, batched=True)

# --- Debugging NER Tokenization Output ---
print("\n--- NER Tokenization and Alignment Output Sample (Mapped Dataset) ---")
# Check the first few examples from the mapped NER train dataset
for k in range(min(5, len(ner_train_dataset))): # Check first 5 examples
    print(f"\nExample {k+1}:")
    original_utterance = train_dataset[k]['utterance']
    original_bio_tags = train_dataset[k]['entity_bio_tag_ids']
    
    # Manually tokenize to get word_ids for comparison
    manual_tokenized = tokenizer(original_utterance, is_split_into_words=False, max_length=128, truncation=True)
    
    print(f"Original Utterance: {original_utterance}")
    print(f"Original Word BIO Tags: {[id_to_entity[tag_id] for tag_id in original_bio_tags]}")
    print(f"Manual Tokenized Input IDs: {manual_tokenized['input_ids']}")
    print(f"Manual Tokenized Tokens: {tokenizer.convert_ids_to_tokens(manual_tokenized['input_ids'])}")
    print(f"Manual Word IDs: {manual_tokenized.word_ids()}")

    # Access the already mapped data
    mapped_example = ner_train_dataset[k]
    print(f"Mapped Tokenized Input IDs: {mapped_example['input_ids']}")
    print(f"Mapped Aligned Labels (IDs): {mapped_example['labels']}")
    print(f"Mapped Aligned Labels (Names): {[id_to_entity[l] if l != -100 else 'IGNORE' for l in mapped_example['labels']]}")
print("--- End NER Tokenization Output Sample ---\n")
# --- End Debugging ---


ner_train_dataset = ner_train_dataset.remove_columns(["utterance", "tokens", "intent_label", "intent_label_id", "entity_bio_tags", "emotion_label", "emotion_label_id", "language"])
ner_dev_dataset = ner_dev_dataset.remove_columns(["utterance", "tokens", "intent_label", "intent_label_id", "entity_bio_tags", "emotion_label", "emotion_label_id", "language"])

ner_training_args = TrainingArguments(
    output_dir=os.path.join(MODEL_OUTPUT_DIR, "ner_model"),
    do_eval=True,
    eval_strategy="steps", # Changed from evaluation_strategy
    eval_steps=STEPS_PER_EPOCH,
    save_steps=STEPS_PER_EPOCH,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=os.path.join(MODEL_OUTPUT_DIR, "ner_model", "logs"),
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none"
)

ner_model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(id_to_entity),
    id2label=id_to_entity,
    label2id=entity_to_id
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

ner_trainer = Trainer(
    model=ner_model,
    args=ner_training_args,
    train_dataset=ner_train_dataset,
    eval_dataset=ner_dev_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics_ner,
)

ner_trainer.train()
ner_trainer.save_model(os.path.join(MODEL_OUTPUT_DIR, "ner_model", "final_model"))
tokenizer.save_pretrained(os.path.join(MODEL_OUTPUT_DIR, "ner_model", "final_model"))
print("NER model training complete and saved.")

'''# --- Training Emotion Classification Model ---
print("\n--- Training Emotion Classification Model ---")
emotion_train_dataset = train_dataset.map(tokenize_for_sequence_classification, batched=True).rename_columns({"emotion_label_id": "labels"})
emotion_dev_dataset = dev_dataset.map(tokenize_for_sequence_classification, batched=True).rename_columns({"emotion_label_id": "labels"})
emotion_train_dataset = emotion_train_dataset.remove_columns(["utterance", "tokens", "intent_label", "intent_label_id", "entity_bio_tags", "entity_bio_tag_ids", "emotion_label", "language"])
emotion_dev_dataset = emotion_dev_dataset.remove_columns(["utterance", "tokens", "intent_label", "intent_label_id", "entity_bio_tags", "entity_bio_tag_ids", "emotion_label", "language"])

emotion_training_args = TrainingArguments(
    output_dir=os.path.join(MODEL_OUTPUT_DIR, "emotion_classifier"),
    do_eval=True,
    eval_strategy="steps", # Changed from evaluation_strategy
    eval_steps=STEPS_PER_EPOCH,
    save_steps=STEPS_PER_EPOCH,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=os.path.join(MODEL_OUTPUT_DIR, "emotion_classifier", "logs"),
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none"
)

emotion_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(id_to_emotion),
    id2label=id_to_emotion,
    label2id=emotion_to_id
)

emotion_trainer = Trainer(
    model=emotion_model,
    args=emotion_training_args,
    train_dataset=emotion_train_dataset,
    eval_dataset=emotion_dev_dataset,
    compute_metrics=compute_metrics_emotion,
)

emotion_trainer.train()
emotion_trainer.save_model(os.path.join(MODEL_OUTPUT_DIR, "emotion_classifier", "final_model"))
tokenizer.save_pretrained(os.path.join(MODEL_OUTPUT_DIR, "emotion_classifier", "final_model"))
print("Emotion Classification model training complete and saved.")'''

print("\nAll models trained and saved successfully!")