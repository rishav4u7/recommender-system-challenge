# Build a Recommendation Model under Realistic Constraints
(Version 1.0, 2026-02-14)

## Overview

Assume that you are a data scientist at a marketing company.
The company operates a platform that sends ads for various items (products) to users via messaging apps (e.g., WhatsApp, Facebook Messenger, LINE).
Each message contains one item to promote, and users can click it to view more details or make a purchase.
Your clients want to promote their items, and your company wants to recommend relevant products to users based on historical interactions and user/item features.

Your task is to build a recommendation system that suggests items to users.
You are given a dataset of user–item interactions, along with basic user and item features.

Your goal is to build **at least one recommendation model** and explain your design choices.


## Constraints
- Submission Format
  - Please submit a **Python script** (**not a notebook**). It can be a single file or a set of multiple files, but it must be runnable from the command line.
  - You must include a **README.md** file. In the README, describe points specified in the "Documentation" section below.
- Expected time limit: 90 minutes (approximate)
- AI tools are allowed
- No external data
- No AutoML
- Python only


## Required Tasks

### Coding
Implement **at least one** recommendation model in Python. Multiple models are welcome but not required.
The Python script should:
1. Split the dataset into training, validation, and test sets. How to split the data is up to you, but use approximately 10% of the data for the test set. Explain your strategy in the README.
2. Train the recommendation model on the training set. If needed, tune hyperparameters on the validation set. Assume model retraining must complete within 30 minutes on a standard laptop.
3. Predict recommendations for users that appear in the test set.
4. Save the recommended items for each test user in a CSV file with the format below.
5. Compute an evaluation metric on the test set and print the result. You can choose the metric and the evaluation strategy, but explain your choice in the README.


The output CSV file must have the following format:
```
USER_ID,RECOMMENDED_ITEMS
[USER_ID_1],[ITEM_ID_1] [ITEM_ID_2] ... [ITEM_ID_N]
...
```
Specifically, the `RECOMMENDED_ITEMS` column should contain up to 10 recommended item IDs for each user (`USER_ID`), separated by spaces. Items must be ordered by rank: the leftmost item is the highest-ranked (most recommended) item. For example:
```
USER_ID,RECOMMENDED_ITEMS
8x3a2,2p9k4 7m4n8 3q8r1 9w1t5 4k6j3 8n2b7 1z5c9 6h9d2 5r3f8 2y7g1
4m7b6,1a5x9 9b3c2 7d8m4 4e2n7 8f6p1 
...
```
The first user (8x3a2) has 10 recommended items, while the second user (4m7b6) has only 5 recommended items.

If no items are recommended for a user, the `RECOMMENDED_ITEMS` column should be left empty for that user:
```
USER_ID,RECOMMENDED_ITEMS
8x3a2,
4m7b6,
...
```


### Documentation (README.md)
Write a concise README including the following points. Bullet points are highly encouraged.

- **Setup & Execution**:
  - Environment requirements (Python version, libraries).
  - Command line instructions to run the script.
- **Model Design & Evaluation Strategy**:
  - Explain your **recommendation approach**, **data splitting strategy**, and **evaluation metric and strategy**.
  - Focus on **why** these were chosen.
- **Dataset Insights & Challenges**:
  - Identify potential biases or challenges (e.g., selection bias, cold start).
  - Briefly estimate the **theoretical upper bound** of your chosen metric for this dataset.
- **AI Tool Usage**:
  - How you used AI tools (if any) during your process.


## Submission
Create a zip file containing your code and README, upload it to a file sharing service (e.g., Google Drive, Dropbox), and share the link with us.

## Confidentiality Notice
Please keep the challenge material and your solution confidential. Do not share the link to the zip file publicly or with any other person. Do not share the challenge material we provided to any other person or on public platforms.

## Files
This zip contains the following files.
- `INSTRUCTION.md`: This file
- Data
  - `interactions.csv`: Interaction data
  - `users.csv`: User features
  - `items.csv`: Item features


## Data Schema

### interactions.csv

| USER_ID | ITEM_ID | INTERACTION | TIMESTAMP           |
|---------|---------|-------------|---------------------|
| a1b2c   | x9y8z   | 1           | 2024-03-15 14:30:00 |

- `USER_ID`: unique identifier for each user.
- `ITEM_ID`: unique identifier for each item.
- `INTERACTION`: user interaction with the item in the message:
  - 1 = the user clicked the sent item
  - 0 = the user did not click the sent item
- `TIMESTAMP`: when the message was sent to the user (not when they clicked).

Note: Only items that were sent to users are observed.



### users.csv

| USER_ID | AGE_BUCKET |
|---------|------------|
| a1b2c   | 30-39      |

- `USER_ID`: unique identifier for each user.
- `AGE_BUCKET`: age group of the user (e.g., 10-19, 20-29, etc.)

### items.csv

| ITEM_ID | CATEGORY | PRICE_BUCKET |
|---------|----------|--------------|
| x9y8z   | Apparel  | $51-$100     |

- `ITEM_ID`: unique identifier for each item.
- `CATEGORY`: category of the item (e.g., Apparel, Bags, etc.)
- `PRICE_BUCKET`: price range of the item (e.g., $0-$50, $51-$100, etc.)

