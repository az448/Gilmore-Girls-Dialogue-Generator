import re
import nltk
import sqlite3
import pandas as pd
import markovify
import matplotlib.pyplot as plt
import seaborn as sns4

# Raw CSV file information
csv_file_name = 'Gilmore_Girls_Lines.csv'
speaker_column_name = 'Character'
line_column_name = 'Line'

# Loading the CSV into a pandas DataFrame
print("\n--- Loading Raw Data into DataFrame ---")
raw_dialogue_df = pd.read_csv(csv_file_name)
print(f"Loaded {len(raw_dialogue_df)} lines from {csv_file_name}")
print("First 5 rows of raw data:")
print(raw_dialogue_df.head())

# Check that the nesseccary columns exist
if speaker_column_name not in raw_dialogue_df.columns or line_column_name not in raw_dialogue_df.columns:
    raise ValueError(f"CSV must contain '{speaker_column_name}' and '{line_column_name}' columns.")

# Handle potential missing values
raw_dialogue_df = raw_dialogue_df[[speaker_column_name, line_column_name]].dropna()
raw_dialogue_df.columns = ['speaker', 'line'] # Rename columns for consistency
print(f"Using {len(raw_dialogue_df)} non-null lines after column selection and dropna.")

# Store data in SQLite satabase
conn = sqlite3.connect('gilmore_girls_dialogue.db')
cursor = conn.cursor()

# Create a table for the dialogue
cursor.execute('DROP TABLE IF EXISTS dialogue') # Ensure table does not exist yet
cursor.execute('''
    CREATE TABLE dialogue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        speaker TEXT,
        line TEXT
    )
''')

# Insert data from DataFrame into dialogue table
for row in raw_dialogue_df.itertuples(index=False):
    cursor.execute('INSERT INTO dialogue (speaker, line) VALUES (?, ?)', (row.speaker, row.line))
conn.commit()
print("Raw Dialogue stored in gilmore_girls_dialogue.db")

# Load data using pandas
df = pd.read_sql_query("SELECT speaker, line FROM dialogue", conn)
conn.close()

# Function to clean the dialogue
def clean_dialogue(text):
    if not isinstance(text, str): #remove anything that is not a string
        return ""
    text = re.sub(r'\([^)]*\)|\[[^\]]*\]', '', text) #remove actions in parentheses
    text = re.sub(r'\s+', ' ', text).strip() #remove double spaces
    text = text.lower() #convert all to lowercase
    return text

# Use function to clean each line
print("\n--- Cleaning Data ---")
df['cleaned_line'] = df['line'].apply(clean_dialogue)

# Combine all cleaned dialogue into one string for markovify
full_cleaned_dialogue = " ".join(df['cleaned_line'].dropna().tolist())
print(f"Total lines loaded from DB: {len(df)}")
print(f"Sample cleaned dialogue (first 200 chars): {full_cleaned_dialogue[:200]}...") # Print first 200 chars

# Markov Chain Model
print("\n--- Building Markov Chain Model ---")
text_model = None # Initialize to None
try:
    text_model = markovify.Text(full_cleaned_dialogue, state_size=3) # can experiment with different state_sizes
    print("Markov chain model built successfully.")
except Exception as e:
    print(f"Error building Markov model: {e}. This often happens with very small or repetitive data")
    print("Try reducing state_size or providing more diverse text")

# New Text Generation
print("\n--- Generating New Dialogue ---")
generated_sentences = []
print ("Generated Gilmore Girls-like lines:")
for i in range(5):
    sentence = text_model.make_sentence(tries=100) 
    if sentence:
        print(f"- {sentence}")
        generated_sentences.append(sentence)
    else:
        print(f"- (Could not generate a unique sentence after 100 tries)")

print("Generated a short sentence:")
short_sentence = text_model.make_short_sentence(max_chars=80, tries=100)
if short_sentence:
    print(f"- {short_sentence}")
    generated_sentences.append(short_sentence)
else:
    print(f"- (Could not generate a short sentence after 100 tries)")

# Load generated sentences into a pandas DataFrame
generated_df = pd.DataFrame({'line': generated_sentences})
generated_df['cleaned_line'] = generated_df['line'].apply(clean_dialogue)
full_cleaned_generated_dialogue = " ".join(generated_df['cleaned_line'].dropna().tolist())


# Data Analysis and Visualization
print("\n--- Performing Data Analysis and Visualization ---")

# Tokenizing and examining original dialogue
original_all_words = nltk.word_tokenize(full_cleaned_dialogue)
original_all_words = [word for word in original_all_words if word.isalnum()]
original_word_freq = pd.Series(original_all_words).value_counts().head(20)

df['original_sentence_length'] = df['cleaned_line'].apply(lambda x: len(x.split()) if x else 0)
original_sentence_lengths = df[df['original_sentence_length'] > 0]['original_sentence_length']

# Tokenizing and examining generated data
generated_all_words = nltk.word_tokenize(full_cleaned_generated_dialogue)
generated_all_words = nltk.word_tokenize(full_cleaned_generated_dialogue)
generated_word_freq = pd.Series(generated_all_words).value_counts().head(20)

generated_df['generated_sentence_length'] = generated_df['cleaned_line'].apply(lambda x: len(x.split()) if x else 0)
generated_sentence_lengths = generated_df[generated_df['generated_sentence_length'] > 0]['generated_sentence_length']

# Plots for analysis and comparison
# Plot 1: Dialogue Contribution by Speaker using SQL query (only original data)
print("\n--- Querying original dialogue database for character contributions ---")
conn_query = sqlite3.connect('gilmore_girls_dialogue.db')
speaker_counts_query = """
    SELECT speaker, COUNT(*) as line_count 
    FROM dialogue 
    GROUP BY speaker 
    ORDER BY line_count DESC 
    LIMIT 10
"""
speaker_counts = pd.read_sql_query(speaker_counts_query, conn_query)
conn_query.close()

plt.figure(figsize=(10, 6))
sns4.barplot(x='speaker', y='line_count', data=speaker_counts, palette='mako', hue='speaker', legend=False)
plt.title('Number of Lines per Character (Original Dialogue)')
plt.xlabel('Speaker')
plt.ylabel('Number of Lines')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
print("Generated plot: Number of Lines per Character (Original Dialogue).")
plt.show()

# Plot 2
print("\n--- Plotting Dialogue Length Distribution (Categorized, Original Data) ---")
conn_query = sqlite3.connect('gilmore_girls_dialogue.db')
dialogue_length_dist_query = """
    SELECT
        CASE
            WHEN (LENGTH(line) - LENGTH(REPLACE(line, ' ', '')) + 1) <= 5 THEN 'Very Short (1-5 words)'
            WHEN (LENGTH(line) - LENGTH(REPLACE(line, ' ', '')) + 1) <= 15 THEN 'Short (6-15 words)'
            WHEN (LENGTH(line) - LENGTH(REPLACE(line, ' ', '')) + 1) <= 30 THEN 'Medium (16-30 words)'
            ELSE 'Long (30+ words)'
        END AS line_length_category,
        COUNT(*) AS number_of_lines
    FROM dialogue
    WHERE line IS NOT NULL AND line != ''
    GROUP BY line_length_category
    ORDER BY number_of_lines DESC;
"""
dialogue_length_dist = pd.read_sql_query(dialogue_length_dist_query, conn_query)
conn_query.close()

category_order = ['Very Short (1-5 words)', 'Short (6-15 words)', 'Medium (16-30 words)', 'Long (30+ words)']
dialogue_length_dist['line_length_category'] = pd.Categorical(dialogue_length_dist['line_length_category'], categories=category_order, ordered=True)
dialogue_length_dist = dialogue_length_dist.sort_values('line_length_category')

plt.figure(figsize=(10, 6))
sns4.barplot(x='line_length_category', y='number_of_lines', data=dialogue_length_dist, palette='flare', hue='line_length_category', legend=False)
plt.title('Distribution of Dialogue Line Lengths (Original Data)')
plt.xlabel('Line Length Category')
plt.ylabel('Number of Lines')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
print("Generated plot: Dialogue Length Distribution (Categorized, Original Data).")
plt.show()

# Plot 3: Top 20 most frequent words (original vs generated)
fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
# Original
sns4.barplot(x=original_word_freq.index, y=original_word_freq.values, palette='viridis', ax=axes[0], hue=original_word_freq.index, legend=False)
axes[0].set_title('Top 20 Most Frequent Words (Original Dialogue)')
axes[0].set_xlabel('Words')
axes[0].set_ylabel('Frequency')
axes[0].tick_params(axis='x', rotation=45)
# Generated
sns4.barplot(x=generated_word_freq.index, y=generated_word_freq.values, palette='magma', ax=axes[1], hue=generated_word_freq.index, legend=False)
axes[1].set_title('Top 20 Most Frequent Words (Generated Dialogue)')
axes[1].set_xlabel('Words')
axes[1].set_ylabel('Frequency')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
print("Displayed Plot: Top 20 Most Frequent Words (Original vs Generated)")
plt.show()

# Plot 4: Distribution of sentence lengths (original vs generated)
plt.figure(figsize=(12, 7))
sns4.histplot(original_sentence_lengths, bins=range(0, original_sentence_lengths.max() + 5, 5), kde=True, color='skyblue', label='Original', stat='density', alpha=0.7)
sns4.histplot(generated_sentence_lengths, bins=range(0, generated_sentence_lengths.max() + 5, 5), kde=True, color='salmon', label='Generated', stat='density', alpha=0.7)
plt.title('Distribution of Sentence Lengths (Original vs Generated)')
plt.xlabel('Sentence Length (words)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
print("Displayed plot: Distribution of Sentence Lengths (Original vs. Generated).")
plt.show()