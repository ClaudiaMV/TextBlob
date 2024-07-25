import pandas as pd
import numpy as np
import os
from docx import Document
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Get the list of stopwords
stop_words = set(stopwords.words('english'))

# Define the text cleaning function
def clean_text(text):
    # Remove special characters and digits
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text
    words = TextBlob(text).words
    
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    
    # Lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join words back into a single string
    cleaned_text = ' '.join(words)
    
    return cleaned_text

# Get the absolute path of the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the document
document_path = os.path.join(script_dir, 'Text')

# Print the path to verify
print(f"Document path: {document_path}")

# Define the main project and text folders path
PROJECT_FOLDER = "Project/path"
TEXT_FOLDERS_PATH = PROJECT_FOLDER  # Define main text folders path

# Obtain a list of all participant folder paths
PARTICIPANT_FOLDERS = [
    os.path.join(TEXT_FOLDERS_PATH, x) 
    for x in os.listdir(TEXT_FOLDERS_PATH) 
    if os.path.isdir(os.path.join(TEXT_FOLDERS_PATH, x)) and x != '.DS_Store'
]
PARTICIPANT_FOLDERS.sort()  # Sort the list of participant folders

# Create a nested list of narrative file paths
ALL_FILE_PATHS = []
for folder in PARTICIPANT_FOLDERS:
    file_paths = [
        os.path.join(folder, file) 
        for file in os.listdir(folder) 
        if file.endswith('.docx') and file != '.DS_Store'
    ]
    ALL_FILE_PATHS.extend(file_paths)

# Display list of all narrative file paths
for path in ALL_FILE_PATHS:
    print(path)

# Create a text data frame to contain path info, file names, participant number and text
text_df = pd.DataFrame()  # initialize empty data frame
text_df['Path'] = ALL_FILE_PATHS  # add column for paths
text_df['File_name'] = text_df['Path'].apply(lambda x: os.path.basename(x))  # add column for file names
text_df['Participant'] = text_df['Path'].apply(lambda x: x.split("/")[-2])
text_df['Participant_number'] = text_df['Participant'].apply(lambda x: int(re.findall(r'\d+', x)[0]))  # Extract participant number using regex

# Add column for narration number (in cases where narration number is in the file name)
text_df['Narration_number'] = text_df['File_name'].apply(lambda x: int(x.split(".")[0].split("_")[-1]) if x.split(".")[0].split("_")[-1].isnumeric() else np.nan)
text_df.sort_values(['Participant_number', 'Narration_number'], axis=0, ascending=True, inplace=True)  # sort the rows by Participant number

# Function to read the contents of a .docx file into a string variable
def read_docx(path_to_docx):
    doc = Document(path_to_docx)
    # Initialize empty string variable
    doc_content = ""
    # Iterate over paragraphs and append them to the string variable
    for i, paragraph in enumerate(doc.paragraphs):
        doc_content += paragraph.text
        if i != 0:
            doc_content += " "
    return doc_content

# Create text column
text_df['Text'] = text_df['Path'].apply(lambda x: read_docx(x))

# Clean the text data
text_df['Cleaned_Text'] = text_df['Text'].apply(clean_text)

# Save text data frame to project folder
text_df.to_csv(os.path.join(PROJECT_FOLDER, 'all_participants_raw_text.csv'), index=False, header=True)

# Function to obtain subjectivity using TextBlob
def compute_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# Compute subjectivity for both raw and cleaned text
text_df['Raw_Subjectivity'] = text_df['Text'].apply(compute_subjectivity)
text_df['Cleaned_Subjectivity'] = text_df['Cleaned_Text'].apply(compute_subjectivity)

# Create data frame to store subjectivity scores
subjectivity_df = text_df.copy(deep=True)

# Write subjectivity scores to csv file
subjectivity_df.to_csv(os.path.join(PROJECT_FOLDER, 'TextBlob_subjectivity.csv'), index=False, header=True)

# Display a sample of the data frame to inspect the changes
print(subjectivity_df[['Participant', 'File_name', 'Raw_Subjectivity', 'Cleaned_Subjectivity', 'Cleaned_Text']].head())
