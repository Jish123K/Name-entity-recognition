import spacy

# Load the spacy model

nlp = spacy.load("en_core_web_sm")

# Preprocess the data

def preprocess_text(text):

  # Remove HTML tags

  text = re.sub(r"<[^>]+>", "", text)

  # Remove special characters

  text = re.sub(r"[^\w\s]", "", text)

  # Remove punctuation marks

  text = re.sub(r"[,.!?]", "", text)

  # Remove stop words

  stop_words = set(stopwords.words("english"))

  text = " ".join([word for word in text.split() if word not in stop_words])

  return text

# Tokenize the data

def tokenize_text(text):

  return nlp.tokenizer(text)

# Load the pre-trained model

model = spacy.load("en_core_web_sm")

# Perform Named Entity Recognition

def named_entity_recognition(text):

  # Tokenize the text

  tokens = tokenize_text(text)

  # Extract named entities

  entities = []

  for token if token.ent_type_ == "PERSON":

      entities.append((token.text, "PERSON"))

    elif token.ent_type_ == "ORGANIZATION":

      entities.append((token.text, "ORGANIZATION"))

    elif token.ent_type_ == "LOCATION":

      entities.append((token.text, "LOCATION"))

    elif token.ent_type_ == "DATE":

      entities.append((token.text, "DATE"))

    elif token.ent_type_ == "TIME":

      entities.append((token.text, "TIME"))

  return entities

# Print the results

text = "The president of the United States visited the White House today."

entities = named_entity_recognition(text)

for entity in entities:

  print(entity)in tokens:
    # Print the results

text = "The president of the United States visited the White House today."

entities = named_entity_recognition(text)

for entity in entities:

  print(entity)

# Print the number of named entities

print("The number of named entities is:", len(entities))

# Print the types of named entities

types = set([entity[1] for entity in entities])

print("The types of named entities are:", types)
# Add a feature to save the results of Named Entity Recognition to a file

def save_results(results, filename):

  with open(filename, "w") as f:

    for entity in results:

      f.write(entity + "\n")

# Add a feature to visualize the results of Named Entity Recognition

def visualize_results(results):

  import matplotlib.pyplot as plt

  # Create a figure

  fig = plt.figure()

  # Create a subplot for each type of named entity

  for type in set([entity[1] for entity in results]):

    ax = fig.add_subplot(1, len(set([entity[1] for entity in results])), len(set([entity[1] for entity in results])) - len(set([entity[1] for entity in results])) + 1 + int(type))

    ax.bar([entity[0] for entity in results if entity[1] == type], [entity[2] for entity in results if entity[1] == type])

    ax.set_title(type)

  # Show the figure

  plt.show()

# Add a feature to export the results of Named Entity Recognition to a JSON file

def export_results(results, filename):

  import json

  with open(filename, "w") as f:

    json.dump(results, f, indent=4)
    # Add a feature to allow the user to select the pre-trained model to use

def select_model():

  # Print a list of available models

  print("Available models:")

  for model in spacy.available_models():

    print(model)

  # Get the user's input

  model_name = input("Select a model: ")

  # Load the model

  nlp = spacy.load(model_name)

  return nlp

# Add a feature to allow the user to specify the entities to be extracted

def select_entities():

  # Print a list of available entities

  print("Available entities:")

  for entity in spacy.get_pipe("ner").labels:

    print(entity)

  # Get the user's input

  entities = input("Select entities to extract (comma-separated): ").split(",")

  return entities

# Add a feature to allow the user to specify the confidence threshold

def select_confidence_threshold():

  # Get the user's input

  confidence_threshold = float(input("Enter a confidence threshold: "))

  return confidence_threshold

# Add a feature to allow the user to specify the output format

def select_output_format():

  # Print a list of available output formats

  print("Available output formats:")

  for format in ["text", "json", "csv"]:

    print(format)
      # Get the user's input

  output_format = input("Select an output format: ")

  return output_format

# Add a main function to call the other functions

def main():

  # Select the pre-trained model

  nlp = select_model()

  # Select the entities to be extracted

  entities = select_entities()

  # Select the confidence threshold

  confidence_threshold = select_confidence_threshold()

  # Select the output format

  output_format = select_output_format()

  # Get the text to be processed

  text = input("Enter text: ")

  # Perform Named Entity Recognition

  entities = named_entity_recognition(text, nlp, entities, confidence_threshold)

  # Print the results

  if output_format == "text":

    for entity in entities:

      print(entity)

  elif output_format == "json":

    json.dump(entities, sys.stdout, indent=4)

  elif output_format == "csv":

    writer = csv.writer(sys.stdout)

    writer.writerow(["Entity", "Type", "Confidence"])

    for entity in entities:

      writer.writerow([entity[0], entity[1], entity[2]])

if __name__ == "__main__":

  main()
  
    

  # Add a feature to allow the user to save the results to a file

def save_results(results, filename):

  with open(filename, "w") as f:

    for entity in results:

      f.write(entity + "\n")

  # Add a feature to visualize the results of Named Entity Recognition

def visualize_results(results):

  import matplotlib.pyplot as plt

  # Create a figure

  fig = plt.figure()

  # Create a subplot for each type of named entity

  for type in set([entity[1] for entity in results]):

    ax = fig.add_subplot(1, len(set([entity[1] for entity in results])), len(set([entity[1] for entity in results])) - len(set([entity[1] for entity in results])) + 1 + int(type))

    ax.bar([entity[0] for entity in results if entity[1] == type], [entity[2] for entity in results if entity[1] == type])

    ax.set_title(type)

  # Show the figure

  plt.show()

  # Add a feature to export the results of Named Entity Recognition to a JSON file

def export_results(results, filename):

  import json

  with open(filename, "w") as f:

    json.dump(results, f, indent=4)

# Add a feature to allow the user to select the pre-trained model to use

def select_model():

  # Print a list of available models

  print("Available models:")

  for model in spacy.available_models():

    print(model)

  # Get the user's input

  model_name = input("Select a model: ")

  # Load the model

  nlp = spacy.load(model_name)

  return nlp
# Add a feature to allow the user to specify the entities to be extracted

def select_entities():

  # Print a list of available entities

  print("Available entities:")

  for entity in spacy.get_pipe("ner").labels:

    print(entity)

  # Get the user's input

  entities = input("Select entities to extract (comma-separated): ").split(",")

  return entities

# Add a feature to allow the user to specify the confidence threshold

def select_confidence_threshold():

  # Get the user's input

  confidence_threshold = float(input("Enter a confidence threshold: "))

  return confidence_threshold

# Add a feature to allow the user to specify the output format

def select_output_format():

  # Print a list of available output formats

  print("Available output formats:")

  for format in ["text", "json", "csv"]:

    print(format)

  # Get the user's input

  output_format = input("Select an output format: ")

  return output_format

# Add a main function to call the other functions

def main():

  # Select the pre-trained model

  nlp = select_model()

  # Select the entities to be extracted

  entities = select_entities()

  # Select the confidence threshold

  confidence_threshold = select_confidence_threshold()

  # Select the output format

  output_format = select_output_format()
  # Get the text to be processed

  text = input("Enter text: ")

  # Perform Named Entity Recognition

  entities = named_entity_recognition(text, nlp, entities, confidence_threshold)

  # Print the results

  if output_format == "text":

    for entity in entities:

      print(entity)

  elif output_format == "json":

    json.dump(entities, sys.stdout, indent=4)

  elif output_format == "csv":

    writer = csv.writer(sys.stdout)

    writer.writerow(["Entity", "Type", "Confidence"])

    for entity in entities:

      writer.writerow([entity[0], entity[1], entity[2]])

  # Save the results to a file

  save_results(entities# Visualize the results of Named Entity Recognition

def visualize_results(results):

  import matplotlib.pyplot as plt

  # Create a figure

  fig = plt.figure()

  # Create a subplot for each type of named entity

  for type in set([entity[1] for entity in results]):

    ax = fig.add_subplot(1, len(set([entity[1] for entity in results])), len(set([entity[1] for entity in results])) - len(set([entity[1] for entity in results])) + 1 + int(type))

    ax.bar([entity[0] for entity in results if entity[1] == type], [entity[2] for entity in results if entity[1] == type])

    ax.set_title(type)

  # Show the figure

  plt.show()

# Export the results of Named Entity Recognition to a JSON file

def export_results(results, filename):

  import json

  with open(filename, "w") as f:

    json.dump(results, f, indent=4)

if __name__ == "__main__":

  main()

  # Add a feature to allow the user to save the results to a file

def save_results(results, filename):

  with open(filename, "w") as f:

    for entity in results:

      f.write(entity + "\n")

  # Add a feature to visualize the results of Named Entity Recognition

def visualize_results(results):

  import matplotlib.pyplot as plt

  # Create a figure

  fig = plt.figure()

  # Create a subplot for each type of named entity

  for type in set([entity[1] for entity in results]):

    ax = fig.add_subplot(1, len(set([entity[1] for entity in results])), len(set([entity[1] for entity in results])) - len(set([entity[1] for entity in results])) + 1 + int(type))

    ax.bar([entity[0] for entity in results if entity[1] == type], [entity[2] for entity in results if entity[1] == type])

    ax.set_title(type), "results.txt")
  # Show the figure

  plt.show()

  # Add a feature to export the results of Named Entity Recognition to a JSON file

def export_results(results, filename):

  import json

  with open(filename, "w") as f:

    json.dump(results, f, indent=4)

# Add a feature to allow the user to select the pre-trained model to use

def select_model():

  # Print a list of available models

  print("Available models:")

  for model in spacy.available_models():

    print(model)

  # Get the user's input

  model_name = input("Select a model: ")

  # Load the model

  nlp = spacy.load(model_name)

  return nlp

# Add a feature to allow the user to specify the entities to be extracted

def select_entities():

  # Print a list of available entities

  print("Available entities:")

  for entity in spacy.get_pipe("ner").labels:

    print(entity)

  # Get the user's input

  entities = input("Select entities to extract (comma-separated): ").split(",")

  return entities

# Add a feature to allow the user to specify the confidence threshold

def select_confidence_threshold():

  # Get the user's input

  confidence_threshold = float(input("Enter a confidence threshold: "))

  return confidence_threshold

# Add a feature to allow the user to specify the output format

def select_output_format():
  # Add a main function to call the other functions

def main():

  # Select the pre-trained model

  nlp = select_model()

  # Select the entities to be extracted

  entities = select_entities()

  # Select the confidence threshold

  confidence_threshold = select_confidence_threshold()

  # Select the output format

  output_format = select_output_format()

  # Get the text to be processed

  text = input("Enter text: ")

  # Perform Named Entity Recognition

  entities = named_entity_recognition(text, nlp, entities, confidence_threshold)

  # Print the results

  if output_format == "text":

    for entity in entities:

      print(entity)

  elif output_format == "json":

    json.dump(entities, sys.stdout, indent=4)

  elif output_format == "csv":

    writer = csv.writer(sys.stdout)

    writer.writerow(["Entity", "Type", "Confidence"])

    for entity in entities:

      writer.writerow([entity[0], entity[1], entity[2]])

  # Save the results to a file

  save_results(entities, "results.txt")

  # Visualize the results

  visualize_results(entities)
  
  
  
