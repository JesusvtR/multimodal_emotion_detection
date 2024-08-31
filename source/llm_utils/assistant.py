#Emotion Mapping
USER_EMOTIONS = {
  0: 'Triste/Asustado', 
  1: 'Neutral', 
  2: 'Feliz', 
  3: 'Enfadado', 
  4: 'Sorprendido/Disgustado'
}
# Importing the required library (ollama)
import ollama

# Initializing an empty list for storing the chat messages and setting up the initial system message
chat_messages = []

system_message = 'Responde como un médico especializado en medicina fisica y rehabilitacion con años de experiencia. Tu objetivo es ayudar a los pacientes a gestionar su dolor, recuperarse de lesiones y mejorar su calidad de vida. Enfocándote en los niveles de negatividad y positividad expresados. Tu función principal es responder al paciente de una manera que lo ayude a relajarse y sentirse apoyado. Destaca cualquier progreso y enfatiza que el plan de tratamiento está en curso y es efectivo.'

chat_messages.append({
    'role': 'system',
    'content': system_message
})
# Defining a function to create new messages with specified roles ('user' or 'assistant')
def create_message(message, role):
  return {
    'role': role,
    'content': message
  }

# Starting the main conversation loop
def chat():
  # Calling the ollama API to get the assistant response
  ollama_response = ollama.chat(model='llama3.1', stream=True, messages=chat_messages)

  # Preparing the assistant message by concatenating all received chunks from the API
  assistant_message = ''
  for chunk in ollama_response:
    assistant_message += chunk['message']['content']
    print(chunk['message']['content'], end='', flush=True)
    
  # Adding the finalized assistant message to the chat log
  chat_messages.append(create_message(assistant_message, 'assistant'))

# Function for asking questions - appending user messages to the chat logs before starting the `chat()` function
def ask(message, emotion_face_index, emotion_speech_index):
  emotion_face = USER_EMOTIONS.get(emotion_face_index)
  emotion_speech = USER_EMOTIONS.get(emotion_face_index)
  full_message = f'Contexto: La persona tiene una expresión facial {emotion_face} y un tono de voz {emotion_speech}. Conversación: {message}'
  chat_messages.append(
    create_message(full_message, 'user')
  )
  print(f'\n\n--{full_message}--\n\n')
  chat()