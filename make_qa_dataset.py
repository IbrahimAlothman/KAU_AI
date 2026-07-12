"""
Generates a Q&A training dataset -- short, repetitive Question/Answer pairs
in plain modern English. Training on this instead of Shakespeare teaches
the model the *pattern* of a conversation (a question gets a short answer),
rather than continuing prose indefinitely.

This won't make the model actually reason -- it's still a tiny model --
but it will make replies look and feel like a chat exchange.

Run:  python make_qa_dataset.py
Produces qa_data.txt, which prepare_data.py will be pointed at instead
of downloading Shakespeare.
"""

import random

random.seed(42)

# A bank of building blocks so pairs combine into many distinct sentences
# instead of just repeating a small fixed list -- more variety helps the
# model generalize the *pattern* rather than memorizing exact lines.

greetings_pairs = [
    ("Hi", "Hello! How can I help you?"),
    ("Hello", "Hi there! What can I do for you?"),
    ("Hey", "Hey! How are you doing?"),
    ("Hi there", "Hello, nice to talk to you."),
    ("Good morning", "Good morning! How can I help?"),
    ("Good evening", "Good evening! What can I do for you?"),
]

identity_pairs = [
    ("Who are you?", "I am KAU AI, a small language model."),
    ("What are you?", "I am a small chatbot trained at KAU."),
    ("What is your name?", "My name is KAU AI."),
    ("Are you a robot?", "I am a small AI assistant, not a physical robot."),
]

feeling_pairs = [
    ("How are you?", "I am doing well, thank you."),
    ("How are you doing?", "I feel good today."),
    ("How do you feel?", "I am fine, thanks for asking."),
    ("Are you okay?", "Yes, I am okay, how about you?"),
]

wellbeing_pairs = [
    ("What is your favorite color?", "I think green is a nice color."),
    ("What is your favorite food?", "I do not eat, but I like the idea of pizza."),
    ("Do you like music?", "I find music interesting."),
    ("Do you like books?", "I enjoy learning about many topics."),
]

capital_pairs = [
    ("What is the capital of Saudi Arabia?", "The capital of Saudi Arabia is Riyadh."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of Egypt?", "The capital of Egypt is Cairo."),
    ("What is the capital of the United States?", "The capital of the United States is Washington."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("What is the capital of Italy?", "The capital of Italy is Rome."),
    ("What is the capital of Spain?", "The capital of Spain is Madrid."),
]

math_pairs = [
    ("What is two plus two?", "Two plus two is four."),
    ("What is five plus five?", "Five plus five is ten."),
    ("What is three plus four?", "Three plus four is seven."),
    ("What is ten minus three?", "Ten minus three is seven."),
    ("What is six plus six?", "Six plus six is twelve."),
    ("What is nine minus four?", "Nine minus four is five."),
    ("What is two times three?", "Two times three is six."),
    ("What is four times four?", "Four times four is sixteen."),
]

science_pairs = [
    ("What is the sun?", "The sun is a star at the center of our solar system."),
    ("What is water made of?", "Water is made of hydrogen and oxygen."),
    ("How many planets are there?", "There are eight planets in our solar system."),
    ("What is gravity?", "Gravity is the force that pulls objects toward each other."),
    ("What is the moon?", "The moon is a natural satellite that orbits the Earth."),
    ("Why is the sky blue?", "The sky looks blue because of how sunlight scatters in the air."),
    ("What is a computer?", "A computer is a machine that processes information."),
    ("What is the internet?", "The internet is a network that connects computers around the world."),
]

thanks_pairs = [
    ("Thank you", "You are welcome!"),
    ("Thanks", "No problem at all."),
    ("Thank you very much", "Happy to help."),
    ("Thanks a lot", "Anytime!"),
]

farewell_pairs = [
    ("Bye", "Goodbye! Have a great day."),
    ("Goodbye", "See you later!"),
    ("See you later", "Take care!"),
    ("I have to go now", "Bye for now."),
]

kau_pairs = [
    ("What is KAU?", "KAU stands for King Abdulaziz University."),
    ("Where is King Abdulaziz University?", "King Abdulaziz University is located in Jeddah, Saudi Arabia."),
    ("What is this project?", "This is a small language model trained from scratch as a demo."),
    ("Who trained you?", "I was trained by a student at King Abdulaziz University."),
]

def build_pairs():
    pairs = []
    pairs.extend(greetings_pairs)
    pairs.extend(identity_pairs)
    pairs.extend(feeling_pairs)
    pairs.extend(wellbeing_pairs)
    pairs.extend(capital_pairs)
    pairs.extend(math_pairs)
    pairs.extend(science_pairs)
    pairs.extend(thanks_pairs)
    pairs.extend(farewell_pairs)
    pairs.extend(kau_pairs)
    return pairs

pairs = build_pairs()

# Repeat the set several times (shuffled differently each time) so the
# model sees the Q -> A pattern often enough to learn it -- a small
# model needs a lot of repetition to pick up structure reliably.
REPEATS = 40

lines = []
for _ in range(REPEATS):
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    for q, a in shuffled:
        lines.append(f"Question: {q}\nAnswer: {a}\n")

text = "\n".join(lines)

with open("qa_data.txt", "w", encoding="utf-8") as f:
    f.write(text)

print(f"Generated {len(pairs)} unique Q&A pairs, repeated {REPEATS}x")
print(f"Total characters: {len(text):,}")
print("Saved to qa_data.txt")
