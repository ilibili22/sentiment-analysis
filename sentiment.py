from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds  

# getting a dataset:
dataset, info = tfds.load(
    "imdb_reviews",
    with_info=True,
    as_supervised=True
)

train_data = dataset["train"]

sentences = []
labels = []

for text, label in tfds.as_numpy(train_data):
    sentences.append(text.decode("utf-8"))
    labels.append(label)

print(sentences[:5])
print(labels[:5])


# making an object form Tokenizer class.
tokenizer = Tokenizer()

# this code means: Tokenizer check our sentences and make a vocabulary from them.
tokenizer.fit_on_texts(sentences)

# this code means: save the vocabulary that Tokenizer make it with token id in word_index.
word_index = tokenizer.word_index

# print(word_index)

# this code means: replace each word in sentences with its token ids. like: dog ->(=) 10.
sequences = tokenizer.texts_to_sequences(sentences)

# print(sequences)

# this code means: add zero(0) at the end of the sequence for same len.
padded = pad_sequences(sequences, padding = "post")


# print(padded)

# vocabulary size: how many word do we have.
vocab_size = len(word_index) + 1

# print(vocab_size)


"""

making model:

    sequential means:
        (لایه های مدل را به ترتیب پشت سر هم قرار میده:مثلا اول امبدینگ بعد دنسه بعد دنسه.)
        مدل ورودی رو از بالا میگیره و به ترتیب از این لایه ها عبور میمنه.
    
    vocab_size means:
        our model have vocab_size(25) tokens.
    
    16 means:
        make vector with 16 number for each token.

    GlobalAveragePooling1D:
        چون امبدینگ میاد توی یک جمله برای هر توکن یا همون کلمه یک 
        وکتور میسازد . یک جمله شامل چندین وکتور میشه به تعداد توکن هاش
        پس این میاد و وکتور های یک جلمه را برمیدارد و یک میانگین از همشون میگیرد
        و این شکلی یک جمله فقط یک وکتور دارد و برای انجام مجاسبات خیلی اسون تر است.
    
    Dense:
        16 means:
            this layar have 16 neural(نرون)
        activation:
            how the output must be:
            relu:
                خروجی های منفی تبدیل به صفر و خروجی های مثبت را نگه دار. 
                چون ما فقط صفر و یک میخوایم.
            sigmoid:
                هر عددی را به عددی بین صفر و یک تبدیل میکند
""" 
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16,activation = "relu"),
    tf.keras.layers.Dense(1,activation = "sigmoid"),
])


"""
مدل چطور اموزش ببینه:

compile:
    مشخص کن مدل با چه روشی آموزش ببیند و اشتباهش چگونه سنجیده شود.

binary_crossentropy:
    میزان اشتباه رو حساب کنه

adam:
    بروزرسانی وزن های مدل در زمان اموزش 
    بعد از این که مدل فهمید اشتباه کرده این میاد و وزن رو بروز میکنه.

accuracy:
    این فقط برای اینه که هنگام آموزش ببینیم مدل چند درصد پیش‌بینی‌هایش درست بوده.

"""
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

padded = np.array(padded)
labels = np.array(labels)

"""
اموزش مدل:

fit:
    داده ها را به مدل بده و اموزش رو شورع کن


"""
model.fit(padded, labels, epochs = 20)


# تست مدل

user_text = input("Enter a sentences: ")


new_sequence = tokenizer.texts_to_sequences([user_text])

new_padded = pad_sequences(new_sequence, padding="post")

# predict: پیش بینی کن
prediction = model.predict(new_padded)

print(prediction)