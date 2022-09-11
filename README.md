# text_generator
Tinkoff ML

Лебкова Марина Дмитриевна

m.lebkova@gmail.com

# Описание

Модель обучается на префиксах длины 1 и 2, подсчитывая встречаемость слов на текстах из папки data

# Использование

```python3 train.py --input-dir data --model model.pkl```

```python3 generate.py --model model.pkl --prefix "я" --length 10```
