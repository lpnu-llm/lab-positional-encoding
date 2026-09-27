# %% [markdown]
# # Позиційне кодування
#
# Після токенізації ми маємо список id. Embedding-таблиця перетворює кожен id
# на вектор, але однаковий токен у різних місцях отримує однаковий вектор.
# Трансформеру також треба знати, *де* стоїть токен.
#
# У цій лабораторній реалізуємо позиційне кодування з нуля, використовуючи
# лише стандартну бібліотеку Python. Потрібні знання з попередніх робіт:
# вектори, скалярний добуток, embedding-таблиця та attention.
#
# Нас цікавлять два способи передати позицію:
#
# 1. Додати позиційний вектор до embedding токена.
# 2. Повернути query та key у self-attention залежно від позицій (RoPE).
#
# У всіх вправах позиції починаються з нуля.

# %% [markdown]
# ## 1. Чому самого списку токенів недостатньо
#
# Розгляньте «кіт бачить пса» і «пес бачить кота». Порядок змінює зміст,
# хоча набір токенів подібний. Self-attention обчислює схожість векторів
# через скалярні добутки. Без позиційного кодування та маски перестановка
# вхідних векторів просто переставить результати attention. Causal mask
# обмежує доступ до майбутніх токенів, але не задає окремого вектора позиції.
#
# Токенізатор *зберігає* порядок у списку id; позиційне кодування робить цей
# порядок доступним обчисленням моделі.
#
# Найпростіший спосіб — додати до embedding кожного токена вектор його позиції:
#
# $$x_p = e_{\mathrm{token}_p} + P_p.$$
#
# Обидва вектори мають довжину $d_{\mathrm{model}}$. Додавання зберігає цю
# довжину, отже наступним шарам не треба змінювати форму входу.

# %%

def add_position(token_embeddings, position_embeddings):
    """Додає два списки векторів однакової форми, не змінюючи входи."""

    # Перевірте кількість рядків і довжину кожної пари рядків.
    # Якщо форми не збігаються, підніміть ValueError.
    ...


def test_add_position():
    tokens = [[1.0, 2.0], [1.0, 2.0]]
    positions = [[0.0, 0.0], [0.5, -0.5]]
    result = add_position(tokens, positions)
    assert result == [[1.0, 2.0], [1.5, 1.5]]
    assert result[0] != result[1]
    assert tokens == [[1.0, 2.0], [1.0, 2.0]]
    assert positions == [[0.0, 0.0], [0.5, -0.5]]
    assert add_position([], []) == []

    for left, right in [
        ([[1.0]], []),
        ([[1.0, 2.0]], [[1.0]]),
    ]:
        try:
            add_position(left, right)
        except ValueError:
            pass
        else:
            raise AssertionError("Форми embedding-ів мають збігатися")


if __name__ == "__main__":
    test_add_position()
    print("✓ add_position")

# %% [markdown]
# Питання:
#
# - Що станеться з двома однаковими токенами на різних позиціях після додавання?
# - Скільки чисел має матриця позицій для послідовності з $L$ токенів?

# %% [markdown]
# ## 2. Фіксоване синусоїдальне кодування
#
# У [початковому Transformer](https://arxiv.org/abs/1706.03762) вектор
# позиції обчислювали без навчуваних параметрів. Для парного
# $d_{\mathrm{model}}$ і $i = 0, \ldots,
# d_{\mathrm{model}}/2-1$:
#
# $$\theta_i = 10000^{-2i/d_{\mathrm{model}}},$$
# $$P_{p,2i} = \sin(p\theta_i), \qquad
#   P_{p,2i+1} = \cos(p\theta_i).$$
#
# Перша пара координат змінюється швидко, наступні — повільніше. Позиція 0
# має вигляд [0, 1, 0, 1, ...]. Кодування можна обчислити і для позиції,
# якої не було під час навчання, але це саме по собі не гарантує якості
# моделі на довших послідовностях.
#
# Параметр `start` дозволяє почати з іншої абсолютної позиції. Він стане
# корисним, коли модель генерує продовження вже обробленого тексту.

# %%

def sinusoidal_encoding(length, d_model, start=0):
    """Матриця форми (length, d_model) для позицій start ... start+length-1."""

    # length і start мають бути невід'ємними, d_model — додатним парним.
    # Для обчислення використайте math.sin і math.cos.
    ...


def test_sinusoidal_encoding():
    from math import cos, isclose, sin

    pe = sinusoidal_encoding(4, 6)
    assert len(pe) == 4
    assert all(len(row) == 6 for row in pe)
    assert pe[0] == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    assert isclose(pe[1][0], sin(1.0), abs_tol=1e-12)
    assert isclose(pe[1][1], cos(1.0), abs_tol=1e-12)
    assert isclose(pe[1][2], sin(1 / 10000 ** (2 / 6)), abs_tol=1e-12)
    assert pe[1] != pe[2]
    assert sinusoidal_encoding(0, 2) == []
    assert sinusoidal_encoding(2, 4, start=3) == sinusoidal_encoding(5, 4)[3:]

    for length, d_model, start in [(1, 3, 0), (1, 0, 0), (-1, 4, 0),
                                   (1, 4, -1)]:
        try:
            sinusoidal_encoding(length, d_model, start)
        except ValueError:
            pass
        else:
            raise AssertionError("Некоректні розміри або позиція мають давати ValueError")


if __name__ == "__main__":
    test_sinusoidal_encoding()
    print("✓ sinusoidal_encoding")

# %% [markdown]
# Питання:
#
# - Чому значення у другій парі координат змінюються повільніше, ніж у першій?
# - Скільки навчуваних параметрів додає ця схема?
# - Навіщо під час генерації враховувати `start`, а не знову починати з нуля?

# %% [markdown]
# ## 3. Навчувана таблиця позицій
#
# Інший спосіб — мати таблицю форми (max_length, d_model). Кожен рядок є
# параметром моделі, який можна оновлювати градієнтним спуском, подібно до
# embedding-таблиці токенів. У цій вправі ми не навчаємо модель: таблиця
# вже задана, потрібно вибрати її рядки.
#
# Навчувана таблиця має max_length × d_model додаткових параметрів.
# Для позиції max_length рядка вже немає. Це інша межа, ніж у формули
# синусоїдального кодування.

# %%

def learned_position_encoding(length, table, start=0):
    """Копії рядків table для позицій start ... start+length-1."""

    # Перевірте невід'ємні length/start, межі таблиці й однакову довжину
    # її рядків. Порожня таблиця дозволена тільки для порожнього результату.
    ...


def test_learned_position_encoding():
    table = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    result = learned_position_encoding(2, table, start=1)
    assert result == [[0.3, 0.4], [0.5, 0.6]]
    result[0][0] = 999
    assert table[1][0] == 0.3
    assert learned_position_encoding(0, []) == []
    assert learned_position_encoding(0, table, start=3) == []

    for length, source, start in [
        (4, table, 0),
        (1, table, 3),
        (1, [[1.0], [2.0, 3.0]], 0),
        (-1, table, 0),
        (1, table, -1),
    ]:
        try:
            learned_position_encoding(length, source, start)
        except ValueError:
            pass
        else:
            raise AssertionError("Некоректна таблиця або позиція мають давати ValueError")


if __name__ == "__main__":
    test_learned_position_encoding()
    print("✓ learned_position_encoding")

# %% [markdown]
# Питання:
#
# - Які саме числа в цій таблиці змінювалися б під час навчання?
# - Чи можна застосувати таблицю з max_length=128 до позиції 128?

# %% [markdown]
# ## 4. Навчувані Fourier features
#
# Синусоїдальна формула використовує наперед задані частоти. Можна залишити
# пари sin/cos, але зробити частоти параметрами моделі:
#
# $$F(p) = [\sin(p f_0),\cos(p f_0),\ldots,
#           \sin(p f_{m-1}),\cos(p f_{m-1})].$$
#
# Такий вектор можна додати до embedding токена. Нижче частоти вже задані;
# функція лише обчислює кодування. Щоб вони справді стали *навчуваними*,
# їх треба передати оптимізатору під час тренування моделі.
# Розмір вектора дорівнює подвоєній кількості частот.

# %%

def fourier_position_encoding(length, frequencies, start=0):
    """Матриця Fourier-векторів для позицій start ... start+length-1."""

    # Для кожної частоти додайте спочатку sin, потім cos.
    # length і start мають бути невід'ємними.
    ...


def test_fourier_position_encoding():
    from math import cos, isclose, sin

    frequencies = [1.0, 0.1]
    encoded = fourier_position_encoding(3, frequencies)
    assert len(encoded) == 3
    assert all(len(row) == 4 for row in encoded)
    assert encoded[0] == [0.0, 1.0, 0.0, 1.0]
    assert isclose(encoded[1][0], sin(1.0), abs_tol=1e-12)
    assert isclose(encoded[1][1], cos(1.0), abs_tol=1e-12)
    assert isclose(encoded[1][2], sin(0.1), abs_tol=1e-12)
    assert isclose(encoded[1][3], cos(0.1), abs_tol=1e-12)
    assert fourier_position_encoding(2, frequencies, start=3) == (
        fourier_position_encoding(5, frequencies)[3:]
    )
    assert fourier_position_encoding(2, []) == [[], []]
    assert frequencies == [1.0, 0.1]

    for length, start in [(-1, 0), (1, -1)]:
        try:
            fourier_position_encoding(length, frequencies, start)
        except ValueError:
            pass
        else:
            raise AssertionError("length і start мають бути невід'ємними")


if __name__ == "__main__":
    test_fourier_position_encoding()
    print("✓ fourier_position_encoding")

# %% [markdown]
# Питання:
#
# - Які параметри цієї схеми навчав би gradient descent?
# - Що зміниться, якщо всі частоти дуже близькі до нуля?
# - Чим це відрізняється від навчуваної таблиці позицій?

# %% [markdown]
# ## 5. RoPE: обертання query і key
#
# Для позиційного кодування не обов'язково додавати щось до embedding токена.
# [Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864)
# обертає пари координат у векторах
# *query* та *key* перед обчисленням їхнього скалярного добутку. Value не
# обертається. У лабораторній про attention ви вже створювали query і key
# лінійними проєкціями та ділили їхній добуток на квадратний корінь розмірності.
#
# Для пари (x, y) з індексом $i$ і позиції $p$ використаємо той самий кут,
# що й у синусоїдальному кодуванні:
#
# $$\phi_{p,i} = p \cdot 10000^{-2i/d},$$
#
# $$R_{\phi}(x,y) =
#   (x\cos\phi-y\sin\phi,\;x\sin\phi+y\cos\phi).$$
#
# Довжина вектора після обертання зберігається. Це обертання, а не додавання.
# У цій лабораторній координати утворюють сусідні пари (0,1), (2,3), ...

# %%

def rotate_pairs(vector, position):
    """Повертає копію пар координат vector, обернених за правилами RoPE."""

    # Вимагайте додатну парну довжину вектора й невід'ємну позицію.
    # Кут для пари i: position / 10000 ** (2*i / len(vector)).
    ...


def test_rotate_pairs():
    from math import cos, isclose, sin

    vector = [1.0, 0.0, 0.0, 1.0]
    assert rotate_pairs(vector, 0) == vector
    rotated = rotate_pairs(vector, 1)
    assert isclose(rotated[0], cos(1), abs_tol=1e-12)
    assert isclose(rotated[1], sin(1), abs_tol=1e-12)
    assert isclose(rotated[2], -sin(0.01), abs_tol=1e-12)
    assert isclose(rotated[3], cos(0.01), abs_tol=1e-12)
    assert vector == [1.0, 0.0, 0.0, 1.0]
    assert isclose(sum(x*x for x in rotated), sum(x*x for x in vector),
                   abs_tol=1e-12)

    for bad_vector, bad_position in [([], 0), ([1.0], 0),
                                     ([1.0, 2.0], -1)]:
        try:
            rotate_pairs(bad_vector, bad_position)
        except ValueError:
            pass
        else:
            raise AssertionError("RoPE потребує парну розмірність і позицію >= 0")


if __name__ == "__main__":
    test_rotate_pairs()
    print("✓ rotate_pairs")

# %% [markdown]
# ## 6. Як RoPE впливає на attention
#
# У попередній роботі скалярний добуток порівнював два вектори. Для однієї
# query на позиції $p$ та одного key на позиції $s$ attention обчислює score:
#
# $$\operatorname{score}(q_p,k_s)
#   = \frac{R_p(q)\cdot R_s(k)}{\sqrt{d}}.$$
#
# Ми повертаємо *обидва* вектори, а тоді рахуємо скалярний добуток.
# Спільний зсув позицій не змінює score: наприклад, пари (1,3) і (5,7)
# мають однакову відстань. Це ключова відносна властивість RoPE.
#
# Тут обчислюємо лише score однієї пари. У повному self-attention далі
# обчислюють такі значення для багатьох пар, застосовують маску й softmax,
# а потім зважують value-вектори.

# %%

def rope_score(query, key, query_position, key_position):
    """Повертає масштабований скалярний добуток RoPE query та key."""

    # Перевірте однакову додатну парну довжину векторів.
    # Використайте rotate_pairs та поділіть результат на sqrt(d).
    ...


def test_rope_score():
    from math import cos, isclose, sqrt

    q = [1.0, 0.0]
    k = [1.0, 0.0]
    assert isclose(rope_score(q, k, 0, 0), 1 / sqrt(2), abs_tol=1e-12)
    assert isclose(rope_score(q, k, 1, 3), cos(2) / sqrt(2), abs_tol=1e-12)
    assert isclose(rope_score(q, k, 1, 3), rope_score(q, k, 5, 7),
                   abs_tol=1e-12)
    assert not isclose(rope_score(q, k, 1, 3), rope_score(q, k, 1, 4),
                       abs_tol=1e-12)
    assert q == [1.0, 0.0] and k == [1.0, 0.0]

    for left, right in [([1.0, 0.0], [1.0, 0.0, 0.0, 1.0]),
                        ([], []), ([1.0], [1.0])]:
        try:
            rope_score(left, right, 0, 0)
        except ValueError:
            pass
        else:
            raise AssertionError("query і key мають мати однакову парну розмірність")


if __name__ == "__main__":
    test_rope_score()
    print("✓ rope_score")

# %% [markdown]
# ### Маленький експеримент
#
# Цей код уже готовий. Після реалізації вправ запустіть його й порівняйте
# оцінки для однакового змісту query/key на різних відстанях. Значення score
# може коливатися: RoPE не задає правило «далі завжди менш важливо».

# %%

if __name__ == "__main__":
    query = [1.0, 0.0, 1.0, 0.0]
    key = [1.0, 0.0, 1.0, 0.0]
    print("RoPE score для однакових query та key:")
    for key_position in range(6):
        print(f"  q@0, k@{key_position}: {rope_score(query, key, 0, key_position): .4f}")

# %% [markdown]
# ## 7. Порівняння
#
# | Спосіб | Де додається позиція | Додаткові параметри | За межами train |
# |---|---|---:|---|
# | Синусоїдальний | до embedding | 0 | формулу можна обчислити |
# | Навчувана таблиця | до embedding | max_length × d_model | рядка немає |
# | Fourier features | до embedding | d_model / 2 частот | формулу можна обчислити |
# | RoPE | обертання query та key | 0 у базовій формі | формулу можна обчислити |
#
# Для синусоїдального кодування, Fourier features і RoPE можливість
# *обчислити* нову позицію не означає, що модель добре працюватиме на
# невідомій довжині контексту.
#
# Фінальні питання:
#
# - Чим додавання позиції до embedding відрізняється від обертання query/key?
# - Чому в RoPE потрібно застосувати обертання і до query, і до key?
# - Чому в експерименті позиції (1,3) і (5,7) дали однаковий score?
# - Що треба змінити, щоб обробити токен на позиції 200 після вже оброблених
#   токенів 0–199: start, індекс таблиці чи позиції RoPE?
# - Які обмеження має кожен спосіб для довгого контексту?
