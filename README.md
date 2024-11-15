# spline dist

Задача - найти ближайшие точки на двух сплайнах.

В качестве сплайна была взята кривая Безье

Поиск ближайших точек эквивалентен поиску минимального расстояния между двумя параметризованными точками на двух кривых $(x_1(t_1), y_1(t_1))$ и $(x_2(t_2), y_2(t_2))$. Каждая кривая параметризована своим параметром. Тогда нужно минимизировать по  $(t_1, t_2)$ такую функцию:

$$
f(t_1, t_2) = (x_1(t_1)-x_2(t_2))^2 + (y_1(t_1)-y_2(t_2))^2
$$

Частные производные которой раны:

$$
\frac{\partial f}{\partial t_1} = 2(x_1-x_2) \frac{\partial x_1}{\partial t_1} + 2(y_1-y_2) \frac{\partial y_1}{\partial t_1}
$$
и
$$
\frac{\partial f}{\partial t_2} = -2(x_1-x_2) \frac{\partial x_2}{\partial t_2} - 2(y_1-y_2) \frac{\partial y_2}{\partial t_2}
$$

Проблема в том, что функция $f$ может иметь много локальных минимумов. Поэтому для поиска глобального минимума выбран [алгоритм имитации отжига](https://ru.wikipedia.org/wiki/%D0%90%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC_%D0%B8%D0%BC%D0%B8%D1%82%D0%B0%D1%86%D0%B8%D0%B8_%D0%BE%D1%82%D0%B6%D0%B8%D0%B3%D0%B0). Но его недостаток в том, что он в локальной близости от минимума долго сходится, от чего точность понижается. Поэтому в конце применяется алгоритм градиентного спуска для увеличения точности. Точка, найденная алгоритмом отжига становится начальной для алгоритма градиентного спуска.

В данном решении не находятся все точки минимума, только первый, но если запустить алгоритм несколько раз подряд, то он может найти разные минимумы, зависит от параметров. Решение подходит для сложных случаев, где полный перебор не уместен и время работы не зависит от количества опорных точек, только от параметров.

## Как пользоваться 

```bash
splines.exe input.txt 2.txt \
    --annealing_iters 100000 \
    --annealing_step 1. \
    --annealing_temperature 1000000 \
    --sgd_learning_rate 1e-1 \
    --sgd_max_iter 10000 \
    --sgd_tolerance 1e-12
```

* `annealing_iters` сколько итераций отжига сделать, никогда не прерывается
* `annealing_step` шаг отжига, на каком стандартном отклонении от текущей точки брать новую
* `temperature` контролирует насколько долго искать глобальный минимум, чем больше, тем больше вероятность поиска в другой случайной точке. Со временем температура снижается и алгоритм превращается в локальный поиск минимума. 
* `sgd_learning_rate` скорость градиентного спуска
* `sgd_max_iter` - сколько итераций градиентного спуска сделать, может прерваться ели необходимая точность достигнута
* `sgd_tolerance` - преждевременное условия остановки спуска, контролирует чтобы градиент не был слишком мал и был смысл дальше делать спуск