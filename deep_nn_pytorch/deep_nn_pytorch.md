# Deep Neural Network com Pytorch

- descrição do curso

# Sumário
## Semana 1
### 1.1 [Tensores 1D](#tensores-1d)
### 1.2 [Tensores Bidimensionais](#tensores-bidimensionais)
### 1.3 [Derivativos no Pytorch](#derivativos-no-pytorch)
### 1.4 [Conjunto de Dados Simples](#conjunto-de-dados-simples)
### 1.5 [Conjunto de Dados](#conjunto-de-dados)

## Semana 2
### 2.1 [Regressão Linear em 1D](#regressão-linear-em-1d)
### 2.2 [Treinamento de Regressão Linear](#treinamento-de-regressão-linear)
### 2.3 [Gradiente Descendente e Custo](#gradiente-descendente-e-custo)
### 2.4 [Pytorch Slope](#pytorch-slope)
### 2.5 [Treinamento de Regressão Linear (repetição)](#treinamento-de-regressão-linear-repetição)
### 2.6 [Gradiente Descendente Estocástico e o Carregador de Dados](#gradiente-descendente-estocástico-e-o-carregador-de-dados)
### 2.7 [Mini-Batch Gradient Descent](#mini-batch-gradient-descent)
### 2.8 [Otimização no Pytorch](#otimização-no-pytorch)
### 2.9 [Treinamento, Validação e Divisão de Dados](#treinamento-validação-e-divisão-de-dados)

## Semana 3
### 3.1 [Regressão Linear Múltipla](#regressão-linear-múltipla)
### 3.2 [Regressão Linear de Múltiplas Saídas](#regressão-linear-de-múltiplas-saídas)
### 3.3 [Regressão Logística para Classificação](#regressão-logística-para-classificação)

## Semana 4
### 4.1 [Previsão Softmax](#previsão-softmax)
### 4.2 [Função Softmax](#função-softmax)
### 4.3 [Softmax Pytorch](#softmax-pytorch)
### 4.4 [Redes Neurais Rasas](#redes-neurais-rasas)

## Semana 5
### 5.1 [Redes Neurais Profundas](#redes-neurais-profundas)
### 5.2 [Desistência](#desistência)
### 5.3 [Pesos e Inicialização da Rede Neural](#pesos-e-inicialização-da-rede-neural)
### 5.4 [Gradient Descent com Momentum](#gradient-descent-com-momentum)
### 5.5 [Normalização em Lote](#normalização-em-lote)

## Semana 6
### 6.1 [Convolução](#convolução)
### 6.2 [Funções de Ativação e Max Polling](#funções-de-ativação-e-max-polling)
### 6.3 [Vários Canais de Entrada e Saída](#vários-canais-de-entrada-e-saída)
### 6.4 [Rede Neural Convolucional](#rede-neural-convolucional)
### 6.5 [Modelos de Visão de Lanterna](#modelos-de-visão-de-lanterna)

## Semana 1
### Tensores 1D
Em PyTorch, tensores são estruturas fundamentais que representam dados multi-dimensionais, semelhantes a arrays ou matrizes em outras linguagens. Eles são a base para a construção e manipulação de modelos de aprendizado de máquina e redes neurais profundas. Aqui estão alguns pontos importantes sobre tensores em PyTorch:

1. **Similaridade com NumPy:** PyTorch tensores são semelhantes aos arrays do NumPy e, muitas vezes, podem ser convertidos de um para o outro. Isso facilita a integração com bibliotecas científicas em Python.

2. **Suporte a GPU:** PyTorch permite a computação em GPUs para acelerar operações. Os tensores podem ser movidos para uma GPU para aproveitar o poder de processamento paralelo.

3. **Operações Matemáticas:** PyTorch fornece uma ampla variedade de operações matemáticas que podem ser aplicadas a tensores. Isso inclui operações aritméticas, funções trigonométricas, álgebra linear, entre outras.

4. **Autograd:** Uma característica fundamental é o sistema de autograd, que automaticamente calcula gradientes para tensores. Isso é crucial para a otimização de modelos de aprendizado de máquina.

5. **Criação de Tensores:** Você pode criar tensores de várias maneiras, seja inicializando-os com valores específicos, gerando números aleatórios, ou a partir de dados existentes.

Exemplo de criação de um tensor em PyTorch:

```python
import torch

# Criando um tensor com valores específicos
tensor_exemplo = torch.tensor([[1, 2, 3], [4, 5, 6]])

print(tensor_exemplo)
```
Tensores 1D em PyTorch são estruturas de dados unidimensionais que podem armazenar uma sequência de elementos. Esses tensores são semelhantes a vetores ou listas unidimensionais em outras linguagens de programação. Eles são úteis para representar dados ao longo de uma única dimensão, como séries temporais, uma linha de pixels de uma imagem ou um conjunto de valores.

Aqui está um exemplo de como criar e trabalhar com um tensor 1D em PyTorch:

```python
import torch

# Criando um tensor 1D
tensor_1d = torch.tensor([1, 2, 3, 4, 5])

# Acessando elementos do tensor
print(tensor_1d[0])  # Saída: 1
print(tensor_1d[2])  # Saída: 3

# Operações matemáticas em tensores 1D
tensor_resultado = tensor_1d * 2
print(tensor_resultado)  # Saída: tensor([2, 4, 6, 8, 10])
```

Neste exemplo, `tensor_1d` é um tensor 1D contendo os valores de 1 a 5. Você pode acessar elementos individualmente e realizar operações matemáticas diretamente nos tensores.

Os tensores 1D são frequentemente usados em problemas onde os dados estão organizados de forma linear, e são a base para a construção de estruturas de dados mais complexas, como matrizes bidimensionais (tensores 2D) e tensores de ordens superiores.
Este é um conceito básico, e há muito mais para explorar ao trabalhar com tensores em PyTorch, especialmente ao construir e treinar redes neurais profundas. Se tiver mais perguntas ou se quiser abordar aspectos específicos, sinta-se à vontade para perguntar!
