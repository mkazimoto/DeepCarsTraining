# DeepCarsTraining

Simulador de treinamento evolutivo de carros autônomos usando redes neurais artificiais — inspirado no projeto [DeepCars](https://github.com/JVictorDias/DeepCars/).

* a cada geração de 1000 indivíduos, os 10 melhores serão selecionados e seus descendentes sofrerão uma mutação genética para a próxima geração.
* a mutação pode boa ou ruim:
  * pode fazer o indivíduo correr devagar demais, deixando-o para trás.
  * pode fazer o indivíduo correr rápido demais, fazendo-o sofrer um acidente.
  * ou pode ser uma vantagem evolutiva, fazendo o indivíduo acelerar nas curvas, destacando-o dos demais.

<img width="842" height="744" alt="DeepCarsTraining_y74G7ZUSOA" src="https://github.com/user-attachments/assets/759b06f7-ef08-4fd4-a37e-8cabf27ea19e" />

## Visão Geral

O projeto treina uma população de carros simulados para navegar em uma pista em formato de estádio com chicanes. Cada carro é controlado por uma rede neural (MLP), e a população evolui ao longo de gerações usando um algoritmo genético simples.

```
Fase 1 — Inicialização
  Cria N redes neurais com pesos aleatórios

Fase 2 — Avaliação  (por geração)
  Simula todos os carros e calcula o fitness de cada rede

Fase 3 — Evolução   (por geração)
  Seleciona a elite, aplica mutações e forma nova geração
```

## Arquitetura da Rede Neural

MLP de 3 camadas com ativação ReLU:

| Camada  | Neurônios | Descrição                                     |
|---------|-----------|-----------------------------------------------|
| Entrada | 18 + 1    | 18 sensores LIDAR + 1 neurônio de viés        |
| Oculta  | 6 + 1     | 6 neurônios ocultos + 1 neurônio de viés      |
| Saída   | 4         | Acelerar · Ré · Virar Esquerda · Virar Direita |

## Simulação

- **Pista:** estádio com dois semicírculos conectados por retas e chicanes
- **Carro:** física 2D simplificada (velocidade, ângulo, atrito, aceleração máxima de 12 px/tick)
- **Sensores LIDAR:** 18 raios espaçados em 180° à frente do carro (normalizados em [0, 1])
- **Fitness:** distância total percorrida antes de colidir com as paredes

## Pré-requisitos

- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)
- Windows (usa Windows Forms para visualização)

## Como Executar

```bash
git clone https://github.com/mkazimoto/DeepCarsTraining.git
cd DeepCarsTraining
dotnet run 
```

Pressione qualquer tecla no terminal para iniciar o treinamento.

## Parâmetros Configuráveis

Edite as variáveis no topo de [DeepCarsTraining/Program.cs](DeepCarsTraining/Program.cs):

| Parâmetro          | Padrão | Descrição                                         |
|--------------------|--------|---------------------------------------------------|
| `populationSize`   | 1000   | Número de indivíduos por geração                  |
| `generations`      | 3      | Número de gerações de treinamento                 |
| `stepsPerEval`     | 500    | Ticks de simulação por indivíduo                  |
| `eliteCount`       | 10     | Indivíduos que sobrevivem sem mutação             |
| `mutationRate`     | 0.10   | Probabilidade de mutar cada peso (10%)            |
| `mutationStr`      | 0.30   | Magnitude máxima da perturbação nos pesos         |
| `seed`             | null   | Seed para reprodutibilidade (null = aleatório)    |
| `showVisualization`| true   | Exibe visualização gráfica em tempo real          |

## Visualização

Quando `showVisualization = true`, uma janela Windows Forms exibe a pista e os carros em tempo real. Ao fim do treinamento, a melhor rede é reproduzida automaticamente.

## Saída

Ao final de cada geração, a melhor rede é salva em um arquivo JSON no diretório de saída:

```
best_network_gen<N>_fit<fitness>.json
```

O arquivo pode ser carregado via `Trainer.ReplaySaved(path, visualizer)` para reproduzir o comportamento da melhor rede.

## Estrutura do Projeto

```
DeepCarsTraining/
├── Program.cs           # Ponto de entrada e parâmetros configuráveis
├── Trainer.cs           # Orquestrador do algoritmo evolutivo (Fases 1–3)
├── NeuralNetwork.cs     # MLP com forward pass e serialização JSON
├── Car.cs               # Simulação do carro (física + sensores LIDAR)
├── Track.cs             # Geometria da pista e raycasting
└── TrackVisualizer.cs   # Visualização em Windows Forms
```
