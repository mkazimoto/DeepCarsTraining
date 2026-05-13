using DeepCarsTraining;

// ── Parâmetros configuráveis ─────────────────────────────────────────────────
int  populationSize   = 1000;  // indivíduos por geração
int  generations      = 10;    // número de gerações
int  stepsPerEval     = 500;   // ticks de simulação por indivíduo (Fase 2)
int  eliteCount       = 10;    // quantos sobrevivem sem mutação (Fase 3)
double mutationRate   = 0.10;  // 10% de chance de mutar cada peso (Fase 3)
double mutationStr    = 0.30;  // magnitude máxima da perturbação (Fase 3)

// Seed opcional para reprodutibilidade — comente para resultados aleatórios
int? seed = null; // ex: int? seed = 42;

// Visualização gráfica em tempo real — desative para treinamento máximo
bool showVisualization = true;

// ── Execução ─────────────────────────────────────────────────────────────────
TrackVisualizer? visualizer = showVisualization
    ? TrackVisualizer.StartOnNewThread()
    : null;

bool restartTraining;
do
{
    visualizer?.ResetTrainingState();
    var trainer = new Trainer(
        populationSize   : populationSize,
        stepsPerEval     : stepsPerEval,
        eliteCount       : eliteCount,
        mutationRate     : mutationRate,
        mutationStrength : mutationStr,
        seed             : seed,
        visualizer       : visualizer
    );

    trainer.Run(generations: generations);

    // Volta a pista para curvas retas após o treinamento
    Track.Rebuild(0.0);

    // ── Replay da melhor rede salva (ou da rede selecionada via ComboBox) ───────
    string? replayPath = trainer.PendingReplayPath ?? trainer.SavedNetworkPath;

    if (replayPath == "")  // usuário selecionou "< Iniciar Treinamento >" durante o treinamento
    {
        restartTraining = true;
    }
    else if (replayPath is not null && visualizer is not null && !visualizer.IsDisposed)
    {
        visualizer.SignalTrainingComplete();
        restartTraining = Trainer.ReplaySaved(replayPath, visualizer);
    }
    else
    {
        restartTraining = false;
        visualizer?.SignalTrainingComplete();
        visualizer?.WaitForClose();
    }
} while (restartTraining && visualizer is not null && !visualizer.IsDisposed);
