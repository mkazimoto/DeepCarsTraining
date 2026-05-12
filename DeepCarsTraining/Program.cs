using DeepCarsTraining;

// ── Parâmetros configuráveis ─────────────────────────────────────────────────
int  populationSize   = 1000;  // indivíduos por geração
int  generations      = 3;    // número de gerações
int  stepsPerEval     = 500;   // ticks de simulação por indivíduo (Fase 2)
int  eliteCount       = 10;    // quantos sobrevivem sem mutação (Fase 3)
double mutationRate   = 0.10;  // 10% de chance de mutar cada peso (Fase 3)
double mutationStr    = 0.30;  // magnitude máxima da perturbação (Fase 3)

// ── Curriculum learning (dificuldade progressiva das curvas) ─────────────────
// Começa com curvas suaves e aumenta gradualmente até a amplitude alvo.
// Use curriculumStart=curriculumEnd para manter a mesma amplitude o treinamento todo.
// Faixas válidas: 0 (reta) a 80 (extrema). Padrão: 0 → 40 (Reta → Média).
double curriculumStart = 0.0;   // amplitude inicial da chicane (px)
double curriculumEnd   = 40.0;  // amplitude final da chicane (px)

// Seed opcional para reprodutibilidade — comente para resultados aleatórios
int? seed = null; // ex: int? seed = 42;

// Visualização gráfica em tempo real — desative para treinamento máximo
bool showVisualization = true;

// ── Execução ─────────────────────────────────────────────────────────────────
TrackVisualizer? visualizer = showVisualization
    ? TrackVisualizer.StartOnNewThread()
    : null;

var trainer = new Trainer(
    populationSize   : populationSize,
    stepsPerEval     : stepsPerEval,
    eliteCount       : eliteCount,
    mutationRate     : mutationRate,
    mutationStrength : mutationStr,
    seed             : seed,
    visualizer       : visualizer,
    curriculumStart  : curriculumStart,
    curriculumEnd    : curriculumEnd
);

trainer.Run(generations: generations);

// ── Replay da melhor rede salva ───────────────────────────────────────────────
if (trainer.SavedNetworkPath is not null && visualizer is not null && !visualizer.IsDisposed)
{
    visualizer.SignalTrainingComplete(); // remove overlay após 1 frame — será substituído pelo replay
    Trainer.ReplaySaved(trainer.SavedNetworkPath, visualizer);
}
else
{
    // Exibe overlay de conclusão e aguarda o usuário fechar a janela
    visualizer?.SignalTrainingComplete();
    visualizer?.WaitForClose();
}
