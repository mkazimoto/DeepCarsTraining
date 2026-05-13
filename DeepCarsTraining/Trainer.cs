using System.Diagnostics;

namespace DeepCarsTraining;

/// <summary>
/// Orquestrador do treinamento evolutivo — divide o processo em 3 fases:
///
///   ┌─────────────────────────────────────────────────────────┐
///   │  FASE 1 — Inicialização                                  │
///   │  Cria a população de N redes neurais com pesos aleatórios│
///   ├─────────────────────────────────────────────────────────┤
///   │  FASE 2 — Avaliação          (loop por geração)          │
///   │  Simula todos os carros e calcula o fitness de cada rede │
///   ├─────────────────────────────────────────────────────────┤
///   │  FASE 3 — Evolução           (loop por geração)          │
///   │  Seleciona a elite, aplica mutações e forma nova geração │
///   └─────────────────────────────────────────────────────────┘
/// </summary>
public sealed class Trainer
{
    // ── Hiperparâmetros ──────────────────────────────────────────────────────
    private readonly int    _populationSize;   // número de indivíduos
    private readonly int    _stepsPerEval;     // ticks de simulação por avaliação
    private readonly int    _eliteCount;       // quantos sobrevivem sem mutação
    private readonly double _mutationRate;     // probabilidade de mutar cada peso
    private readonly double _mutationStrength; // magnitude da perturbação

    // ── Curriculum learning ──────────────────────────────────────────────────
    /// <summary>Amplitude de chicane inicial (geração 1).</summary>
    private readonly double _curriculumStart;
    /// <summary>Amplitude de chicane final (última geração).</summary>
    private readonly double _curriculumEnd;
    /// <summary>Amplitude aplicada na geração atual.</summary>
    private double _currentChicane;

    // ── Estado interno ───────────────────────────────────────────────────────
    private List<NeuralNetwork>   _population  = [];
    private int                   _generation  = 0;
    private readonly Random       _rng;
    private readonly TrackVisualizer? _visualizer;

    // ── Estatísticas ─────────────────────────────────────────────────────────
    public double BestFitnessEver     { get; private set; }
    public NeuralNetwork? BestNetwork { get; private set; }

    /// <summary>Pasta onde as redes neurais são salvas.</summary>
    public static string NetworkFolder => Path.Combine(AppContext.BaseDirectory, "RedeNeural");

    // ── Construtor ───────────────────────────────────────────────────────────
    public Trainer(
        int    populationSize   = 1000,
        int    stepsPerEval     = 500,
        int    eliteCount       = 10,
        double mutationRate     = 0.10,
        double mutationStrength = 0.30,
        int?   seed             = null,
        TrackVisualizer? visualizer = null,
        double curriculumStart  = -1.0,  // -1 = usar amplitude atual da pista
        double curriculumEnd    = -1.0)  // -1 = igual ao start (sem progressão)
    {
        _populationSize   = populationSize;
        _stepsPerEval     = stepsPerEval;
        _eliteCount       = eliteCount;
        _mutationRate     = mutationRate;
        _mutationStrength = mutationStrength;
        _rng              = seed.HasValue ? new Random(seed.Value) : new Random();
        _visualizer       = visualizer;

        // Se não especificado, usa a amplitude atual da pista
        double currentAmp   = Track.ChicaneAmplitude;
        _curriculumStart    = curriculumStart  < 0 ? currentAmp : curriculumStart;
        _curriculumEnd      = curriculumEnd    < 0 ? _curriculumStart : curriculumEnd;
        _currentChicane     = _curriculumStart;
    }

    // ════════════════════════════════════════════════════════════════════════
    // PONTO DE ENTRADA
    // ════════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Executa o ciclo completo de treinamento:
    ///   Fase 1 (uma vez) → [Fase 2 → Fase 3] × <paramref name="generations"/>
    /// </summary>
    public void Run(int generations = 50)
    {
        PrintHeader();

        var totalSw = Stopwatch.StartNew();

        // ── FASE 1 ────────────────────────────────────────────────────────────
        Phase1_Initialize();

        // ── Loop gerações: Fase 2 + Fase 3 ───────────────────────────────────
        for (int gen = 1; gen <= generations; gen++)
        {
            _generation = gen;
            var genSw = Stopwatch.StartNew();

            // Curriculum: interpola a amplitude da chicane linearmente entre gerações
            double t = generations > 1 ? (gen - 1.0) / (generations - 1.0) : 1.0;
            _currentChicane = _curriculumStart + t * (_curriculumEnd - _curriculumStart);

            Phase2_Evaluate();

            if (PendingReplayPath is not null)
            {
                genSw.Stop();
                Console.WriteLine("\n  [Treinamento interrompido] Rede selecionada via ComboBox.");
                break;
            }

            Phase3_Evolve();

            genSw.Stop();
            PrintGenerationSummary(genSw.ElapsedMilliseconds);
        }

        totalSw.Stop();
        PrintFinalSummary(totalSw.Elapsed);
        SavedNetworkPath = SaveBestNetwork();
    }

    /// <summary>Caminho do arquivo JSON salvo ao final de <see cref="Run"/>. Nulo se nenhuma rede foi salva.</summary>
    public string? SavedNetworkPath { get; private set; }

    /// <summary>Rede selecionada via ComboBox durante o treinamento. Nulo se o treinamento concluiu normalmente.</summary>
    public string? PendingReplayPath { get; private set; }

    // ════════════════════════════════════════════════════════════════════════
    // FASE 1 — INICIALIZAÇÃO
    // ════════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Cria <see cref="_populationSize"/> redes neurais com pesos aleatórios em [-1, 1].
    /// </summary>
    private void Phase1_Initialize()
    {
        Console.WriteLine();
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("═══════════════════════════════════════════════════════");
        Console.WriteLine("  FASE 1 — INICIALIZAÇÃO DA POPULAÇÃO");
        Console.WriteLine("═══════════════════════════════════════════════════════");
        Console.ResetColor();

        _population = new List<NeuralNetwork>(_populationSize);

        for (int i = 0; i < _populationSize; i++)
        {
            var nn = new NeuralNetwork();
            nn.InitializeRandom(_rng);
            _population.Add(nn);
        }

        var sample = _population[0];
        Console.WriteLine($"  Indivíduos criados : {_populationSize}");
        Console.WriteLine($"  Arquitetura        : {NeuralNetwork.InputNeurons}(+viés) → " +
                          $"{NeuralNetwork.HiddenNeurons}(+viés) → {NeuralNetwork.OutputNeurons}");
        Console.WriteLine($"  Total de pesos     : {sample.TotalWeights} por rede");
        Console.WriteLine($"  Pesos totais (pop) : {(long)sample.TotalWeights * _populationSize:N0}");
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine("  ✔  Fase 1 concluída.");
        Console.ResetColor();
    }

    // ════════════════════════════════════════════════════════════════════════
    // FASE 2 — AVALIAÇÃO
    // ════════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Cria um <see cref="Car"/> para cada rede, executa <see cref="_stepsPerEval"/> ticks
    /// de simulação e atribui a distância percorrida como fitness.
    /// </summary>
    private void Phase2_Evaluate()
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"\n  ── FASE 2 — AVALIAÇÃO  (geração {_generation}) ──");
        Console.ResetColor();

        // Posição de largada na pista
        double startX = Track.StartX;
        double startY = Track.StartY;

        // Executar a simulação de todos os carros em paralelo (Thread-safe pois
        // cada Car/NeuralNetwork é independente; _rng NÃO é usado aqui)
        var cars = new Car[_populationSize];
        for (int i = 0; i < _populationSize; i++)
            cars[i] = new Car(_population[i], startX, startY, startAngle: Track.StartAngle);

        if (_visualizer is null || _visualizer.IsDisposed)
        {
            // ── Modo rápido: avaliação totalmente paralela ────────────────────
            Parallel.For(0, _populationSize, i =>
            {
                var car = cars[i];
                for (int step = 0; step < _stepsPerEval; step++)
                    car.Step();

                car.Brain.Fitness    = car.DistanceTraveled + car.Steps * 0.1;
                car.Brain.Generation = _generation;
            });
        }
        else
        {
            // ── Modo visual: tick a tick com atualização da janela ────────────
            const int FrameInterval = 5;   // atualizar display a cada N ticks
            const int FrameMs       = 16;  // ms de pausa por atualização (~60 fps)

            for (int step = 0; step < _stepsPerEval; step++)
            {
                // Avança todos os carros (Step() já trata IsAlive internamente)
                Parallel.For(0, _populationSize, i => cars[i].Step());

                if (step % FrameInterval == 0 && !_visualizer.IsDisposed)
                {
                    // Verifica se o usuário selecionou uma rede durante o treinamento
                    string? requested = _visualizer.TakeReplayRequest();
                    if (requested is not null)
                    {
                        PendingReplayPath = requested;
                        break;
                    }

                    int bestIdx = FindBestIndex(cars);
                    (var snaps, int aliveCount) = BuildSnapshots(cars, bestIdx);
                    _visualizer.UpdateState(snaps, _generation, BestFitnessEver,
                                            aliveCount, _populationSize);
                    Thread.Sleep(FrameMs);
                }
            }

            // Calcular fitness após o loop visual
            for (int i = 0; i < _populationSize; i++)
            {
                cars[i].Brain.Fitness    = cars[i].DistanceTraveled + cars[i].Steps * 0.1;
                cars[i].Brain.Generation = _generation;
            }
        }

        // Atualizar melhor rede de todos os tempos
        foreach (var nn in _population)
        {
            if (nn.Fitness > BestFitnessEver)
            {
                BestFitnessEver = nn.Fitness;
                BestNetwork     = nn.Clone();
            }
        }

        // Estatísticas rápidas
        double bestGen = _population.Max(n => n.Fitness);
        double avgGen  = _population.Average(n => n.Fitness);
        int    alive   = cars.Count(c => c.IsAlive);

        Console.WriteLine($"    Melhor fitness (geração)      : {bestGen,10:F2}");
        Console.WriteLine($"    Fitness médio                 : {avgGen,10:F2}");
        Console.WriteLine($"    Carros sobreviventes ao final : {alive,10}");
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine("    ✔  Fase 2 concluída.");
        Console.ResetColor();
    }

    // ════════════════════════════════════════════════════════════════════════
    // FASE 3 — EVOLUÇÃO
    // ════════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Seleciona os melhores indivíduos (elite), preserva-os intactos e
    /// preenche o restante da população com clones mutados.
    /// Método: Random Mutations (sem crossover, conforme DeepCars original).
    /// </summary>
    private void Phase3_Evolve()
    {
        Console.ForegroundColor = ConsoleColor.Magenta;
        Console.WriteLine($"\n  ── FASE 3 — EVOLUÇÃO   (geração {_generation}) ──");
        Console.ResetColor();

        // 1. Ordenar população por fitness (decrescente)
        _population.Sort((a, b) => b.Fitness.CompareTo(a.Fitness));

        // 2. Preservar elite (sem mutação)
        var eliteCount  = Math.Min(_eliteCount, _populationSize);
        var newPop      = new List<NeuralNetwork>(_populationSize);

        for (int i = 0; i < eliteCount; i++)
            newPop.Add(_population[i].Clone());

        // 3. Preencher o restante com mutações da elite
        int offspring = _populationSize - eliteCount;
        for (int i = 0; i < offspring; i++)
        {
            // Sortear um pai da elite usando torneio simples (top-elite)
            int parentIdx = _rng.Next(0, eliteCount);
            var child     = _population[parentIdx].Clone();
            child.Mutate(_rng, _mutationRate, _mutationStrength);
            child.Fitness = 0.0; // resetar para a próxima avaliação
            newPop.Add(child);
        }

        _population = newPop;

        Console.WriteLine($"    Elite preservada         : {eliteCount}");
        Console.WriteLine($"    Novos indivíduos         : {offspring}");
        Console.WriteLine($"    Taxa de mutação          : {_mutationRate * 100:F0}%  " +
                          $"(força = {_mutationStrength:F2})");
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine("    ✔  Fase 3 concluída.");
        Console.ResetColor();
    }

    // ════════════════════════════════════════════════════════════════════════
    // HELPERS DE LOG
    // ════════════════════════════════════════════════════════════════════════

    // ── Helpers para o modo visual ────────────────────────────────────────────

    /// <summary>Retorna o índice do carro que percorreu a maior distância.</summary>
    private static int FindBestIndex(Car[] cars)
    {
        int best = 0;
        for (int i = 1; i < cars.Length; i++)
            if (cars[i].DistanceTraveled > cars[best].DistanceTraveled)
                best = i;
        return best;
    }

    /// <summary>
    /// Constrói o array de snapshots para o visualizador. O carro em
    /// <paramref name="bestIdx"/> recebe IsBest=true e seus sensores são copiados.
    /// </summary>
    private static (CarSnapshot[] Snapshots, int AliveCount) BuildSnapshots(Car[] cars, int bestIdx)
    {
        var snaps = new CarSnapshot[cars.Length];
        int alive = 0;
        for (int i = 0; i < cars.Length; i++)
        {
            var c    = cars[i];
            bool best = i == bestIdx;
            snaps[i] = new CarSnapshot
            {
                X       = (float)c.X,
                Y       = (float)c.Y,
                Angle   = (float)c.Angle,
                IsAlive = c.IsAlive,
                IsBest  = best,
                Sensors = best ? (double[])c.Sensors.Clone() : null
            };
            if (c.IsAlive) alive++;
        }
        return (snaps, alive);
    }

    private string? SaveBestNetwork()
    {
        if (BestNetwork is null) return null;

        Directory.CreateDirectory(NetworkFolder);
        string fileName = $"best_network_gen{BestNetwork.Generation}_fit{BestFitnessEver:F0}.json";
        string fullPath = Path.Combine(NetworkFolder, fileName);

        BestNetwork.SaveToJson(fullPath);
        _visualizer?.RefreshNetworkList();

        Console.WriteLine();
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("  Melhor rede neural salva em:");
        Console.WriteLine($"  {fullPath}");
        Console.ResetColor();
        return fullPath;
    }

    // ════════════════════════════════════════════════════════════════════════
    // REPLAY — executa em loop a rede carregada do JSON
    // ════════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Carrega a rede salva em <paramref name="jsonPath"/> e a executa em loop
    /// no <paramref name="visualizer"/> até o usuário fechar a janela.
    /// Ao morrer, o carro é reiniciado na posição de largada.
    /// </summary>
    public static void ReplaySaved(string jsonPath, TrackVisualizer visualizer)
    {
        // Verifica se o usuário já selecionou outra rede antes do replay iniciar
        string? pending = visualizer.TakeReplayRequest();
        var nn = NeuralNetwork.LoadFromJson(pending ?? jsonPath);

        Console.WriteLine();
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("  Iniciando replay da melhor rede...");
        Console.WriteLine($"  Geração: {nn.Generation}  |  Fitness: {nn.Fitness:F2}");
        Console.ResetColor();

        const int FrameInterval = 3;
        const int FrameMs       = 16;

        int lapCount = 0;

        while (!visualizer.IsDisposed)
        {
            var car     = new Car(nn, Track.StartX, Track.StartY, Track.StartAngle);
            bool switched = false;

            while (car.IsAlive && !visualizer.IsDisposed)
            {
                // Verifica solicitação de troca de rede pelo ComboBox
                string? requested = visualizer.TakeReplayRequest();
                if (requested != null)
                {
                    try
                    {
                        nn = NeuralNetwork.LoadFromJson(requested);
                        Console.WriteLine($"\n  [Replay] Rede alterada: {Path.GetFileNameWithoutExtension(requested)}");
                        Console.WriteLine($"  Geração: {nn.Generation}  |  Fitness: {nn.Fitness:F2}");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"\n  [Replay] Erro ao carregar rede: {ex.Message}");
                    }
                    lapCount  = 0;
                    switched  = true;
                    break;
                }

                for (int i = 0; i < FrameInterval && car.IsAlive; i++)
                    car.Step();

                var snap = new CarSnapshot
                {
                    X       = (float)car.X,
                    Y       = (float)car.Y,
                    Angle   = (float)car.Angle,
                    IsAlive = car.IsAlive,
                    IsBest  = true,
                    Sensors = (double[])car.Sensors.Clone()
                };

                visualizer.UpdateState([snap], nn.Generation, nn.Fitness,
                                       car.IsAlive ? 1 : 0, 1);
                Thread.Sleep(FrameMs);
            }

            if (!switched)
            {
                lapCount++;
                Console.WriteLine($"  [Replay] Tentativa {lapCount} — distância: {car.DistanceTraveled:F1}");
            }
        }
    }

    private static void PrintHeader()
    {
        Console.ForegroundColor = ConsoleColor.White;
        Console.WriteLine();
        Console.WriteLine("╔═══════════════════════════════════════════════════════╗");
        Console.WriteLine("║           DEEP CARS — Treinamento Evolutivo           ║");
        Console.WriteLine("║        Rede Neural MLP + Random Mutations (C#)        ║");
        Console.WriteLine("╚═══════════════════════════════════════════════════════╝");
        Console.ResetColor();
    }

    private void PrintGenerationSummary(long elapsedMs)
    {
        Console.ForegroundColor = ConsoleColor.DarkCyan;
        Console.WriteLine($"\n  >> Geração {_generation,3} concluída em {elapsedMs} ms  |  " +
                          $"Melhor fitness histórico: {BestFitnessEver:F2}  |  " +
                          $"Chicane: {_currentChicane:F0}px");
        Console.WriteLine("  ─────────────────────────────────────────────────────");
        Console.ResetColor();
    }

    private void PrintFinalSummary(TimeSpan elapsed)
    {
        Console.WriteLine();
        Console.ForegroundColor = ConsoleColor.White;
        Console.WriteLine("╔═══════════════════════════════════════════════════════╗");
        Console.WriteLine("║                 TREINAMENTO CONCLUÍDO                 ║");
        Console.WriteLine("╚═══════════════════════════════════════════════════════╝");
        Console.ResetColor();
        Console.WriteLine($"  Gerações executadas   : {_generation}");
        Console.WriteLine($"  Tamanho da população  : {_populationSize}");
        Console.WriteLine($"  Melhor fitness final  : {BestFitnessEver:F2}");
        Console.WriteLine($"  Tempo total           : {elapsed:mm\\:ss\\.fff}");
        if (BestNetwork != null)
            Console.WriteLine($"  Melhor rede surgiu na geração: {BestNetwork.Generation}");
    }
}
