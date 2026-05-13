using System.Text.Json;

namespace DeepCarsTraining;

/// <summary>
/// Rede Neural Artificial — Perceptron Multilayer (MLP) de 3 camadas
/// Arquitetura fiel ao projeto DeepCars (https://github.com/JVictorDias/DeepCars/):
///   Entrada : 18 sensores LIDAR + 1 velocidade + 1 viés = 20 neurônios
///   Oculta  :  6 neurônios + 1 viés =  7 neurônios
///   Saída   :  4 neurônios (Acelerar, Ré, Virar Esquerda, Virar Direita)
/// Ativação : ReLU na camada oculta; linear na camada de saída
/// </summary>
public sealed class NeuralNetwork
{
    // ── Dimensões fixas da arquitetura ──────────────────────────────────────
    public const int InputNeurons  = 19;   // 18 sensores LIDAR + 1 velocidade normalizada
    public const int HiddenNeurons = 6;
    public const int OutputNeurons = 4;    // Acelerar | Ré | Esquerda | Direita

    private const int InputWithBias  = InputNeurons  + 1; // 19
    private const int HiddenWithBias = HiddenNeurons + 1; // 7

    // ── Pesos ───────────────────────────────────────────────────────────────
    // [entrada+viés, neurônio oculto]
    public double[,] WeightsInputHidden  { get; private set; } = new double[InputWithBias,  HiddenNeurons];
    // [oculto+viés,  neurônio de saída]
    public double[,] WeightsHiddenOutput { get; private set; } = new double[HiddenWithBias, OutputNeurons];

    // ── Metadados de evolução ───────────────────────────────────────────────
    public double Fitness    { get; set; } = 0.0;
    public int    Generation { get; set; } = 0;

    // ── Construtor ──────────────────────────────────────────────────────────
    public NeuralNetwork() { }

    // ── Inicialização aleatória ─────────────────────────────────────────────
    /// <summary>Inicializa todos os pesos com valores uniformes em [-1, 1].</summary>
    public void InitializeRandom(Random rng)
    {
        for (int i = 0; i < InputWithBias; i++)
            for (int h = 0; h < HiddenNeurons; h++)
                WeightsInputHidden[i, h] = rng.NextDouble() * 2.0 - 1.0;

        for (int h = 0; h < HiddenWithBias; h++)
            for (int o = 0; o < OutputNeurons; o++)
                WeightsHiddenOutput[h, o] = rng.NextDouble() * 2.0 - 1.0;
    }

    // ── Forward pass ────────────────────────────────────────────────────────
    /// <summary>
    /// Propaga os sensores pela rede e retorna os 4 valores de saída.
    /// </summary>
    /// <param name="sensors">Array de 18 doubles (valores dos sensores LIDAR).</param>
    /// <returns>Array de 4 doubles — [Acelerar, Ré, Esquerda, Direita].</returns>
    public double[] Forward(double[] sensors)
    {
        if (sensors.Length != InputNeurons)
            throw new ArgumentException($"Esperado {InputNeurons} sensores, recebido {sensors.Length}.");

        // ── Camada de entrada → oculta ──────────────────────────────────────
        // Montar vetor de entrada com viés = 1.0 no índice 18
        Span<double> inputWithBias = stackalloc double[InputWithBias];
        for (int i = 0; i < InputNeurons; i++)
            inputWithBias[i] = sensors[i];
        inputWithBias[InputNeurons] = 1.0; // viés

        Span<double> hidden = stackalloc double[HiddenNeurons];
        for (int h = 0; h < HiddenNeurons; h++)
        {
            double sum = 0.0;
            for (int i = 0; i < InputWithBias; i++)
                sum += inputWithBias[i] * WeightsInputHidden[i, h];
            hidden[h] = ReLU(sum);
        }

        // ── Camada oculta → saída ───────────────────────────────────────────
        // Montar vetor oculto com viés = 1.0 no índice 6
        Span<double> hiddenWithBias = stackalloc double[HiddenWithBias];
        for (int h = 0; h < HiddenNeurons; h++)
            hiddenWithBias[h] = hidden[h];
        hiddenWithBias[HiddenNeurons] = 1.0; // viés

        double[] output = new double[OutputNeurons];
        for (int o = 0; o < OutputNeurons; o++)
        {
            double sum = 0.0;
            for (int h = 0; h < HiddenWithBias; h++)
                sum += hiddenWithBias[h] * WeightsHiddenOutput[h, o];
            output[o] = sum; // ativação linear — permite valores negativos para que a comparação sempre resolva
        }

        return output;
    }

    // ── Mutação (Random Mutations) ──────────────────────────────────────────
    /// <summary>
    /// Aplica mutações aleatórias nos pesos.
    /// Cada peso tem <paramref name="mutationRate"/> de chance de ser perturbado.
    /// </summary>
    public void Mutate(Random rng, double mutationRate = 0.10, double mutationStrength = 0.30)
    {
        for (int i = 0; i < InputWithBias; i++)
            for (int h = 0; h < HiddenNeurons; h++)
                if (rng.NextDouble() < mutationRate)
                    WeightsInputHidden[i, h] += (rng.NextDouble() * 2.0 - 1.0) * mutationStrength;

        for (int h = 0; h < HiddenWithBias; h++)
            for (int o = 0; o < OutputNeurons; o++)
                if (rng.NextDouble() < mutationRate)
                    WeightsHiddenOutput[h, o] += (rng.NextDouble() * 2.0 - 1.0) * mutationStrength;
    }

    // ── Clone ───────────────────────────────────────────────────────────────
    /// <summary>Retorna uma cópia profunda desta rede neural.</summary>
    public NeuralNetwork Clone()
    {
        var clone = new NeuralNetwork
        {
            Fitness    = Fitness,
            Generation = Generation
        };

        Buffer.BlockCopy(WeightsInputHidden,  0, clone.WeightsInputHidden,  0,
            InputWithBias  * HiddenNeurons * sizeof(double));
        Buffer.BlockCopy(WeightsHiddenOutput, 0, clone.WeightsHiddenOutput, 0,
            HiddenWithBias * OutputNeurons * sizeof(double));

        return clone;
    }
    // ── Persistência JSON ──────────────────────────────────────────────────
    /// <summary>Serializa a rede e seus metadados para um arquivo JSON.</summary>
    public void SaveToJson(string path)
    {
        var dto = new NeuralNetworkDto
        {
            Fitness             = Fitness,
            Generation          = Generation,
            WeightsInputHidden  = ToJagged(WeightsInputHidden,  InputWithBias,  HiddenNeurons),
            WeightsHiddenOutput = ToJagged(WeightsHiddenOutput, HiddenWithBias, OutputNeurons)
        };
        string json = JsonSerializer.Serialize(dto, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(path, json);
    }

    /// <summary>Carrega uma rede neural a partir de um arquivo JSON.</summary>
    public static NeuralNetwork LoadFromJson(string path)
    {
        string json = File.ReadAllText(path);
        var dto = JsonSerializer.Deserialize<NeuralNetworkDto>(json)
                  ?? throw new InvalidDataException("Arquivo JSON inválido.");

        var nn = new NeuralNetwork { Fitness = dto.Fitness, Generation = dto.Generation };
        FromJagged(dto.WeightsInputHidden,  nn.WeightsInputHidden,  InputWithBias,  HiddenNeurons);
        FromJagged(dto.WeightsHiddenOutput, nn.WeightsHiddenOutput, HiddenWithBias, OutputNeurons);
        return nn;
    }

    private static double[][] ToJagged(double[,] src, int rows, int cols)
    {
        var jag = new double[rows][];
        for (int r = 0; r < rows; r++)
        {
            jag[r] = new double[cols];
            for (int c = 0; c < cols; c++)
                jag[r][c] = src[r, c];
        }
        return jag;
    }

    private static void FromJagged(double[][] src, double[,] dst, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                dst[r, c] = src[r][c];
    }

    // DTO privado para serialização — System.Text.Json não suporta double[,] nativamente
    private sealed class NeuralNetworkDto
    {
        public double     Fitness             { get; set; }
        public int        Generation          { get; set; }
        public double[][] WeightsInputHidden  { get; set; } = [];
        public double[][] WeightsHiddenOutput { get; set; } = [];
    }
    // ── Helpers ─────────────────────────────────────────────────────────────
    private static double ReLU(double x) => x > 0.0 ? x : 0.0;

    public int TotalWeights =>
        InputWithBias * HiddenNeurons + HiddenWithBias * OutputNeurons; // 19×6 + 7×4 = 142
}
