namespace DeepCarsTraining;

/// <summary>
/// Representa um carro simulado.
/// 
/// Física simplificada (sem motor gráfico):
///   • O carro se move em um espaço 2D [0, TrackWidth] × [0, TrackHeight]
///   • 18 sensores LIDAR medem a distância às paredes da pista (normalizados em [0,1])
///   • A rede neural recebe os 18 sensores e devolve 4 ações
///   • Cada Step avança a simulação em 1 tick
///   • O carro é marcado como morto quando colide com as paredes
/// </summary>
public sealed class Car
{
    // ── Configurações da pista ──────────────────────────────────────────────
    public const double TrackWidth  = 800.0;
    public const double TrackHeight = 600.0;

    // ── Parâmetros de movimento ─────────────────────────────────────────────
    private const double Acceleration = 2.0;
    private const double Deceleration = -1.5;
    private const double MaxSpeed     = 12.0;
    private const double TurnSpeed    = 3.5;   // graus por tick
    private const double Friction     = 0.90;  // fator de desaceleração natural

    // ── Estado do carro ─────────────────────────────────────────────────────
    public double X        { get; private set; }
    public double Y        { get; private set; }
    public double Angle    { get; private set; } // graus — 0 = direita, 90 = cima
    public double Speed    { get; private set; }
    public bool   IsAlive  { get; private set; } = true;

    /// <summary>Distância total percorrida (base do fitness).</summary>
    public double DistanceTraveled { get; private set; }

    /// <summary>Número de ticks executados.</summary>
    public int Steps { get; private set; }

    // ── Rede neural do carro ─────────────────────────────────────────────────
    public NeuralNetwork Brain { get; }

    // ── Sensores ─────────────────────────────────────────────────────────────
    /// <summary>19 entradas: [0-17] leituras LIDAR normalizadas, [18] velocidade normalizada.</summary>
    public double[] Sensors { get; } = new double[NeuralNetwork.InputNeurons];

    // ── Ângulos dos sensores (relativo à frente do carro) ────────────────────────
    // 18 sensores LIDAR espaçados em 180° à frente do carro (-90° a +90°, passo de 10°)
    private static readonly double[] SensorAngles = GenerateSensorAngles();

    // ── Construtor ───────────────────────────────────────────────────────────
    public Car(NeuralNetwork brain, double startX, double startY, double startAngle = 0.0)
    {
        Brain = brain;
        X     = startX;
        Y     = startY;
        Angle = startAngle;
        Speed = 0.0;
    }

    // ── Passo de simulação ───────────────────────────────────────────────────
    /// <summary>
    /// Executa um tick de simulação:
    ///  1. Atualiza sensores LIDAR
    ///  2. Propaga pela rede neural → obtém ações
    ///  3. Aplica ações ao estado cinemático
    ///  4. Verifica colisão
    /// </summary>
    public void Step()
    {
        if (!IsAlive) return;

        // 1. Atualizar sensores
        UpdateSensors();

        // 2. Forward pass → ações [Acelerar, Ré, Esquerda, Direita]
        double[] actions = Brain.Forward(Sensors);

        // Interpretar ação com maior ativação (argmax)
        bool accel  = actions[0] > actions[1]; // Acelerar vs Ré
        bool reverse= actions[1] > actions[0];
        bool left   = actions[2] > actions[3]; // Esquerda vs Direita
        bool right  = actions[3] > actions[2];

        // 3. Aplicar ações
        if (accel)   Speed += Acceleration;
        if (reverse) Speed += Deceleration;

        Speed *= Friction;
        Speed  = Math.Clamp(Speed, -MaxSpeed * 0.5, MaxSpeed);

        if (left)  Angle -= TurnSpeed;
        if (right) Angle += TurnSpeed;

        // Normalizar ângulo [0, 360)
        Angle = ((Angle % 360.0) + 360.0) % 360.0;

        // Mover
        double rad = Angle * Math.PI / 180.0;
        double dx  = Math.Cos(rad) * Speed;
        double dy  = Math.Sin(rad) * Speed;

        X += dx;
        Y += dy;

        DistanceTraveled += Math.Sqrt(dx * dx + dy * dy);
        Steps++;

        // 4. Verificar colisão com as paredes da pista
        if (!Track.IsOnTrack(X, Y))
        {
            IsAlive = false;
        }
    }

    // ── Sensores LIDAR ───────────────────────────────────────────────────────
    /// <summary>
    /// Simula 18 raycasts a partir da posição do carro.
    /// Cada sensor retorna a distância normalizada à borda da pista mais próxima.
    /// Valor 1.0 = caminho livre; valor próximo de 0.0 = parede muito perto.
    /// O sensor extra (index 18) é a velocidade normalizada em [0, 1].
    /// </summary>
    private void UpdateSensors()
    {
        const double MaxSensorRange = 200.0;

        for (int s = 0; s < 18; s++)
        {
            double sensorAngleRad = (Angle + SensorAngles[s]) * Math.PI / 180.0;
            double dirX = Math.Cos(sensorAngleRad);
            double dirY = Math.Sin(sensorAngleRad);

            double distance = Track.Raycast(X, Y, dirX, dirY, MaxSensorRange);
            Sensors[s] = Math.Clamp(distance / MaxSensorRange, 0.0, 1.0);
        }

        // Úcltima entrada: velocidade normalizada em [0, 1]
        Sensors[18] = Math.Clamp(Math.Abs(Speed) / MaxSpeed, 0.0, 1.0);
    }

    // ── Utilidades ───────────────────────────────────────────────────────────
    private static double[] GenerateSensorAngles()
    {
        // 18 sensores LIDAR de -85° a +85° em passos de ~10°
        const int LidarCount = 18;
        var angles = new double[LidarCount];
        double start = -85.0;
        double step  = 170.0 / (LidarCount - 1);
        for (int i = 0; i < LidarCount; i++)
            angles[i] = start + step * i;
        return angles;
    }

    public override string ToString() =>
        $"Car[pos=({X:F1},{Y:F1}) angle={Angle:F1}° speed={Speed:F2} dist={DistanceTraveled:F1} alive={IsAlive}]";
}
