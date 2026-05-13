using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace DeepCarsTraining;

/// <summary>
/// Snapshot imutável do estado de um carro — transferência thread-safe ao visualizador.
/// </summary>
public readonly struct CarSnapshot
{
    public float     X       { get; init; }
    public float     Y       { get; init; }
    public float     Angle   { get; init; }  // graus
    public bool      IsAlive { get; init; }
    public bool      IsBest  { get; init; }
    public double[]? Sensors { get; init; }  // apenas para o carro IsBest
}

/// <summary>
/// Janela gráfica que exibe a pista e os carros em tempo real durante o treinamento evolutivo.
/// Executa em uma thread STA dedicada; a thread de simulação atualiza o estado via
/// <see cref="UpdateState"/>.
/// </summary>
public sealed class TrackVisualizer : Form
{
  // ── Layout ───────────────────────────────────────────────────────────────
  private const int HudHeight = 72;
  private const int TrackPadding = 20;

  private static readonly int ClientW = (int)Car.TrackWidth + TrackPadding * 2;   // 840
  private static readonly int ClientH = (int)Car.TrackHeight + TrackPadding * 2 + HudHeight; // 712

  private const float TrackOriginX = TrackPadding;
  private const float TrackOriginY = HudHeight + TrackPadding;

  // ── Parâmetros LIDAR (devem coincidir com Car.cs) ────────────────────────
  private const double SensorMaxRange = 200.0;
  private static readonly double[] SensorAngles = BuildSensorAngles();

  // ── Estado compartilhado ─────────────────────────────────────────────────
  private CarSnapshot[]? _snapshots;
  private int _generation;
  private double _bestFitness;
  private int _aliveCount;
  private int _populationSize;
  private bool _trainingDone;
  private readonly object _stateLock = new();
  // ── Botão de curvas ─────────────────────────────────────────────
  private static readonly double[] CurvePresets = [0.0, 20.0, 40.0, 60.0, 80.0];
  private static readonly string[] CurveLabels  = ["Reta", "Leve", "Média", "Forte", "Extrema"];
  private int _curveIndex = 0; // padrão: 0 px 
  private Button? _btnCurves;
  // ── ComboBox de seleção de rede neural ────────────────────────────────
  private ComboBox? _cmbNetworks;
  private string?   _requestedReplayPath;
  private readonly object _replayLock = new();
  // ── Recursos de desenho (pré-criados para evitar alocações no OnPaint) ───
  private readonly Font _fontTitle = new("Consolas", 11, FontStyle.Bold);
  private readonly Font _fontInfo = new("Consolas", 9);
  private readonly Font _fontDone = new("Consolas", 16, FontStyle.Bold);

  private readonly SolidBrush _brushHudBack = new(Color.FromArgb(200, 18, 18, 18));
  private readonly SolidBrush _brushTitle = new(Color.FromArgb(80, 200, 255));
  private readonly SolidBrush _brushText = new(Color.White);
  private readonly SolidBrush _brushSubText = new(Color.FromArgb(150, 150, 150));
  private readonly SolidBrush _brushGrass   = new(Color.FromArgb(28,  72,  20));  // fundo externo à pista
  private readonly SolidBrush _brushIsland  = new(Color.FromArgb(34,  85,  25));  // ilha interna
  private readonly SolidBrush _brushTrack = new(Color.FromArgb(45, 45, 45));      // asfalto
  private readonly SolidBrush _brushDead = new(Color.FromArgb(50, 210, 60, 60));
  private readonly SolidBrush _brushAlive = new(Color.FromArgb(180, 60, 210, 60));
  private readonly SolidBrush _brushBest = new(Color.Gold);
  private readonly SolidBrush _brushLidarDot = new(Color.FromArgb(200, 0, 220, 255));
  private readonly SolidBrush _brushDoneBack = new(Color.FromArgb(200, 10, 10, 10));
  private readonly SolidBrush _brushDoneText = new(Color.FromArgb(80, 220, 130));

  private readonly Pen _penBorder = new(Color.FromArgb(200, 200, 200), 2f);
  private readonly Pen _penCollision = new(Color.FromArgb(80, 220, 160, 40), 1f) { DashStyle = DashStyle.Dash };
  private readonly Pen _penBestBorder = new(Color.White, 1.5f);
  private readonly Pen _penLidar = new(Color.FromArgb(80, 0, 180, 255), 1f);

  // ── Construtor privado ───────────────────────────────────────────────────
  private TrackVisualizer()
  {
    Text = "Deep Cars — Treinamento Evolutivo da Rede Neural";
    ClientSize = new Size(ClientW, ClientH);
    DoubleBuffered = true;
    FormBorderStyle = FormBorderStyle.FixedSingle;
    MaximizeBox = false;
    BackColor = Color.FromArgb(25, 25, 25);

    var timer = new System.Windows.Forms.Timer { Interval = 16 }; // ~60 FPS
    timer.Tick += (_, _) => { if (!IsDisposed) Invalidate(); };
    timer.Start();

    // Botão para alterar as curvas da pista
    _btnCurves = new Button
    {
      Text      = $"Curvas: {CurveLabels[_curveIndex]} ({CurvePresets[_curveIndex]:F0}px)",
      Size      = new Size(180, 28),
      Location  = new Point(ClientW - 190, (HudHeight - 28) / 2),
      FlatStyle = FlatStyle.Flat,
      BackColor = Color.FromArgb(55, 55, 80),
      ForeColor = Color.White,
      Font      = new Font("Consolas", 8),
      Cursor    = Cursors.Hand,
    };
    _btnCurves.FlatAppearance.BorderColor = Color.FromArgb(100, 130, 200);
    _btnCurves.Click += OnBtnCurvesClick;
    Controls.Add(_btnCurves);

    // ── ComboBox de redes neurais ─────────────────────────────────────────
    _cmbNetworks = new ComboBox
    {
      Size          = _btnCurves.Size,
      Location      = new Point(ClientW - 190, _btnCurves.Location.Y + _btnCurves.Size.Height + 2),
      DropDownStyle = ComboBoxStyle.DropDownList,
      BackColor     = Color.FromArgb(40, 40, 60),
      ForeColor     = Color.White,
      Font          = new Font("Consolas", 8),
      Cursor        = Cursors.Hand,
    };
    _cmbNetworks.DropDown             += (_, _) => RefreshNetworkList();
    _cmbNetworks.SelectedIndexChanged += OnCmbNetworkSelected;
    Controls.Add(_cmbNetworks);
    RefreshNetworkList();
  }

  // ── API pública ──────────────────────────────────────────────────────────
  private void OnBtnCurvesClick(object? sender, EventArgs e)
  {
    _curveIndex = (_curveIndex + 1) % CurvePresets.Length;
    double amp  = CurvePresets[_curveIndex];
    Track.Rebuild(amp);
    if (_btnCurves is not null)
      _btnCurves.Text = $"Curvas: {CurveLabels[_curveIndex]} ({amp:F0}px)";
  }
  /// <summary>
  /// Atualiza o estado exibido. Thread-safe — pode ser chamado da thread de simulação.
  /// </summary>
  public void UpdateState(CarSnapshot[] snapshots, int generation,
                          double bestFitness, int aliveCount, int populationSize)
  {
    if (IsDisposed) return;
    lock (_stateLock)
    {
      _snapshots = snapshots;
      _generation = generation;
      _bestFitness = bestFitness;
      _aliveCount = aliveCount;
      _populationSize = populationSize;
    }
  }

  /// <summary>
  /// Exibe "Treinamento concluído" no HUD.
  /// </summary>
  public void SignalTrainingComplete()
  {
    if (IsDisposed) return;
    lock (_stateLock) { _trainingDone = true; }
  }

  /// <summary>Bloqueia até o usuário fechar a janela.</summary>
  public void WaitForClose()
  {
    while (!IsDisposed)
      Thread.Sleep(100);
  }

  /// <summary>
  /// Retorna (e limpa) o caminho de rede solicitado pelo ComboBox. Thread-safe.
  /// Retorna null se não houver solicitação pendente.
  /// </summary>
  public string? TakeReplayRequest()
  {
    lock (_replayLock)
    {
      var path = _requestedReplayPath;
      _requestedReplayPath = null;
      return path;
    }
  }

  /// <summary>
  /// Atualiza a lista do ComboBox com os arquivos JSON da pasta RedeNeural.
  /// Thread-safe — pode ser chamado da thread de treinamento.
  /// </summary>
  public void RefreshNetworkList()
  {
    if (_cmbNetworks is null || IsDisposed) return;

    string folder = Path.Combine(AppContext.BaseDirectory, "RedeNeural");
    if (!Directory.Exists(folder)) return;

    var files = Directory.GetFiles(folder, "*.json")
                         .OrderByDescending(f => f)
                         .ToArray();

    void UpdateUI()
    {
      var prev = _cmbNetworks.SelectedItem as NetworkItem;
      _cmbNetworks.SelectedIndexChanged -= OnCmbNetworkSelected;
      _cmbNetworks.Items.Clear();
      foreach (var f in files)
        _cmbNetworks.Items.Add(new NetworkItem(f));

      if (prev != null)
      {
        var match = _cmbNetworks.Items.Cast<NetworkItem>()
                                .FirstOrDefault(n => n.Path == prev.Path);
        if (match != null) _cmbNetworks.SelectedItem = match;
        else if (_cmbNetworks.Items.Count > 0) _cmbNetworks.SelectedIndex = 0;
      }
      else if (_cmbNetworks.Items.Count > 0)
        _cmbNetworks.SelectedIndex = 0;

      _cmbNetworks.SelectedIndexChanged += OnCmbNetworkSelected;
    }

    if (_cmbNetworks.InvokeRequired)
      _cmbNetworks.Invoke(UpdateUI);
    else
      UpdateUI();
  }

  private void OnCmbNetworkSelected(object? sender, EventArgs e)
  {
    if (_cmbNetworks?.SelectedItem is NetworkItem item)
    {
      lock (_replayLock)
        _requestedReplayPath = item.Path;
    }
  }

  /// <summary>
  /// Cria e exibe o visualizador em uma thread STA dedicada.
  /// Retorna somente após a janela estar pronta para receber atualizações.
  /// </summary>
  public static TrackVisualizer StartOnNewThread()
  {
    TrackVisualizer? instance = null;
    using var ready = new ManualResetEventSlim(false);

    var uiThread = new Thread(() =>
    {
      Application.EnableVisualStyles();
      Application.SetCompatibleTextRenderingDefault(false);
      instance = new TrackVisualizer();
      instance.StartPosition = FormStartPosition.CenterScreen;
      instance.HandleCreated += (_, _) => ready.Set();
      Application.Run(instance);
    });

    uiThread.SetApartmentState(ApartmentState.STA);
    uiThread.IsBackground = true;
    uiThread.Name = "VisualizerThread";
    uiThread.Start();

    ready.Wait();
    return instance!;
  }

  // ── Renderização ─────────────────────────────────────────────────────────

  protected override void OnPaint(PaintEventArgs e)
  {
    CarSnapshot[]? snaps;
    int gen; double best; int alive; int pop; bool done;

    lock (_stateLock)
    {
      snaps = _snapshots;
      gen = _generation;
      best = _bestFitness;
      alive = _aliveCount;
      pop = _populationSize;
      done = _trainingDone;
    }

    var g = e.Graphics;
    g.SmoothingMode = SmoothingMode.AntiAlias;

    DrawHUD(g, gen, best, alive, snaps?.Length ?? 0, pop);

    // Translação para a área da pista
    g.TranslateTransform(TrackOriginX, TrackOriginY);
    DrawTrack(g);

    if (snaps is { Length: > 0 })
    {
      DrawDeadCars(g, snaps);
      DrawAliveCars(g, snaps);
      DrawBestCar(g, snaps);
      DrawLidar(g, snaps);
    }

    g.ResetTransform();

    if (done)
      DrawDoneOverlay(g);
  }

  private void DrawHUD(Graphics g, int gen, double best, int alive, int total, int pop)
  {
    g.FillRectangle(_brushHudBack, 0, 0, ClientW, HudHeight);
    g.DrawString("DEEP CARS — Treinamento Evolutivo da Rede Neural", _fontTitle, _brushTitle, 10, 7);
    g.DrawString(
        $"Geração: {gen,3}   Melhor Fitness: {best,10:F1}   Sobreviventes: {alive,4}/{total,-4}   População: {pop}",
        _fontInfo, _brushText, 10, 35);
    g.DrawString(
        "● Melhor carro (ouro)   ● Sobreviventes (verde)   ● Mortos (verm.)   ─ LIDAR (azul)",
        _fontInfo, _brushSubText, 10, 53);
  }

  private void DrawTrack(Graphics g)
  {
    float tw = (float)Car.TrackWidth;
    float th = (float)Car.TrackHeight;

        // 1. Fundo geral (grama externa)
        g.FillRectangle(_brushGrass, 0, 0, tw, th);

        // 2. Área da pista (asfalto) usando FillMode.Alternate:
        //    preenche o anel entre o polígono externo e o interno
        using var path = new System.Drawing.Drawing2D.GraphicsPath(
            System.Drawing.Drawing2D.FillMode.Alternate);
        path.AddPolygon(Track.OuterWallF);
        path.AddPolygon(Track.InnerWallF);
        g.FillPath(_brushTrack, path);

        // 3. Ilha interna (grama do miolo)
        g.FillPolygon(_brushIsland, Track.InnerWallF);

        // 4. Bordas das paredes
        g.DrawPolygon(_penBorder, Track.OuterWallF);
        g.DrawPolygon(_penBorder, Track.InnerWallF);

        // 5. Linha de partida — reta inferior, linha branca tracejada
        float startX = (float)Track.StartX;
        float innerY = (float)(300.0 + 120.0); // topo da parede interna na reta
        float outerY = (float)(300.0 + 200.0); // fundo da parede externa na reta
        using var penStart = new Pen(Color.FromArgb(220, 220, 220), 2f)
        {
            DashStyle = System.Drawing.Drawing2D.DashStyle.Dash
        };
        g.DrawLine(penStart, startX, innerY, startX, outerY);
  }

  private void DrawDeadCars(Graphics g, CarSnapshot[] snaps)
  {
    foreach (var s in snaps)
      if (!s.IsAlive && !s.IsBest)
        DrawCarShape(g, s, _brushDead, null);
  }

  private void DrawAliveCars(Graphics g, CarSnapshot[] snaps)
  {
    foreach (var s in snaps)
      if (s.IsAlive && !s.IsBest)
        DrawCarShape(g, s, _brushAlive, null);
  }

  private void DrawBestCar(Graphics g, CarSnapshot[] snaps)
  {
    foreach (var s in snaps)
    {
      if (!s.IsBest) continue;
      DrawCarShape(g, s, _brushBest, _penBestBorder);
      break;
    }
  }

  private static void DrawCarShape(Graphics g, CarSnapshot s, Brush fill, Pen? outline)
  {
    double rad = s.Angle * Math.PI / 180.0;
    const float frontLen = 10f;
    const float sideOff = 2.5f; // offset angular para os cantos da base (≈143°)

    var nose = new PointF(s.X + (float)(Math.Cos(rad) * frontLen),
                           s.Y + (float)(Math.Sin(rad) * frontLen));
    var left = new PointF(s.X + (float)(Math.Cos(rad + sideOff) * 6),
                           s.Y + (float)(Math.Sin(rad + sideOff) * 6));
    var right = new PointF(s.X + (float)(Math.Cos(rad - sideOff) * 6),
                           s.Y + (float)(Math.Sin(rad - sideOff) * 6));

    PointF[] tri = [nose, left, right];
    g.FillPolygon(fill, tri);
    if (outline is not null) g.DrawPolygon(outline, tri);
  }

  private void DrawLidar(Graphics g, CarSnapshot[] snaps)
  {
    foreach (var s in snaps)
    {
      if (!s.IsBest || s.Sensors is null) continue;

      for (int i = 0; i < SensorAngles.Length; i++) // SensorAngles.Length = 18 (LIDAR apenas)
      {
        double sRad = (s.Angle + SensorAngles[i]) * Math.PI / 180.0;
        double dist = s.Sensors[i] * SensorMaxRange;
        float ex = s.X + (float)(Math.Cos(sRad) * dist);
        float ey = s.Y + (float)(Math.Sin(sRad) * dist);

        g.DrawLine(_penLidar, s.X, s.Y, ex, ey);
        g.FillEllipse(_brushLidarDot, ex - 2.5f, ey - 2.5f, 5f, 5f);
      }
      break; // apenas o melhor carro
    }
  }

  private void DrawDoneOverlay(Graphics g)
  {
    const int w = 440, h = 70;
    int x = (ClientW - w) / 2;
    int y = (ClientH - h) / 2;

    g.FillRectangle(_brushDoneBack, x, y, w, h);
    using var pen = new Pen(Color.FromArgb(80, 220, 130), 2f);
    g.DrawRectangle(pen, x, y, w, h);
    g.DrawString("✔  Treinamento concluído!", _fontDone, _brushDoneText,
                 x + 20, y + 18);
  }

  private static double[] BuildSensorAngles()
  {
    // Apenas os 18 sensores LIDAR — o último input (velocidade) não é direcional
    const int LidarCount = 18;
    var a = new double[LidarCount];
    double step = 170.0 / (LidarCount - 1);
    for (int i = 0; i < LidarCount; i++)
      a[i] = -85.0 + step * i;
    return a;
  }

  private void InitializeComponent()
  {
    SuspendLayout();
    // 
    // TrackVisualizer
    // 
    ClientSize = new Size(284, 261);
    Name = "TrackVisualizer";
    StartPosition = FormStartPosition.CenterScreen;
    ResumeLayout(false);

  }

  // ── Classe auxiliar para itens do ComboBox ────────────────────────────
  private sealed class NetworkItem(string path)
  {
    public string Path { get; } = path;
    public override string ToString() => System.IO.Path.GetFileNameWithoutExtension(Path);
  }

  protected override void Dispose(bool disposing)
  {
    if (disposing)
    {
      _fontTitle.Dispose(); _fontInfo.Dispose(); _fontDone.Dispose();
      _brushHudBack.Dispose(); _brushTitle.Dispose(); _brushText.Dispose();
      _brushSubText.Dispose(); _brushGrass.Dispose(); _brushIsland.Dispose();
      _brushTrack.Dispose(); _brushDead.Dispose();
      _brushAlive.Dispose(); _brushBest.Dispose(); _brushLidarDot.Dispose();
      _brushDoneBack.Dispose(); _brushDoneText.Dispose();
      _penBorder.Dispose(); _penCollision.Dispose();
      _penBestBorder.Dispose(); _penLidar.Dispose();
    }
    base.Dispose(disposing);
  }
}
