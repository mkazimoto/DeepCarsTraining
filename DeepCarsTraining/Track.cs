using System.Drawing;

namespace DeepCarsTraining;

/// <summary>
/// Define a geometria de uma pista em forma de estádio:
///   dois semicírculos (esquerdo e direito) conectados por retas superior e inferior.
///
///   Largura da pista: OuterR - InnerR = 80 px em todo o percurso.
///
/// Oferece:
///   • <see cref="IsOnTrack"/> — verifica se um ponto está na área válida da pista
///   • <see cref="Raycast"/>   — distância à parede mais próxima (LIDAR)
/// </summary>
public static class Track
{
    // ── Geometria ─────────────────────────────────────────────────────────────
    private const double LeftCx  = 250.0;  // centro x do semicírculo esquerdo
    private const double RightCx = 550.0;  // centro x do semicírculo direito
    private const double TrackCy = 300.0;  // centro y (compartilhado pelos dois arcos)

    private const double OuterR  = 200.0;  // raio externo → largura = OuterR − InnerR = 80 px
    private const double InnerR  = 120.0;  // raio interno (ilha central)

    private const int SemiPts = 48;        // amostras por semicírculo (mais = mais suave)

    // ── Posição e ângulo de largada ──────────────────────────────────────────
    /// <summary>Centro geométrico da reta inferior da pista.</summary>
    public const double StartX     = 400.0;
    /// <summary>Centro da faixa na reta inferior (OuterR + InnerR) / 2 = 460.</summary>
    public const double StartY     = 460.0;
    /// <summary>Aponta para a direita — sentido anti-horário visto no ecrã.</summary>
    public const double StartAngle = 0.0;

    // ── Polígonos das paredes (double, para Car.cs / LIDAR) ───────────────────
    /// <summary>Parede externa da pista como sequência de vértices.</summary>
    public static readonly (double X, double Y)[] OuterWall = BuildStadium(OuterR);

    /// <summary>Parede interna (ilha) como sequência de vértices.</summary>
    public static readonly (double X, double Y)[] InnerWall = BuildStadium(InnerR);

    // ── Versão PointF para o renderizador GDI+ ───────────────────────────────
    /// <summary>Parede externa em PointF (para GDI+).</summary>
    public static readonly PointF[] OuterWallF = ToPointF(OuterWall);

    /// <summary>Parede interna em PointF (para GDI+).</summary>
    public static readonly PointF[] InnerWallF = ToPointF(InnerWall);

    // ═════════════════════════════════════════════════════════════════════════
    // API PÚBLICA
    // ═════════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Retorna <see langword="true"/> se o ponto (x, y) estiver dentro da área
    /// pavimentada da pista (interior do polígono externo e exterior do interno).
    /// </summary>
    public static bool IsOnTrack(double x, double y)
        => IsInsidePolygon(OuterWall, x, y) && !IsInsidePolygon(InnerWall, x, y);

    /// <summary>
    /// Lança um raio a partir de (ox, oy) na direção (dirX, dirY) e retorna
    /// a distância até a parede mais próxima, limitada a <paramref name="maxRange"/>.
    /// </summary>
    public static double Raycast(
        double ox, double oy,
        double dirX, double dirY,
        double maxRange)
    {
        double t = maxRange;
        RaycastPolygon(OuterWall, ox, oy, dirX, dirY, ref t);
        RaycastPolygon(InnerWall, ox, oy, dirX, dirY, ref t);
        return t;
    }

    // ═════════════════════════════════════════════════════════════════════════
    // CONSTRUÇÃO DOS POLÍGONOS
    // ═════════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Constrói o polígono "estádio" para o raio dado.
    ///
    /// Ordem dos vértices (sentido horário na tela, y crescendo para baixo):
    ///   1. Semicírculo esquerdo de 270° → 90° (passando por 180°)
    ///   2. Semicírculo direito  de  90° → −90° (passando por 0°)
    ///   3. Fechamento implícito: topo-direito → topo-esquerdo (reta superior)
    ///
    /// As retas inferior e superior emergem naturalmente das arestas que conectam
    /// o final de um arco ao início do seguinte.
    /// </summary>
    private static (double X, double Y)[] BuildStadium(double radius)
    {
        var pts = new List<(double X, double Y)>(SemiPts * 2 + 4);

        // Semicírculo esquerdo: 270° → 90° (lado esquerdo, em sentido horário na tela)
        for (int i = 0; i <= SemiPts; i++)
        {
            double a = (270.0 - 180.0 * i / SemiPts) * Math.PI / 180.0;
            pts.Add((LeftCx + radius * Math.Cos(a),
                     TrackCy + radius * Math.Sin(a)));
        }

        // Semicírculo direito: 90° → −90° (lado direito, em sentido horário na tela)
        for (int i = 0; i <= SemiPts; i++)
        {
            double a = (90.0 - 180.0 * i / SemiPts) * Math.PI / 180.0;
            pts.Add((RightCx + radius * Math.Cos(a),
                     TrackCy + radius * Math.Sin(a)));
        }

        return pts.ToArray();
    }

    private static PointF[] ToPointF((double X, double Y)[] src)
    {
        var dst = new PointF[src.Length];
        for (int i = 0; i < src.Length; i++)
            dst[i] = new PointF((float)src[i].X, (float)src[i].Y);
        return dst;
    }

    // ═════════════════════════════════════════════════════════════════════════
    // RAYCASTING CONTRA UM POLÍGONO
    // ═════════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Testa a interseção do raio com cada aresta do polígono e atualiza
    /// <paramref name="tMin"/> com o menor t positivo encontrado.
    ///
    /// Fórmula paramétrica (Möller-Trumbore 2D):
    ///   raio  P = O + t·d            (t ≥ 0)
    ///   aresta Q = A + s·(B−A)       (s ∈ [0,1])
    ///   cruzamento 2D: d × (B−A) = denominador ≠ 0
    /// </summary>
    private static void RaycastPolygon(
        (double X, double Y)[] poly,
        double ox, double oy,
        double dirX, double dirY,
        ref double tMin)
    {
        int n = poly.Length;
        for (int i = 0; i < n; i++)
        {
            var (ax, ay) = poly[i];
            var (bx, by) = poly[(i + 1) % n];

            double edX = bx - ax;
            double edY = by - ay;

            // denominador = d × e  (produto vetorial 2D)
            double denom = dirX * edY - dirY * edX;
            if (Math.Abs(denom) < 1e-10) continue;  // paralelo → ignora

            double fx = ax - ox;
            double fy = ay - oy;

            double t = (fx * edY - fy * edX) / denom;  // distância pelo raio
            double s = (fx * dirY - fy * dirX) / denom; // parâmetro na aresta

            if (t > 0.0 && s >= 0.0 && s <= 1.0 && t < tMin)
                tMin = t;
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // POINT-IN-POLYGON (ray casting horizontal)
    // ═════════════════════════════════════════════════════════════════════════

    private static bool IsInsidePolygon((double X, double Y)[] poly, double px, double py)
    {
        bool inside = false;
        int  n = poly.Length;
        int  j = n - 1;

        for (int i = 0; i < n; i++)
        {
            double xi = poly[i].X, yi = poly[i].Y;
            double xj = poly[j].X, yj = poly[j].Y;

            if ((yi > py) != (yj > py) &&
                px < (xj - xi) * (py - yi) / (yj - yi) + xi)
            {
                inside = !inside;
            }

            j = i;
        }

        return inside;
    }
}
