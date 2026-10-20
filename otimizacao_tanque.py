"""
Trabalho Computacional - Modelagem Computacional (IPRJ01-10760/IPRJ01-11873)
Instituto Politécnico - UERJ - Prof. Gustavo Barbosa Libotte

Otimização do custo de um tanque cilíndrico de armazenamento, com volume
mínimo garantido, usando:
    - Função de Penalização (para tratar a restrição de desigualdade)
    - Derivadas Parciais Numéricas (Diferenças Finitas Simples Adiantadas)
    - Método do Gradiente (descida de gradiente)

Autores: Helena T. Mussi Gervasio e David Cavalcanti Pinheiro Barbas
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Definição do problema
# ---------------------------------------------------------------------------

def f(r, h):
    """Custo total do tanque (Eq. 1):  f(r, h) = 350*pi*r^2 + 200*pi*r*h"""
    return 350 * np.pi * r**2 + 200 * np.pi * r * h


def volume(r, h):
    """Volume do cilindro: V = pi * r^2 * h"""
    return np.pi * r**2 * h


def g(r, h, v_min=500.0):
    """Restrição de desigualdade (Eq. 2):  g(r, h) = 500 - pi*r^2*h <= 0"""
    return v_min - volume(r, h)


def P(r, h, mu, v_min=500.0):
    """Função auxiliar penalizada (Eq. 3):

        P(r, h) = f(r, h) + mu * [max(0, g(r, h))]^2

    Se o volume for >= 500, a penalidade é zero. Se for menor, o custo
    "explode" proporcionalmente ao quadrado da violação.
    """
    return f(r, h) + mu * max(0.0, g(r, h, v_min)) ** 2


# ---------------------------------------------------------------------------
# 2. Derivadas parciais numéricas (diferenças finitas adiantadas)
# ---------------------------------------------------------------------------

def gradiente_numerico(r, h, mu, delta=1e-5):
    """Gradiente de P em (r, h) por diferenças finitas simples adiantadas
    (Eq. 4 e 5):

        dP/dr ~ [P(r+delta, h) - P(r, h)] / delta
        dP/dh ~ [P(r, h+delta) - P(r, h)] / delta
    """
    P0 = P(r, h, mu)
    dP_dr = (P(r + delta, h, mu) - P0) / delta
    dP_dh = (P(r, h + delta, mu) - P0) / delta
    return np.array([dP_dr, dP_dh])


# ---------------------------------------------------------------------------
# 3. Método do gradiente
# ---------------------------------------------------------------------------
#
# Versão PADRÃO (normalizar=False) -- exatamente as Eq. 6 e 7 do enunciado:
#       r(k+1) = r(k) - alpha * dP/dr        h(k+1) = h(k) - alpha * dP/dh
# Critério de parada (como no enunciado): norma do gradiente muito próxima
# de zero (||grad P|| < tol_grad) ou r e h praticamente sem variação.
#
# Versão NORMALIZADA (normalizar=True) -- a direção de descida é dividida
# pela norma do gradiente:
#       r(k+1) = r(k) - alpha * (dP/dr) / ||grad P||
# Assim alpha é o comprimento do passo no plano (r, h). Serve para valores
# grandes de mu, em que a versão padrão exige alpha minúsculo (instabilidade:
# é preciso alpha < ~1,25e-5/mu). Como o passo tem comprimento fixo, o ponto
# oscila em torno do ótimo e o deslocamento nunca zera; por isso o critério
# de parada compara a média do custo de janelas consecutivas de `janela`
# iterações (variação < tol_custo, em R$).
#
# Nos dois casos há um limite máximo de iterações e um teste de divergência.

def metodo_do_gradiente(r0, h0, mu, alpha, normalizar=False, delta=1e-5,
                        tol_grad=1.0, tol_passo=1e-12,
                        janela=1000, tol_custo=1.0, max_iter=1_000_000):
    """Minimiza P(r, h) pelo método do gradiente.

    Retorna: (r, h) final, histórico do custo, histórico do volume,
    número de iterações, status ("convergiu", "limite" ou "divergiu")
    e o tempo de execução em segundos.
    """
    inicio = time.perf_counter()
    r, h = r0, h0
    hist_custo = [f(r, h)]
    hist_vol = [volume(r, h)]
    media_anterior = None
    status = "limite"

    for k in range(1, max_iter + 1):
        grad = gradiente_numerico(r, h, mu, delta)
        norma = np.hypot(grad[0], grad[1])

        if not np.isfinite(norma):
            status = "divergiu"
            break

        if normalizar:
            if norma < 1e-12:
                status = "convergiu"
                break
            passo = alpha * grad / norma
        else:
            if norma < tol_grad:                 # derivadas ~ 0 (enunciado)
                status = "convergiu"
                break
            passo = alpha * grad

        r_novo = r - passo[0]
        h_novo = h - passo[1]

        # teste de divergência: o passo joga o ponto para dimensões sem
        # sentido físico (r ou h <= 0, ou valores absurdamente grandes)
        if r_novo <= 1e-6 or h_novo <= 1e-6 or r_novo > 1e4 or h_novo > 1e6:
            status = "divergiu"
            r, h = r_novo, h_novo
            break

        deslocamento = np.hypot(r_novo - r, h_novo - h)
        r, h = r_novo, h_novo
        hist_custo.append(f(r, h))
        hist_vol.append(volume(r, h))

        if not normalizar and deslocamento < tol_passo:   # r e h pararam
            status = "convergiu"
            break

        if normalizar and k % janela == 0:
            media = np.mean(hist_custo[-janela:])
            if media_anterior is not None and abs(media - media_anterior) < tol_custo:
                status = "convergiu"
                break
            media_anterior = media

    return (r, h), hist_custo, hist_vol, k, status, time.perf_counter() - inicio


# ---------------------------------------------------------------------------
# 4. Referência analítica (apenas para validar o resultado numérico)
# ---------------------------------------------------------------------------

def solucao_analitica():
    """Com a restrição ativa (V = 500), h = 500/(pi r^2) e
    f(r) = 350*pi*r^2 + 100000/r. Derivando: r^3 = 100000/(700*pi)."""
    r = (100000 / (700 * np.pi)) ** (1 / 3)
    h = 500 / (np.pi * r**2)
    return r, h, f(r, h)


# ---------------------------------------------------------------------------
# 5. Execução: casos base, sensibilidade a mu e alpha, e tempos
# ---------------------------------------------------------------------------

def registro(r, h, n_it, status, tempo, hc=None, **extra):
    """Organiza o resultado de uma execução para tabelas e relatório."""
    d = dict(r=r, h=h, V=volume(r, h), custo=f(r, h), iteracoes=n_it,
             status=status, tempo=tempo, **extra)
    if hc is not None:
        d["faixa"] = [min(hc[-1000:]), max(hc[-1000:])]
    return d


if __name__ == "__main__":

    r0, h0 = 1.0, 1.0   # chute inicial (inviável: V(1,1) = pi << 500)
    NORM_MAX = 200_000  # limite de iterações da versão normalizada
    resultados = {}

    r_an, h_an, f_an = solucao_analitica()
    resultados["analitico"] = dict(r=r_an, h=h_an, custo=f_an, V=volume(r_an, h_an))
    print(f"Referência analítica: r={r_an:.4f}  h={h_an:.4f}  custo=R$ {f_an:.2f}\n")

    def mostrar(rotulo, d):
        print(f"{rotulo} | it={d['iteracoes']:8d} | {d['status']:9s} | r={d['r']:.4f} | "
              f"h={d['h']:.4f} | V={d['V']:.2f} | custo={d['custo']:.2f} | {d['tempo']:.2f}s")

    # ----- Casos base -----
    print("===== CASOS BASE =====")
    (r1, h1), hc1, hv1, n1, st1, t1 = metodo_do_gradiente(r0, h0, 100, 1e-7)
    resultados["base_padrao"] = registro(r1, h1, n1, st1, t1, mu=100, alpha=1e-7)
    mostrar("padrão     mu=1e2 a=1e-7", resultados["base_padrao"])

    (r2, h2), hc2, hv2, n2, st2, t2 = metodo_do_gradiente(
        r0, h0, 1e6, 0.01, normalizar=True, max_iter=NORM_MAX)
    resultados["base_norm"] = registro(r2, h2, n2, st2, t2, hc2, mu=1e6, alpha=0.01)
    mostrar("normalizado mu=1e6 a=0.01", resultados["base_norm"])

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for j, (hc, hv, tit, cor) in enumerate([
            (hc1, hv1, "Gradiente padrão (Eq. 6 e 7): mu=1e2, alpha=1e-7", "#1f5fa8"),
            (hc2, hv2, "Gradiente normalizado: mu=1e6, alpha=0.01", "#2ca02c")]):
        ax[0, j].plot(np.arange(1, len(hc) + 1), hc, color=cor)
        ax[0, j].set_xscale("log")
        ax[0, j].axhline(f_an, color="gray", ls="--", lw=1, label="custo analítico")
        ax[0, j].set_yscale("log")
        ax[0, j].set_ylabel("Custo f(r, h) [R$] (log)")
        ax[0, j].set_title(tit, fontsize=10)
        ax[0, j].legend()
        ax[0, j].grid(alpha=0.3, which="both")
        ax[1, j].plot(np.arange(1, len(hv) + 1), hv, color="#c0392b")
        ax[1, j].set_xscale("log")
        ax[1, j].axhline(500, color="gray", ls="--", lw=1, label="volume mínimo (500 m³)")
        ax[1, j].set_yscale("log")
        ax[1, j].set_xlabel("Iteração (escala log)")
        ax[1, j].set_ylabel("Volume [m³] (log)")
        ax[1, j].legend()
        ax[1, j].grid(alpha=0.3, which="both")
    fig.suptitle("Evolução do custo e do volume nos dois casos base")
    fig.tight_layout()
    fig.savefig("convergencia_caso_base.png", dpi=150)
    plt.close(fig)

    # ----- Sensibilidade a mu: versão padrão (alpha = 1e-5/mu, p/ estabilidade) -----
    print("\n===== PADRÃO: variando mu (alpha = 1e-5/mu) =====")
    resultados["mu_padrao"] = []
    for mu_t in [1, 10, 100, 1e3, 1e4]:
        a_t = 1e-5 / mu_t
        (r_t, h_t), _, _, n_t, st_t, tp_t = metodo_do_gradiente(r0, h0, mu_t, a_t)
        d = registro(r_t, h_t, n_t, st_t, tp_t, mu=mu_t, alpha=a_t)
        resultados["mu_padrao"].append(d)
        mostrar(f"mu={mu_t:.0e} a={a_t:.0e}", d)

    # ----- Sensibilidade a mu: versão normalizada (alpha = 0.01) -----
    print("\n===== NORMALIZADO: variando mu (alpha = 0.01) =====")
    resultados["mu_norm"] = []
    plt.figure(figsize=(7, 5))
    for mu_t in [1, 1e2, 1e4, 1e6]:
        (r_t, h_t), hc_t, _, n_t, st_t, tp_t = metodo_do_gradiente(
            r0, h0, mu_t, 0.01, normalizar=True, max_iter=NORM_MAX)
        d = registro(r_t, h_t, n_t, st_t, tp_t, hc_t, mu=mu_t, alpha=0.01)
        resultados["mu_norm"].append(d)
        mostrar(f"mu={mu_t:.0e}", d)
        plt.plot(hc_t, label=f"mu={mu_t:.0e}")
    plt.axhline(f_an, color="gray", ls="--", lw=1, label="custo analítico")
    plt.ylim(0, 65000)
    plt.xlabel("Iteração")
    plt.ylabel("Custo f(r, h) [R$]")
    plt.title("Gradiente normalizado: efeito de mu (alpha=0.01)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("sensibilidade_mu.png", dpi=150)
    plt.close()

    # ----- Sensibilidade a alpha: versão padrão (mu = 100) -----
    print("\n===== PADRÃO: variando alpha (mu = 100) =====")
    resultados["alpha_padrao"] = []
    for a_t in [1e-3, 1e-5, 1e-6, 1e-7, 1e-8]:
        (r_t, h_t), _, _, n_t, st_t, tp_t = metodo_do_gradiente(r0, h0, 100, a_t)
        d = registro(r_t, h_t, n_t, st_t, tp_t, mu=100, alpha=a_t)
        resultados["alpha_padrao"].append(d)
        mostrar(f"alpha={a_t:.0e}", d)

    # ----- Sensibilidade a alpha: versão normalizada (mu = 1e6) -----
    print("\n===== NORMALIZADO: variando alpha (mu = 1e6) =====")
    resultados["alpha_norm"] = []
    hist_alpha = {}
    for a_t in [0.5, 0.05, 0.01, 0.001, 0.0005]:
        (r_t, h_t), hc_t, _, n_t, st_t, tp_t = metodo_do_gradiente(
            r0, h0, 1e6, a_t, normalizar=True, max_iter=NORM_MAX)
        hist_alpha[a_t] = hc_t
        d = registro(r_t, h_t, n_t, st_t, tp_t, hc_t, mu=1e6, alpha=a_t)
        resultados["alpha_norm"].append(d)
        mostrar(f"alpha={a_t}", d)
        print(f"   faixa do custo (últimas 1000 it.): {d['faixa'][0]:.2f} a {d['faixa'][1]:.2f}")

    cores = {0.5: "#1f77b4", 0.05: "#ff7f0e", 0.01: "#2ca02c",
             0.001: "#9467bd", 0.0005: "#d62728"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for a_t, hc_t in hist_alpha.items():
        ax[0].plot(hc_t[:40000], color=cores[a_t], label=f"alpha={a_t}",
                   lw=0.6 if a_t == 0.5 else 1.4, alpha=0.45 if a_t == 0.5 else 1.0)
    ax[0].axhline(f_an, color="gray", ls="--", lw=1, label="custo analítico")
    ax[0].set_ylim(0, 65000)
    ax[0].set_xlabel("Iteração (primeiras 40 000)")
    ax[0].set_ylabel("Custo f(r, h) [R$]")
    ax[0].set_title("Visão geral")
    ax[0].legend(loc="lower right", fontsize=8)
    ax[0].grid(alpha=0.3)
    for a_t in [0.05, 0.01, 0.001]:
        hc_t = np.array(hist_alpha[a_t])
        ax[1].plot(hc_t, color=cores[a_t], lw=0.5, alpha=0.25)
        mm = np.convolve(hc_t, np.ones(500) / 500, mode="valid")
        ax[1].plot(np.arange(len(mm)) + 250, mm, color=cores[a_t], lw=2.0,
                   label=f"alpha={a_t}")
    ax[1].axhline(f_an, color="gray", ls="--", lw=1, label="custo analítico")
    ax[1].set_ylim(41900, 43500)
    ax[1].set_xlim(0, 40000)
    ax[1].set_xlabel("Iteração (primeiras 40 000)")
    ax[1].set_ylabel("Custo f(r, h) [R$]")
    ax[1].set_title("Zoom: média móvel (linha grossa) e oscilação (faixa clara)")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.suptitle("Gradiente normalizado: efeito de alpha (mu=1e6)")
    fig.tight_layout()
    fig.savefig("sensibilidade_alpha.png", dpi=150)
    plt.close(fig)

    # ----- Exemplo de divergência: Eq. 6 e 7 puras com mu grande -----
    print("\n===== EXEMPLO DE DIVERGÊNCIA: padrão, mu=1e6, alpha=1e-6 =====")
    (r_s, h_s), _, _, n_s, st_s, tp_s = metodo_do_gradiente(r0, h0, 1e6, 1e-6)
    resultados["divergencia"] = registro(r_s, h_s, n_s, st_s, tp_s, mu=1e6, alpha=1e-6)
    mostrar("padrão mu=1e6 a=1e-6", resultados["divergencia"])

    # ----- Limite de estabilidade do passo fixo (padrão) -----
    # Hessiana da penalidade ~ 2*mu*grad(g)grad(g)^T  =>  alpha < 2/(2*mu*||grad g||^2)
    gg = (2 * np.pi * r_an * h_an) ** 2 + (np.pi * r_an**2) ** 2
    resultados["estabilidade"] = dict(norma_grad_g_quad=gg, const=1 / gg)
    print(f"\n||grad g||^2 no ótimo = {gg:.1f}  ->  alpha_max ~ {1/gg:.3e}/mu")

    with open("resultados.json", "w") as arq:
        json.dump(resultados, arq, indent=2, default=float)
