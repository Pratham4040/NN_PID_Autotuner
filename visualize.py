import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def plot_all(temps, powers, errors, losses, gain_history, params_history, setpoint):
    steps = list(range(len(temps)))

    # retune step indices for vertical markers
    retune_steps = [g[0] for g in gain_history[1:]]  # skip step-0 initial gains

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("NN-PID Autotuner — Thermal Chamber Diagnostics", fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.35)

    # ── helper: draw retune markers ──────────────────────────────────────
    def add_retune_markers(ax, ymin, ymax):
        for i, s in enumerate(retune_steps):
            ax.axvline(s, color="#e74c3c", linestyle="--", linewidth=1.0,
                       alpha=0.7, label="PID retune" if i == 0 else "_")

    # ─────────────────────────────────────────────────────────────────────
    # 1. Temperature vs Steps
    # ─────────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, temps, color="#2980b9", linewidth=1.4, label="Chamber Temp (°C)")
    ax1.axhline(setpoint, color="#27ae60", linestyle="-.", linewidth=1.3,
                label=f"Setpoint ({setpoint}°C)")
    add_retune_markers(ax1, min(temps), max(temps))
    ax1.set_title("Chamber Temperature")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ─────────────────────────────────────────────────────────────────────
    # 2. Heater Power vs Steps
    # ─────────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, powers, color="#e67e22", linewidth=1.2, label="Heater Power")
    add_retune_markers(ax2, 0, 1)
    ax2.set_title("Heater Power Output")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Power (0–1)")
    ax2.set_ylim(-0.05, 1.1)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ─────────────────────────────────────────────────────────────────────
    # 3. Control Error vs Steps
    # ─────────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(steps, errors, color="#8e44ad", linewidth=1.2, label="Error (SP − Temp)")
    ax3.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    add_retune_markers(ax3, min(errors), max(errors))
    ax3.set_title("Control Error")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Error (°C)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ─────────────────────────────────────────────────────────────────────
    # 4. NN Training Loss vs Steps
    # ─────────────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    valid_losses = np.array(losses, dtype=float)
    # smooth with rolling window for readability
    window = 20
    smoothed = np.convolve(
        np.where(np.isnan(valid_losses), 0, valid_losses),
        np.ones(window) / window, mode='same'
    )
    ax4.plot(steps, valid_losses, color="#bdc3c7", linewidth=0.7, alpha=0.6, label="Raw loss")
    ax4.plot(steps, smoothed, color="#c0392b", linewidth=1.6, label=f"Smoothed (w={window})")
    add_retune_markers(ax4, 0, max(np.nan_to_num(valid_losses)))
    ax4.set_title("NN Plant-Model Training Loss (MSE)")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("MSE Loss")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale("log")

    # ─────────────────────────────────────────────────────────────────────
    # 5. PID Gains over Time (step-function)
    # ─────────────────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    g_steps = [g[0] for g in gain_history]
    kps = [g[1] for g in gain_history]
    kis = [g[2] for g in gain_history]
    kds = [g[3] for g in gain_history]

    # extend to end of simulation so step-function reaches step 600
    g_steps_ext = g_steps + [len(temps)]
    kps_ext = kps + [kps[-1]]
    kis_ext = kis + [kis[-1]]
    kds_ext = kds + [kds[-1]]

    ax5.step(g_steps_ext, kps_ext, where="post", color="#2980b9", linewidth=1.5, label="Kp")
    ax5.step(g_steps_ext, kis_ext, where="post", color="#27ae60", linewidth=1.5, label="Ki")
    ax5.step(g_steps_ext, kds_ext, where="post", color="#e74c3c", linewidth=1.5, label="Kd")
    for s in retune_steps:
        ax5.axvline(s, color="gray", linestyle=":", linewidth=0.9, alpha=0.6)
    ax5.set_title("PID Gains Over Time")
    ax5.set_xlabel("Step")
    ax5.set_ylabel("Gain Value")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ─────────────────────────────────────────────────────────────────────
    # 6. NN-estimated Plant Parameters (τ and K)
    # ─────────────────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    if params_history:
        p_steps = [p[0] for p in params_history]
        taus = [p[1] for p in params_history]
        Ks = [p[2] for p in params_history]

        ax6_twin = ax6.twinx()
        ax6.scatter(p_steps, taus, color="#16a085", s=60, zorder=5, label="τ (time const.)")
        ax6.plot(p_steps, taus, color="#16a085", linewidth=1.1, alpha=0.5)
        ax6_twin.scatter(p_steps, Ks, color="#d35400", s=60, marker="^", zorder=5, label="K (gain)")
        ax6_twin.plot(p_steps, Ks, color="#d35400", linewidth=1.1, alpha=0.5)

        # true values from simulator for reference
        ax6.axhline(40, color="#16a085", linestyle="--", linewidth=0.9, alpha=0.5, label="True τ=40")
        ax6_twin.axhline(20, color="#d35400", linestyle="--", linewidth=0.9, alpha=0.5, label="True K=20")

        ax6.set_ylabel("τ (s)", color="#16a085")
        ax6_twin.set_ylabel("K (°C/unit)", color="#d35400")

        lines1, labels1 = ax6.get_legend_handles_labels()
        lines2, labels2 = ax6_twin.get_legend_handles_labels()
        ax6.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    else:
        ax6.text(0.5, 0.5, "No parameter estimates\n(need step > 100 with valid a,b)",
                 ha="center", va="center", transform=ax6.transAxes, fontsize=10, color="gray")

    ax6.set_title("NN-Estimated Plant Parameters vs True Values")
    ax6.set_xlabel("Step")
    ax6.grid(True, alpha=0.3)

    # ─────────────────────────────────────────────────────────────────────
    # Legend annotation for retune markers
    # ─────────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.002,
             "Red dashed lines ( — ) = PID retune events triggered by NN parameter estimation",
             ha="center", fontsize=9, color="#c0392b", style="italic")

    plt.savefig("pid_autotuner_diagnostics.png", dpi=150, bbox_inches="tight")
    print("Plot saved to pid_autotuner_diagnostics.png")
    plt.show()
