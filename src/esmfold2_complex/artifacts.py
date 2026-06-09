"""Artifact rendering for ESMFold2 complex prediction.

This module owns the matplotlib and py3Dmol-backed artifact rendering:
pLDDT line plot, PAE heatmap, chain-pair iPTM heatmap, and the interactive
HTML viewer. All artifact writers are independent of the inference loop and
accept plain numpy arrays plus chain specs.
"""

from __future__ import annotations

import html as _html
from pathlib import Path

import numpy as np

from esmfold2_complex.contracts import ChainSpec, PLDDT_COLORS, plddt_hex


def _get_plot_modules():
    """Return the matplotlib modules used by the artifact writers.

    Forces the ``Agg`` backend so plotting never tries to open a display
    (the model is usually run on a GPU host with no X server attached).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    return plt, mpatches


def save_structure_views_html(
    cif_text: str,
    chain_specs: list[ChainSpec],
    plddt: np.ndarray,
    output_path: Path,
) -> None:
    """Render the chain-colored and pLDDT-colored 3Dmol.js views to a single HTML."""
    import py3Dmol

    chain_view = py3Dmol.view(width=520, height=420)
    chain_view.addModel(cif_text, "mmcif")
    for chain in chain_specs:
        chain_view.setStyle(
            {"chain": chain.chain_id},
            {"cartoon": {"color": chain.color}},
        )
    chain_view.zoomTo()

    plddt_view = py3Dmol.view(width=520, height=420)
    plddt_view.addModel(cif_text, "mmcif")
    plddt_view.setStyle({}, {})
    for chain in chain_specs:
        chain_plddt = plddt[chain.start : chain.end]
        for resi, score in enumerate(chain_plddt, start=1):
            plddt_view.setStyle(
                {"chain": chain.chain_id, "resi": int(resi)},
                {"cartoon": {"color": plddt_hex(float(score))}},
            )
    plddt_view.zoomTo()

    chain_legend = " &nbsp; ".join(
        (
            f'<span style="color:{chain.color};">&#9632;</span> '
            f"{_html.escape(chain.chain_id)}"
        )
        for chain in chain_specs
    )
    plddt_legend = (
        f'<span style="color:{PLDDT_COLORS["very_low"]};">&#9632;</span> &lt;50'
        f' &nbsp; <span style="color:{PLDDT_COLORS["low"]};">&#9632;</span> 50-70'
        f' &nbsp; <span style="color:{PLDDT_COLORS["confident"]};">&#9632;</span> 70-90'
        f' &nbsp; <span style="color:{PLDDT_COLORS["very_high"]};">&#9632;</span> &gt;90'
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ESMFold2 structure views</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      color: #1f2937;
      background: #f8fafc;
    }}
    .grid {{
      display: flex;
      gap: 24px;
      flex-wrap: wrap;
    }}
    .panel {{
      background: white;
      border: 1px solid #dbe4ee;
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
    }}
    .title {{
      margin: 0 0 10px 0;
      font-size: 18px;
      font-weight: 700;
    }}
    .legend {{
      margin-top: 10px;
      font-size: 13px;
      color: #475569;
    }}
    .note {{
      margin-bottom: 18px;
      font-size: 13px;
      color: #64748b;
    }}
  </style>
</head>
<body>
  <h1 style="margin-top:0;">ESMFold2 structure views</h1>
  <p class="note">If the 3D viewer is blank, refresh the page in a browser with access to the 3Dmol.js CDN.</p>
  <div class="grid">
    <div class="panel">
      <p class="title">Chain identity</p>
      {chain_view._make_html()}
      <div class="legend">{chain_legend}</div>
    </div>
    <div class="panel">
      <p class="title">Confidence (pLDDT)</p>
      {plddt_view._make_html()}
      <div class="legend">{plddt_legend}</div>
    </div>
  </div>
</body>
</html>
"""
    output_path.write_text(page)


def save_plddt_plot(
    plddt: np.ndarray,
    chain_specs: list[ChainSpec],
    output_path: Path,
) -> None:
    """Render the per-residue pLDDT line plot with chain backgrounds and labels."""
    plt, _ = _get_plot_modules()

    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(1, len(plddt) + 1)

    for chain in chain_specs:
        ax.axvspan(
            chain.start + 0.5,
            chain.end + 0.5,
            color=chain.color,
            alpha=0.08,
            linewidth=0,
        )

    for threshold, color in (
        (50, PLDDT_COLORS["very_low"]),
        (70, PLDDT_COLORS["low"]),
        (90, PLDDT_COLORS["confident"]),
    ):
        ax.axhline(threshold, color=color, linestyle="--", linewidth=1)

    ax.plot(x, plddt, color="#1f2937", linewidth=1.6)

    for chain in chain_specs[:-1]:
        ax.axvline(chain.end + 0.5, color="#94a3b8", linewidth=1)

    for chain in chain_specs:
        midpoint = (chain.start + chain.end) / 2 + 0.5
        ax.text(
            midpoint,
            102.5,
            chain.chain_id,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=chain.color,
        )

    ax.set_xlim(1, len(plddt))
    ax.set_ylim(0, 105)
    ax.set_xlabel("Residue index")
    ax.set_ylabel("pLDDT")
    ax.set_title("Per-residue pLDDT")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pae_plot(
    pae: np.ndarray,
    chain_specs: list[ChainSpec],
    output_path: Path,
) -> None:
    """Render the predicted-aligned-error heatmap with chain color bars."""
    plt, mpatches = _get_plot_modules()

    total_len = pae.shape[0]
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        pae,
        cmap="Greens_r",
        vmin=0,
        vmax=30,
        origin="upper",
        aspect="equal",
    )

    bar_thickness = max(total_len * 0.015, 1.0)
    for chain in chain_specs:
        ax.add_patch(
            mpatches.Rectangle(
                (chain.start - 0.5, -bar_thickness - 1),
                chain.length,
                bar_thickness,
                facecolor=chain.color,
                edgecolor="black",
                linewidth=0.5,
                clip_on=False,
            )
        )
        ax.add_patch(
            mpatches.Rectangle(
                (-bar_thickness - 1, chain.start - 0.5),
                bar_thickness,
                chain.length,
                facecolor=chain.color,
                edgecolor="black",
                linewidth=0.5,
                clip_on=False,
            )
        )

    for chain in chain_specs[1:]:
        ax.axhline(chain.start - 0.5, color="black", linewidth=1.2)
        ax.axvline(chain.start - 0.5, color="black", linewidth=1.2)

    legend_handles = [
        mpatches.Patch(facecolor=chain.color, edgecolor="black", label=chain.chain_id)
        for chain in chain_specs
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(len(chain_specs), 5),
        frameon=True,
        fontsize=10,
        title="Chain",
    )

    ax.set_xlim(-bar_thickness - 2, total_len)
    ax.set_ylim(total_len, -bar_thickness - 2)
    ax.set_xlabel("Scored residue")
    ax.set_ylabel("Aligned residue")
    ax.set_title("Predicted aligned error (PAE)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Expected position error (A)")
    cbar.set_ticks([0, 5, 10, 15, 20, 25, 30])
    cbar.ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pair_iptm_plot(
    pair_chains_iptm: np.ndarray,
    chain_specs: list[ChainSpec],
    output_path: Path,
) -> None:
    """Render the chain-pair iPTM heatmap with per-cell numeric annotations."""
    plt, _ = _get_plot_modules()

    labels = [chain.chain_id for chain in chain_specs[: pair_chains_iptm.shape[0]]]
    fig_size = max(4.5, 1.4 * len(labels) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(pair_chains_iptm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Chain")
    ax.set_ylabel("Chain")
    ax.set_title("Chain-pair iPTM")

    for i in range(pair_chains_iptm.shape[0]):
        for j in range(pair_chains_iptm.shape[1]):
            value = float(pair_chains_iptm[i, j])
            text_color = "white" if value >= 0.6 else "black"
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("iPTM")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_seed_artifacts(
    *,
    seed_output_path: Path,
    output_dir: Path,
    mmcif_text: str,
    plddt: np.ndarray,
    pae: np.ndarray | None,
    pair_chains_iptm: np.ndarray | None,
    chain_specs: list[ChainSpec],
) -> dict[str, Path]:
    """Write all per-seed artifacts (HTML + PNGs) and return their paths.

    The CLI writes the mmCIF itself after inference; this helper focuses on the
    HTML/PNG side outputs that live in the same per-seed subdirectory as the
    structure file.
    """
    artifact_paths: dict[str, Path] = {
        "structure_views_html": output_dir / f"{seed_output_path.stem}_structure_views.html",
        "plddt_plot_png": output_dir / f"{seed_output_path.stem}_plddt.png",
    }
    if pae is not None:
        artifact_paths["pae_plot_png"] = output_dir / f"{seed_output_path.stem}_pae.png"
    if pair_chains_iptm is not None:
        artifact_paths["pair_iptm_plot_png"] = (
            output_dir / f"{seed_output_path.stem}_pair_iptm.png"
        )

    save_structure_views_html(
        mmcif_text,
        chain_specs,
        plddt,
        artifact_paths["structure_views_html"],
    )
    save_plddt_plot(
        plddt,
        chain_specs,
        artifact_paths["plddt_plot_png"],
    )
    if pae is not None:
        save_pae_plot(pae, chain_specs, artifact_paths["pae_plot_png"])
    if pair_chains_iptm is not None:
        save_pair_iptm_plot(
            pair_chains_iptm,
            chain_specs,
            artifact_paths["pair_iptm_plot_png"],
        )

    return artifact_paths
