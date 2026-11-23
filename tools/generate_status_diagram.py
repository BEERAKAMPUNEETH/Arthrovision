import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
import os

def create_implementation_status_diagram():
    # Setup figure
    fig, ax = plt.subplots(figsize=(11, 7.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color Palette matching the reference image
    bg_purple = '#EDE9FE'       # soft lavender box fill
    border_purple = '#6D28D9'   # rich purple box border
    dark_purple = '#2E1065'     # text & icon purple
    pill_purple = '#4C1D95'     # "Completed" pill fill
    
    bg_orange = '#FFF7ED'       # soft cream/orange box fill
    border_orange = '#FB923C'   # orange box border
    pill_orange = '#EA580C'     # "Pending" pill fill
    
    # 6 Steps Definition
    steps = [
        {"icon": "</>", "title": "Image Preprocessing & ROI Extraction", "sub": "CLAHE enhancement, joint isolation, patient-disjoint splits"},
        {"icon": "TDA", "title": "Topological Feature Extraction", "sub": "GUDHI Persistent Homology, H0 components & H1 bone loops"},
        {"icon": "AI",  "title": "Deep Visual Backbone & Multimodal Fusion", "sub": "ResNet-50 feature extractor + Late Probability Fusion"},
        {"icon": "STAT","title": "Research Benchmarking & Statistical Audit", "sub": "18-phase orchestration, 5-fold CV, McNemar test (p < 1e-29)"},
        {"icon": "WEB", "title": "Full-Stack Portal & FastAPI Backend", "sub": "Interactive Streamlit clinical UI with Grad-CAM & Barcodes"},
        {"icon": "PACS","title": "Clinical PACS Integration & Cloud Deployment", "sub": "Hospital DICOM integration, prospective multi-center trials"}
    ]

    # Geometry settings
    box_x = 32
    box_w = 64
    box_h = 10.5
    gap = 3.6
    start_y = 86

    box_centers_y = []

    # Draw 6 Steps
    for i, step in enumerate(steps):
        y = start_y - i * (box_h + gap)
        box_centers_y.append(y + box_h / 2)
        is_pending = (i == 5)

        fill_col = bg_orange if is_pending else bg_purple
        stroke_col = border_orange if is_pending else border_purple

        # Box
        rect = patches.FancyBboxPatch(
            (box_x, y), box_w, box_h,
            boxstyle=patches.BoxStyle("Round", pad=0, rounding_size=4.5),
            facecolor=fill_col,
            edgecolor=stroke_col,
            linewidth=2.2,
            zorder=3
        )
        ax.add_patch(rect)

        # Icon circle on the left of the card
        icon_cx = box_x + 6.5
        icon_cy = y + box_h / 2
        
        # Draw custom badge for the icon
        icon_rect = patches.FancyBboxPatch(
            (icon_cx - 3.8, icon_cy - 3.5), 7.6, 7.0,
            boxstyle=patches.BoxStyle("Round", pad=0, rounding_size=1.8),
            facecolor='#FFFFFF',
            edgecolor=stroke_col,
            linewidth=1.4,
            zorder=4
        )
        ax.add_patch(icon_rect)
        
        # Icon text
        icon_symbols = {
            "</>": "</>",
            "TDA": "⟨H₁⟩",
            "AI": "AI",
            "STAT": "📊",
            "WEB": "💻",
            "PACS": "🚀"
        }
        symbol = icon_symbols.get(step["icon"], step["icon"])
        
        ax.text(
            icon_cx, icon_cy, symbol,
            color=dark_purple if not is_pending else '#9A3412',
            fontsize=13, fontweight='black',
            ha='center', va='center', zorder=5, fontfamily='sans-serif'
        )

        # Title text
        ax.text(
            box_x + 13.5, icon_cy + 1.2, step["title"],
            color=dark_purple,
            fontsize=13.5, fontweight='bold',
            ha='left', va='center', zorder=5, fontfamily='sans-serif'
        )
        
        # Subtitle / description
        ax.text(
            box_x + 13.5, icon_cy - 2.2, step["sub"],
            color='#581C87' if not is_pending else '#9A3412',
            fontsize=9.2, fontweight='medium',
            ha='left', va='center', zorder=5, fontfamily='sans-serif'
        )

        # Downward Arrow to next step
        if i < len(steps) - 1:
            arrow_x = box_x + box_w / 2
            arrow_top = y - 0.2
            arrow_bot = y - gap + 0.2
            ax.annotate(
                '', xy=(arrow_x, arrow_bot), xytext=(arrow_x, arrow_top),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=border_purple,
                    lw=2.2,
                    mutation_scale=16,
                    shrinkA=1, shrinkB=1
                ),
                zorder=2
            )

    # -------------------------------------------------------------
    # Left Bracket for Completed Steps (Steps 0 to 4)
    # -------------------------------------------------------------
    top_y = box_centers_y[0] + 4.2
    bot_y = box_centers_y[4] - 4.2
    mid_y = (top_y + bot_y) / 2
    bracket_x = box_x - 3.8
    bracket_back = box_x - 9.0

    # Draw smooth curly bracket curve
    verts = [
        (bracket_x, top_y),
        (bracket_back, top_y),
        (bracket_back, mid_y + 4.5),
        (bracket_back - 3.5, mid_y),
        (bracket_back, mid_y - 4.5),
        (bracket_back, bot_y),
        (bracket_x, bot_y)
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3
    ]
    bracket_path = Path(verts, codes)
    bracket_patch = patches.PathPatch(
        bracket_path, facecolor='none', edgecolor=pill_purple, lw=3.0, zorder=3
    )
    ax.add_patch(bracket_patch)

    # "Completed" Pill Badge
    pill_w = 16.5
    pill_h = 6.2
    pill_cx = bracket_back - 3.5 - pill_w / 2 - 0.5
    pill_cy = mid_y

    comp_pill = patches.FancyBboxPatch(
        (pill_cx - pill_w / 2, pill_cy - pill_h / 2), pill_w, pill_h,
        boxstyle=patches.BoxStyle("Round", pad=0, rounding_size=3.1),
        facecolor=pill_purple,
        edgecolor=pill_purple,
        zorder=6
    )
    ax.add_patch(comp_pill)

    ax.text(
        pill_cx, pill_cy, "Completed",
        color='#FFFFFF', fontsize=12, fontweight='bold',
        ha='center', va='center', zorder=7, fontfamily='sans-serif'
    )

    # -------------------------------------------------------------
    # "Pending" Badge for Step 5
    # -------------------------------------------------------------
    step5_y = box_centers_y[5]
    pend_pill = patches.FancyBboxPatch(
        (pill_cx - pill_w / 2, step5_y - pill_h / 2), pill_w, pill_h,
        boxstyle=patches.BoxStyle("Round", pad=0, rounding_size=3.1),
        facecolor=pill_orange,
        edgecolor=pill_orange,
        zorder=6
    )
    ax.add_patch(pend_pill)

    ax.text(
        pill_cx, step5_y, "Pending",
        color='#FFFFFF', fontsize=12, fontweight='bold',
        ha='center', va='center', zorder=7, fontfamily='sans-serif'
    )

    # Orange Arrow from "Pending" to Box 6
    ax.annotate(
        '', xy=(box_x - 1.2, step5_y), xytext=(pill_cx + pill_w / 2 + 1.2, step5_y),
        arrowprops=dict(
            arrowstyle="-|>",
            color=pill_orange,
            lw=2.5,
            mutation_scale=16,
            shrinkA=1, shrinkB=1
        ),
        zorder=6
    )

    # Output directory
    os.makedirs("results", exist_ok=True)
    out_png = "results/implementation_status_diagram.png"
    plt.tight_layout(pad=1.0)
    plt.savefig(out_png, dpi=300, facecolor='#FFFFFF', bbox_inches='tight')
    plt.close()
    print(f"Diagram successfully saved to {out_png}")

if __name__ == "__main__":
    create_implementation_status_diagram()
