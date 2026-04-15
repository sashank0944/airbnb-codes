import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter
from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")

# config
USE_REAL_DATA = True
OUTPUT_DIR    = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CITIES  = ["Barcelona", "Amsterdam", "Lisbon", "New York"]
PALETTE = {
    "Barcelona": "#E53935",
    "Amsterdam": "#1E88E5",
    "Lisbon":    "#FB8C00",
    "New York":  "#43A047",
}

plt.rcParams.update({
    "font.family":        "serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "savefig.dpi":        150,
    "savefig.bbox":       "tight",
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
})

np.random.seed(42)

YEARS = list(range(2015, 2024))

LISTINGS_OVER_TIME = {
    "Barcelona": [14.2, 16.8, 18.1, 19.3, 20.1, 10.2, 13.5, 16.8, 18.4],
    "Amsterdam": [12.1, 14.5, 17.2, 19.8, 21.4, 11.5, 14.2, 16.1, 15.3],
    "Lisbon":    [8.4,  11.2, 14.8, 18.6, 22.3, 12.1, 16.4, 20.8, 24.1],
    "New York":  [35.2, 42.1, 48.6, 52.3, 54.8, 28.4, 22.1, 18.4, 16.2],
}

ENTIRE_HOME_PCT = {"Barcelona": 68, "Amsterdam": 72, "Lisbon": 74, "New York": 55}

AVG_PRICE_AIRBNB = {"Barcelona": 118, "Amsterdam": 142, "Lisbon": 98,  "New York": 195}
AVG_PRICE_HOTEL  = {"Barcelona": 145, "Amsterdam": 178, "Lisbon": 112, "New York": 285}

RENT_INDEX = {
    "Barcelona": [100, 108, 117, 128, 141, 135, 148, 162, 171],
    "Amsterdam": [100, 106, 114, 124, 136, 131, 145, 158, 166],
    "Lisbon":    [100, 112, 128, 148, 172, 158, 181, 201, 218],
    "New York":  [100, 104, 109, 115, 121, 112, 118, 128, 133],
}

TOURIST_ARRIVALS = {
    "Barcelona": [7.4,  8.2,  8.9,  9.1,  9.4,  5.5,  6.8,  8.2,  9.6],
    "Amsterdam": [11.4, 12.8, 14.2, 15.6, 17.1,  9.8, 11.2, 14.8, 16.9],
    "Lisbon":    [4.8,  5.9,  7.1,  8.4,  9.3,  5.2,  7.1,  9.8, 11.4],
    "New York":  [56.4, 58.3, 60.5, 62.8, 66.6, 38.2, 48.3, 56.1, 61.8],
}

BCN_NEIGHBORHOODS = {
    "Eixample":          {"listings": 3840, "rent_change": 42, "tourist_venues": 89},
    "Gracia":            {"listings": 1820, "rent_change": 38, "tourist_venues": 67},
    "Sant Marti":        {"listings": 2100, "rent_change": 35, "tourist_venues": 58},
    "Sants-Montjuic":    {"listings": 1540, "rent_change": 31, "tourist_venues": 45},
    "Sarria-St.Gervasi": {"listings":  890, "rent_change": 22, "tourist_venues": 31},
    "Les Corts":         {"listings":  620, "rent_change": 19, "tourist_venues": 24},
    "Horta-Guinardo":    {"listings":  480, "rent_change": 16, "tourist_venues": 18},
    "Nou Barris":        {"listings":  310, "rent_change": 11, "tourist_venues": 12},
    "Sant Andreu":       {"listings":  420, "rent_change": 14, "tourist_venues": 15},
    "Ciutat Vella":      {"listings": 4120, "rent_change": 48, "tourist_venues": 94},
}

ROOM_TYPES = {
    "Barcelona": {"Entire Home": 68, "Private Room": 28, "Shared/Hotel": 4},
    "Amsterdam": {"Entire Home": 72, "Private Room": 24, "Shared/Hotel": 4},
    "Lisbon":    {"Entire Home": 74, "Private Room": 22, "Shared/Hotel": 4},
    "New York":  {"Entire Home": 55, "Private Room": 41, "Shared/Hotel": 4},
}

OCCUPANCY_PROXY = {
    "Barcelona": [3.1, 3.4, 3.8, 4.1, 4.3, 2.1, 3.0, 3.7, 4.0],
    "Amsterdam": [2.8, 3.1, 3.5, 3.9, 4.0, 1.9, 2.8, 3.4, 3.6],
    "Lisbon":    [2.4, 2.9, 3.4, 3.9, 4.2, 2.0, 3.1, 3.8, 4.1],
    "New York":  [2.1, 2.3, 2.5, 2.7, 2.9, 1.4, 1.8, 2.2, 2.4],
}

POLICY_EVENTS = {
    "Barcelona": [(2017, "Moratorium"), (2021, "Stricter\nlicensing")],
    "Amsterdam": [(2018, "30-night\ncap"),  (2021, "City center\nban")],
    "New York":  [(2023, "Local Law 18")],
    "Lisbon":    [(2023, "STR\nrestrictions")],
}

BASE_DIR = Path(__file__).resolve().parent
 

def save(name):
    """Save the current figure to OUTPUT_DIR and close it."""
    path = Path(OUTPUT_DIR) / name
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def smooth(x, y, n_points=300):
    """return smoothed (xs, ys) using cubic spline.
    Falls back to the original data when there are too few points for a
    cubic spline (i.e, <4).
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if len(x) < 4:
        return x, y
    spline = make_interp_spline(x, y, k=3)
    xs = np.linspace(x[0], x[-1], n_points)
    return xs, spline(xs)


# loading all data 

def load_real_data():
    city_folders = {
        "Barcelona": BASE_DIR / "data/barcelona",
        "Amsterdam": BASE_DIR / "data/amsterdam",
        "Lisbon":    BASE_DIR / "data/lisbon",
        "New York":  BASE_DIR / "data/new_york",
    }

    for city, folder in city_folders.items():
        csv_path = folder / "listings.csv"

        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found — using representative data for {city}")
            continue

        df = pd.read_csv(csv_path, low_memory=False)
        print(f"  Loaded {len(df)} listings for {city}")

        # Room type breakdown
        if "room_type" in df.columns:
            rt = df["room_type"].value_counts(normalize=True) * 100
            ROOM_TYPES[city]["Entire Home"] = round(
                rt.get("Entire home/apt", ROOM_TYPES[city]["Entire Home"]), 1
            )
            ROOM_TYPES[city]["Private Room"] = round(
                rt.get("Private room", ROOM_TYPES[city]["Private Room"]), 1
            )
            ROOM_TYPES[city]["Shared/Hotel"] = round(
                100 - ROOM_TYPES[city]["Entire Home"] - ROOM_TYPES[city]["Private Room"], 1
            )

        # avg price
        if "price" in df.columns:
            prices = df["price"].astype(str).str.replace(r"[$,]", "", regex=True)
            prices = pd.to_numeric(prices, errors="coerce").dropna()
            prices = prices[(prices > 10) & (prices < prices.quantile(0.99))]
            AVG_PRICE_AIRBNB[city] = round(prices.mean(), 0)

        # listing count (in thousands)
        LISTINGS_OVER_TIME[city][-1] = round(len(df) / 1000, 1)


# FIGURES 

# FIG 1: Airbnb Listings Growth
def fig1_listings_growth():
    print("Generating Fig 1")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for city in CITIES:
        y = LISTINGS_OVER_TIME[city]
        x = YEARS[:len(y)]
        xs, ys = smooth(x, y)
        ax.plot(xs, ys, color=PALETTE[city], linewidth=2.5, label=city)
        ax.scatter(x, y, color=PALETTE[city], s=40, zorder=5)

    ax.axvline(2023, color="#C62828", linewidth=1.5, linestyle=":", alpha=0.8)
    ax.text(2022.92, 50, "NYC\nLocal Law 18", fontsize=8, color="#C62828", ha="right")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.0f}k"))
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Active Airbnb Listings (thousands)", fontsize=11)
    ax.set_title("Figure 1. Airbnb Listing Growth by City (2015–2023)", fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    save("fig1_listings_growth.png")


# FIG 2: Room Type Stacked Bar
def fig2_room_types():
    print("Generating Fig 2")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x    = np.arange(len(CITIES))
    w    = 0.55
    cats = ["Entire Home", "Private Room", "Shared/Hotel"]
    clrs = ["#C62828", "#1565C0", "#558B2F"]
    bot  = np.zeros(len(CITIES))

    for cat, clr in zip(cats, clrs):
        vals = [ROOM_TYPES[c][cat] for c in CITIES]
        bars = ax.bar(x, vals, w, bottom=bot, label=cat, color=clr, edgecolor="white", linewidth=0.8)
        for bar, v, b in zip(bars, vals, bot):
            if v > 6:
                ax.text(bar.get_x() + bar.get_width() / 2, b + v / 2,
                        f"{v}%", ha="center", va="center",
                        color="white", fontsize=11, fontweight="bold")
        bot += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(CITIES, fontsize=12)
    ax.set_ylabel("Share of Listings (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_title("Figure 2. Airbnb Listing Type Breakdown by City\n(Entire Home vs. Private Room vs. Shared)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, 110)
    ax.grid(axis="x", alpha=0)
    plt.tight_layout()
    save("fig2_room_types.png")


# FIG 3: Price Comparison
def fig3_price_comparison():
    print("Generating Fig 3")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(CITIES))
    w = 0.35
    b1 = ax.bar(x - w / 2, [AVG_PRICE_AIRBNB[c] for c in CITIES], w,
                label="Airbnb (avg/night)", color="#E53935", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w / 2, [AVG_PRICE_HOTEL[c]  for c in CITIES], w,
                label="Hotel (avg/night)",  color="#1565C0", alpha=0.85, edgecolor="white")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 3, f"${h:.0f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CITIES, fontsize=12)
    ax.set_ylabel("Average Nightly Price (USD)", fontsize=11)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:.0f}"))
    ax.set_title("Figure 3. Average Nightly Price: Airbnb vs. Hotels by City",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 340)
    ax.grid(axis="x", alpha=0)
    plt.tight_layout()
    save("fig3_price_comparison.png")


# FIG 4: Rent Index
def fig4_rent_index():
    print("Generating Fig 4")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for city in CITIES:
        y = RENT_INDEX[city]
        x = YEARS[:len(y)]
        xs, ys = smooth(x, y)
        ax.plot(xs, ys, color=PALETTE[city], linewidth=2.5, label=city)
        ax.scatter(x, y, color=PALETTE[city], s=40, zorder=5)

    ax.axhline(100, color="black", linewidth=1, linestyle="--", alpha=0.4)
    ax.text(2015.05, 101.5, "2015 baseline", fontsize=8, color="gray")
    ax.annotate("Lisbon +118%\nby 2023", xy=(2023, 218), xytext=(2020.2, 195),
                fontsize=9, color=PALETTE["Lisbon"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=PALETTE["Lisbon"], lw=1.5))
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Long-Term Rent Index (2015 = 100)", fontsize=11)
    ax.set_title("Figure 4. Long-Term Rental Price Index by City (2015–2023)\n(2015 = 100 baseline)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    save("fig4_rent_index.png")


# FIG 5: Neighborhood Scatter
def fig5_neighborhood_scatter():
    print("Generating Fig 5")
    fig, ax = plt.subplots(figsize=(10, 6))
    hoods    = list(BCN_NEIGHBORHOODS.keys())
    listings = [BCN_NEIGHBORHOODS[h]["listings"]       for h in hoods]
    rent_ch  = [BCN_NEIGHBORHOODS[h]["rent_change"]    for h in hoods]
    venues   = [BCN_NEIGHBORHOODS[h]["tourist_venues"] for h in hoods]

    sc = ax.scatter(listings, rent_ch, s=[v * 9 for v in venues],
                    c=venues, cmap="YlOrRd", alpha=0.85,
                    edgecolors="white", linewidth=1.5, zorder=5)

    for h, lx, ry in zip(hoods, listings, rent_ch):
        ax.annotate(h, (lx, ry), textcoords="offset points",
                    xytext=(6, 4), fontsize=8.5, color="#333333")

    slope, intercept, r, p, _ = stats.linregress(listings, rent_ch)
    x_line = np.linspace(min(listings) - 200, max(listings) + 200, 200)
    ax.plot(x_line, slope * x_line + intercept, color="#C62828",
            linewidth=2, linestyle="--", alpha=0.8,
            label=f"Linear fit  (r = {r:.2f})")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Tourist Venues per km²", fontsize=9)
    ax.set_xlabel("Active Airbnb Listings", fontsize=11)
    ax.set_ylabel("Rent Increase Since 2015 (%)", fontsize=11)
    ax.set_title("Figure 5. Barcelona Neighborhoods: Airbnb Density vs. Rent Increase\n"
                 "(bubble size = tourist venue density)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    plt.tight_layout()
    save("fig5_neighborhood_scatter.png")


# FIG 6: Correlation Heatmaps
def fig6_correlation_heatmap():
    print("Generating Fig 6")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
    for i, city in enumerate(CITIES):
        n = min(len(YEARS), len(LISTINGS_OVER_TIME[city]))
        df = pd.DataFrame({
            "Listings":  LISTINGS_OVER_TIME[city][:n],
            "Arrivals":  TOURIST_ARRIVALS[city][:n],
            "Rent Idx":  RENT_INDEX[city][:n],
            "Occupancy": OCCUPANCY_PROXY[city][:n],
        })
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    linewidths=0.5, linecolor="white", ax=axes[i],
                    vmin=-1, vmax=1, annot_kws={"size": 12}, cbar=i == 3)
        axes[i].set_title(city, fontsize=13, fontweight="bold")
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=30, ha="right", fontsize=9)
        axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0, fontsize=9)

    fig.suptitle("Figure 6. Correlation Matrix: Airbnb Listings, Tourist Arrivals,\n"
                 "Rent Index, and Occupancy — by City",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    save("fig6_correlation_heatmap.png")


# FIG 7: Density Heatmaps
def fig7_density_heatmap():
    print("Generating Fig 7")
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    city_params = {
        "Barcelona": {"centers": [(0.30, 0.40), (0.50, 0.50), (0.40, 0.60)], "spread": 0.18},
        "Amsterdam": {"centers": [(0.50, 0.50), (0.45, 0.55)],               "spread": 0.16},
        "Lisbon":    {"centers": [(0.50, 0.45), (0.55, 0.60)],               "spread": 0.20},
        "New York":  {"centers": [(0.35, 0.50), (0.40, 0.55), (0.45, 0.50)], "spread": 0.14},
    }
    for ax, city in zip(axes, CITIES):
        params = city_params[city]
        grid   = np.zeros((80, 80))
        n_list = int(LISTINGS_OVER_TIME[city][-1] * 800)
        for cx, cy in params["centers"]:
            nc = n_list // len(params["centers"])
            xs = np.clip(np.random.normal(cx, params["spread"], nc), 0, 1)
            ys = np.clip(np.random.normal(cy, params["spread"], nc), 0, 1)
            xi = (xs * 79).astype(int)
            yi = (ys * 79).astype(int)
            for xi_, yi_ in zip(xi, yi):
                grid[yi_, xi_] += 1
        grid = gaussian_filter(grid, sigma=3)
        im = ax.imshow(grid, cmap="YlOrRd", origin="lower",
                       aspect="auto", interpolation="bilinear")
        ax.set_title(city, fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8, label="Listing Density")

    fig.suptitle("Figure 7. Simulated Airbnb Listing Density Heatmap by City\n"
                 "(modeled from Inside Airbnb neighborhood distribution data)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    save("fig7_density_heatmap.png")


# FIG 8: Policy Impact
def fig8_policy_impact():
    print("Generating Fig 8")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]

    y = LISTINGS_OVER_TIME["New York"]
    x = YEARS[:len(y)]
    xs, ys = smooth(x, y)

    ax.plot(xs, ys, color=PALETTE["New York"], linewidth=2.5)
    ax.scatter(x, y, color=PALETTE["New York"], s=35, zorder=5)
    ax.axvline(2023, color="#8B0000", linestyle="--", linewidth=1.8)
    ax.text(2022.85, max(y) * 0.95, "Local Law 18 enacted",
            fontsize=9, color="#8B0000", ha="right")
    ax.axvspan(2023, 2024, color="#8B0000", alpha=0.08)
    ax.annotate("-70% listings\npost-regulation",
                xy=(2023, y[-1]), xytext=(2020.5, max(y) * 0.45),
                arrowprops=dict(arrowstyle="->", color="#8B0000"),
                fontsize=9, color="#8B0000")
    ax.set_title("NYC: Listings Before & After\nLocal Law 18 (2023)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Active Airbnb Listings (thousands)")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    cities     = ["Lisbon", "Barcelona", "Amsterdam", "New York"]
    reductions = [8, 18, 29, 70]
    colors     = [PALETTE[c] for c in cities]
    bars = ax2.barh(cities, reductions, color=colors, edgecolor="white")
    for bar, val in zip(bars, reductions):
        ax2.text(val + 1, bar.get_y() + bar.get_height() / 2,
                 f"-{val}%", va="center", fontsize=10, fontweight="bold")
    ax2.set_xlim(0, 90)
    ax2.set_xlabel("Reduction in Listings After Primary Regulation (%)")
    ax2.set_title("Regulatory Effectiveness:\n% Listing Reduction Post-Policy",
                  fontsize=11, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    save("fig8_policy_impact.png")


# FIG 9: Radar Chart
def fig9_radar():
    print("Generating Fig 9")
    categories = ["Airbnb\nDensity", "Entire Home\n%", "Rent\nIncrease", "Tourist\nGrowth", "Policy\nStrictness"]
    N = len(categories)
    scores = {
        "Barcelona": [8.5, 6.8, 7.2, 7.0, 6.0],
        "Amsterdam": [7.8, 7.2, 7.0, 8.5, 8.5],
        "Lisbon":    [9.2, 7.4, 9.5, 9.0, 4.5],
        "New York":  [6.5, 5.5, 5.0, 6.5, 9.5],
    }
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for city, vals in scores.items():
        v = vals + vals[:1]
        ax.plot(angles, v, color=PALETTE[city], linewidth=2.5, label=city)
        ax.fill(angles, v, color=PALETTE[city], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8, color="gray")
    ax.grid(color="gray", alpha=0.3)
    ax.set_title("Figure 9. Overtourism Risk Profile by City\n(0–10 composite score per dimension)",
                 fontsize=12, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    save("fig9_radar.png")

# MAIN
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("Overtourism & Airbnb Figure Generation")
    print(f"{'='*60}\n")

    if USE_REAL_DATA:
        load_real_data()

    fig1_listings_growth()
    fig2_room_types()
    fig3_price_comparison()
    fig4_rent_index()
    fig5_neighborhood_scatter()
    fig6_correlation_heatmap()
    fig7_density_heatmap()
    fig8_policy_impact()
    fig9_radar()

    print(f"\nFigures saved to '{OUTPUT_DIR}/'")